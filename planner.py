from enum import Enum
from copy import copy
import numpy as np
import pandas as pd
from pydrake.all import (
    LeafSystem, AbstractValue, InputPortIndex, PiecewisePolynomial,
    PiecewisePose, RigidTransform, RollPitchYaw, Sphere, Rgba,
    MathematicalProgram, Solve, RevoluteJoint, AddMeshcatTriad
)
from setup import BRICK_LENGTH, BRICK_WIDTH, BRICK_HEIGHT, TABLE_HEIGHT
from gripper import MakePickGripperFrames, MakeGripperPoseTrajectoryCustom, MakeGripperCommandTrajectoryPick, MakePlaceGripperFrames, MakeGripperCommandTrajectoryPlace

class PlannerState(Enum):
    WAIT_FOR_OBJECTS_TO_SETTLE = 1
    PICKING_OBJECT = 2
    COM_ESTIMATION_PHASE = 3
    PLACING_OBJECT = 4
    GO_HOME = 5


class Planner(LeafSystem):
    def __init__(self, plant, brick_bodies, meshcat):
        super().__init__()
        self._plant = plant  # Store the plant
        self._brick_bodies = brick_bodies  # List of brick body indices
        self._gripper_body_index = plant.GetBodyByName("body").index()
        self._iiwa_model = plant.GetModelInstanceByName("iiwa")
        self._iiwa_joints = [
            plant.get_joint(j) for j in plant.GetJointIndices(self._iiwa_model)
            if isinstance(plant.get_joint(j), RevoluteJoint)
        ]
        self._iiwa_link_indices = [
            plant.GetBodyByName(f"iiwa_link_{i}", self._iiwa_model).index()
            for i in range(1, 8)
        ]
        num_iiwa_positions = len(self._iiwa_joints)  # Should be 7
        num_iiwa_velocities = num_iiwa_positions
        self._placement_positions = []
        
        self._meshcat = meshcat

        # Inputs
        # self.DeclareAbstractInputPort("body_poses", AbstractValue.Make([RigidTransform()]))
        self._body_poses_index = self.DeclareAbstractInputPort(
            "body_poses", AbstractValue.Make([RigidTransform()])).get_index()
        self._wsg_state_index = self.DeclareVectorInputPort("wsg_state", 2).get_index()
        self._iiwa_position_index = self.DeclareVectorInputPort("iiwa_position", num_iiwa_positions).get_index()
        self._iiwa_state_index = self.DeclareVectorInputPort("iiwa_state", num_iiwa_positions + num_iiwa_velocities).get_index()
        self._external_torque_index = self.DeclareVectorInputPort("external_torque", num_iiwa_positions).get_index()

        # States
        self._mode_index = self.DeclareAbstractState(AbstractValue.Make(PlannerState.WAIT_FOR_OBJECTS_TO_SETTLE))
        self._traj_X_G_index = self.DeclareAbstractState(AbstractValue.Make(PiecewisePose()))
        self._traj_wsg_index = self.DeclareAbstractState(AbstractValue.Make(PiecewisePolynomial()))
        self._times_index = self.DeclareAbstractState(AbstractValue.Make({"initial": 0.0}))
        self._attempts_index = self.DeclareDiscreteState(1)
        self._q0_index = self.DeclareDiscreteState(num_iiwa_positions)  # for q0
        self._traj_q_index = self.DeclareAbstractState(AbstractValue.Make(PiecewisePolynomial()))

        # Outputs
        self.DeclareAbstractOutputPort("X_WG", lambda: AbstractValue.Make(RigidTransform()), self.CalcGripperPose)
        self.DeclareVectorOutputPort("wsg_position", 1, self.CalcWsgPosition)
        self.DeclareAbstractOutputPort("control_mode", lambda: AbstractValue.Make(InputPortIndex(0)), self.CalcControlMode)
        self.DeclareAbstractOutputPort("reset_diff_ik", lambda: AbstractValue.Make(False), self.CalcDiffIKReset)
        self.DeclareVectorOutputPort("iiwa_position_command", num_iiwa_positions, self.CalcIiwaPosition)

        self.DeclareInitializationDiscreteUpdateEvent(self.Initialize)

        self.DeclarePeriodicUnrestrictedUpdateEvent(0.1, 0.0, self.Update)

        # Variables to store data
        self._picked_bricks = []
        self._current_brick_body_index = None
        self._collected_torques = []
        self._collected_positions = []
        self._cumulative_mass = 0.0
        self._cumulative_com = np.array([0.0, 0.0, 0.0])
        self._estimated_com = None
        self._estimated_mass = None
        self._estimated_masses = {}  # Store mass estimates by brick index
        self._estimated_coms = {}

    def Initialize(self, context, discrete_state):
        discrete_state.set_value(
            int(self._q0_index),
            self.get_input_port(int(self._iiwa_position_index)).Eval(context),
        )

    def Update(self, context, state):
        mode = context.get_abstract_state(int(self._mode_index)).get_value()
        current_time = context.get_time()
        times = context.get_abstract_state(int(self._times_index)).get_value()
        # print(f"Current brick: {self._current_brick_body_index}, time: {current_time}, mode: {mode}")
        body_poses = self.get_input_port(self._body_poses_index).Eval(context)
        X_G = body_poses[int(self._gripper_body_index)]

        if mode == PlannerState.WAIT_FOR_OBJECTS_TO_SETTLE:
            if current_time - times["initial"] > 1.0:
                # Transition to picking state
                state.get_mutable_abstract_state(int(self._mode_index)).set_value(PlannerState.PICKING_OBJECT)
                self.PlanPick(context, state)
            return
        elif mode == PlannerState.PICKING_OBJECT:
            # Check if trajectory has finished
            traj_X_G = context.get_abstract_state(int(self._traj_X_G_index)).get_value()
            if not traj_X_G.is_time_in_range(current_time):
                # Check if gripper is closed, i.e., object is grasped
                wsg_state = self.get_input_port(self._wsg_state_index).Eval(context)
                if wsg_state[0] < 0.18:
                    # Gripper is closed, proceed to COM estimation phase
                    state.get_mutable_abstract_state(int(self._mode_index)).set_value(PlannerState.COM_ESTIMATION_PHASE)
                    self._collected_torques = []
                    self._collected_positions = []
                    self.PlanCOMEstimation(context, state)
                else:
                    # Failed to grasp, replan
                    self.PlanPick(context, state)
            return
        elif mode == PlannerState.COM_ESTIMATION_PHASE:
            # Collect torques and positions at each update
            external_torques = self.get_input_port(self._external_torque_index).Eval(context)
            iiwa_state = self.get_input_port(self._iiwa_state_index).Eval(context)
            q = iiwa_state[:len(self._iiwa_joints)]
            self._collected_torques.append(external_torques)
            self._collected_positions.append(q)

            # Check if estimation trajectory has finished
            traj_X_G: PiecewisePose = context.get_abstract_state(int(self._traj_X_G_index)).get_value()
            # X_G: RigidTransform = traj_X_G.GetPose(current_time)
            
            if not traj_X_G.is_time_in_range(current_time):
                # Proceed to estimate COM
                self.EstimateCOM(context, body_poses, X_G)
                # Transition to placing object
                state.get_mutable_abstract_state(int(self._mode_index)).set_value(PlannerState.PLACING_OBJECT)
                self.PlanPlace(context, state)
            return
        elif mode == PlannerState.PLACING_OBJECT:
            # Check if trajectory has finished
            traj_X_G = context.get_abstract_state(int(self._traj_X_G_index)).get_value()
            if not traj_X_G.is_time_in_range(current_time):
                # Mark the brick as placed
                if self._current_brick_body_index is not None:
                    self._picked_bricks.append(self._current_brick_body_index)
                    self._current_brick_body_index = None
                # Check if more bricks to pick
                if len(self._picked_bricks) < len(self._brick_bodies):
                    # Go back to picking state
                    state.get_mutable_abstract_state(int(self._mode_index)).set_value(PlannerState.PICKING_OBJECT)
                    self.PlanPick(context, state)
                else:
                    # All bricks placed, go home
                    state.get_mutable_abstract_state(int(self._mode_index)).set_value(PlannerState.GO_HOME)
                    self.GoHome(context, state)
            return
        elif mode == PlannerState.GO_HOME:
            traj_q = context.get_abstract_state(int(self._traj_q_index)).get_value()
            if not traj_q.is_time_in_range(current_time):
                print("Finished going home.")
            return
        
        if self._current_brick_body_index is not None and self._estimated_com is not None:
            # Get current brick pose
            body_poses = self.get_input_port(self._body_poses_index).Eval(context)
            X_WB = body_poses[self._current_brick_body_index]
            
            # Transform COM offset from body frame to world frame
            estimated_com_W = X_WB.translation() + X_WB.rotation().matrix() @ self._estimated_com
            
            # Update visualization
            self._meshcat.SetTransform(
                f"brick{self._current_brick_body_index}_estimated_com", 
                RigidTransform(estimated_com_W)
            )

    def PlanPick(self, context, state):
        current_time = context.get_time()
        X_G = {
            "initial": self.get_input_port(0).Eval(context)[
                int(self._gripper_body_index)
            ]
        }

        # Select an object to pick
        object_indices = [idx for idx in self._brick_bodies if idx not in self._picked_bricks]

        if not object_indices:
            print("All bricks have been picked and placed.")
            state.get_mutable_abstract_state(int(self._mode_index)).set_value(PlannerState.GO_HOME)
            self.GoHome(context, state)
            return

        object_poses = [self.get_input_port(0).Eval(context)[i] for i in object_indices]

        # Choose the first object
        X_O = object_poses[0]
        self._current_brick_body_index = object_indices[0]

        # Define the pick pose
        X_G["pick"] = RigidTransform(
            X_O.rotation().multiply(RollPitchYaw(-np.pi / 2, 0, np.pi / 2).ToRotationMatrix()),
            X_O.translation() + np.array([0, 0, 0.1])
        )

        # Generate pick frames and times
        X_G, times = MakePickGripperFrames(X_G, t0=current_time)
        traj_X_G = MakeGripperPoseTrajectoryCustom(X_G, times, [
            "initial", "prepare", "prepick", "pick_start", "pick_end", "postpick"
        ])
        traj_wsg_command = MakeGripperCommandTrajectoryPick(times)

        # Update state
        state.get_mutable_abstract_state(int(self._traj_X_G_index)).set_value(traj_X_G)
        state.get_mutable_abstract_state(int(self._traj_wsg_index)).set_value(traj_wsg_command)
        state.get_mutable_abstract_state(int(self._times_index)).set_value(times)

        print(f"Planned pick {self._plant.get_body(self._current_brick_body_index).name()} at time {current_time}.")

    def PlanCOMEstimation(self, context, state):
        current_time = context.get_time()
        X_G_initial = self.get_input_port(0).Eval(context)[
            int(self._gripper_body_index)
        ]

        X_G = {"initial": X_G_initial}
        # poses = ["pose1", "pose2", "pose3", "pose4"]
        poses = ["pose1", "pose2"]
        times = {"initial": current_time}
        delta_t = 2.0

        for i, pose_name in enumerate(poses):
            t = current_time + (i + 1) * delta_t
            times[pose_name] = t
            if pose_name == "pose1":
                # delta_rotation = RollPitchYaw(np.pi / 6, 0.0, 0.0).ToRotationMatrix()
                delta_rotation = RollPitchYaw(0.0, 0.0, 0.0).ToRotationMatrix()
                delta_position = np.array([0.0, 0, -0.05])
            elif pose_name == "pose2":
                # delta_rotation = RollPitchYaw(0.0, np.pi / 6, 0.0).ToRotationMatrix()
                # delta_rotation = RollPitchYaw(np.pi / 6, 0.0, 0.0).ToRotationMatrix()
                delta_rotation = RollPitchYaw(0.0, 0.0, 0.0).ToRotationMatrix()
                delta_position = np.array([0.0, 0, 0.08])
            # elif pose_name == "pose3":
            #     delta_rotation = RollPitchYaw(0.0, 0.0, np.pi / 6).ToRotationMatrix()
            # elif pose_name == "pose4":
            #     delta_rotation = RollPitchYaw(np.pi / 6, np.pi / 6, np.pi / 6).ToRotationMatrix()

            X_G[pose_name] = RigidTransform(
                delta_rotation @ X_G_initial.rotation(),
                X_G_initial.translation() + delta_position
            )

        trajectory_keys = ["initial"] + poses
        sample_times = [times[key] for key in trajectory_keys]
        sample_poses = [X_G[key] for key in trajectory_keys]

        traj_X_G = PiecewisePose.MakeLinear(sample_times, sample_poses)

        # traj_wsg_command = PiecewisePolynomial.ZeroOrderHold(
        #     [current_time, times["pose4"] + 2.0],
        #     np.array([[0.0] * 2] * 2).T
        # )
        traj_wsg_command = PiecewisePolynomial.ZeroOrderHold(
            [current_time, times["pose2"]],
            np.array([0.0] * 2).reshape(1, 2)
        )
            

        state.get_mutable_abstract_state(int(self._traj_X_G_index)).set_value(traj_X_G)
        state.get_mutable_abstract_state(int(self._traj_wsg_index)).set_value(traj_wsg_command)
        state.get_mutable_abstract_state(int(self._times_index)).set_value(times)

        print(f"Planned COM estimation for {self._plant.get_body(self._current_brick_body_index).name()} at time {current_time}.")
    

    def PlanPlace(self, context, state):
        current_time = context.get_time()
        X_G = {
            "initial": self.get_input_port(0).Eval(context)[
                int(self._gripper_body_index)
            ]
        }

        m_n = self._estimated_mass
        d_n = self._estimated_com

        if not hasattr(self, '_cumulative_mass'):
            self._cumulative_mass = 0.0
            self._cumulative_com = 0.0
            self._placement_positions = []

        # Calculate placement position
        if len(self._picked_bricks) == 0:
            x_n = -0.008
            self._cumulative_mass = m_n
            self._cumulative_com = x_n + d_n[0]
        else:
            x_last = self._placement_positions[-1]
            M_prev = self._cumulative_mass
            x_com_prev = self._cumulative_com
            
            # Calculate maximum theoretical position
            x_n_max = float(-M_prev * x_com_prev - m_n * d_n[0]) / m_n
            
            # Apply safety factor and ensure minimum overlap
            safety_factor = 0.95
            min_overlap = 0.16 * BRICK_LENGTH  # 20% overlap with previous brick
            max_step = BRICK_LENGTH - min_overlap
            
            # Calculate bounded position
            x_n = min(x_n_max * safety_factor, x_last + max_step)
            x_n = max(x_n, x_last + min_overlap)  # Ensure forward progress
            
            # Update cumulative properties
            self._cumulative_mass += m_n
            self._cumulative_com = (M_prev * x_com_prev + m_n * (x_n + d_n[0])) / (self._cumulative_mass)


        self._placement_positions.append(x_n)


        base_x = 0.05  # Base offset
        place_x = x_n + base_x
        place_y = -0.5 - d_n[1]  # Center alignment in y
        place_z = TABLE_HEIGHT + (len(self._picked_bricks) + 0.48) * BRICK_HEIGHT

        print(f"Placing {self._plant.get_body(self._current_brick_body_index).name()} at ({place_x}, {place_y}, {place_z})")
        print(f"Stack COM: {self._cumulative_com}, Total mass: {self._cumulative_mass}")
        print(f"Brick COM offset: {d_n}, mass: {m_n}")
        print(f"Placement position relative to support: {x_n}")

        self.VisualizeStackStability(context)
        self.VisualizeOverhangCalculation(context, x_n, m_n, d_n)
    

        # Define the place pose
        X_G["place"] = RigidTransform(
            RollPitchYaw(-np.pi / 2, 0, np.pi / 2),
            [place_x, place_y, place_z]
        )

        # Generate place frames and times
        X_G, times = MakePlaceGripperFrames(X_G, t0=current_time)
        traj_X_G = MakeGripperPoseTrajectoryCustom(X_G, times, [
            "initial", "prepare", "preplace", "place_start", "place_end", "postplace"
        ])
        traj_wsg_command = MakeGripperCommandTrajectoryPlace(times)

        # Update state
        state.get_mutable_abstract_state(int(self._traj_X_G_index)).set_value(traj_X_G)
        state.get_mutable_abstract_state(int(self._traj_wsg_index)).set_value(traj_wsg_command)
        state.get_mutable_abstract_state(int(self._times_index)).set_value(times)

        # Visualize COM after placement
        body_poses = self.get_input_port(self._body_poses_index).Eval(context)
        X_WB = body_poses[self._current_brick_body_index]
        
        # Update estimated COM visualization
        estimated_com_W = X_WB.translation() + X_WB.rotation().matrix() @ d_n
        self._meshcat.SetObject(
            f"brick{self._current_brick_body_index}_estimated_com",
            Sphere(0.01),
            Rgba(0, 1, 0, 1)
        )
        self._meshcat.SetTransform(
            f"brick{self._current_brick_body_index}_estimated_com",
            RigidTransform(estimated_com_W)
        )
        
        # Update actual COM visualization
        plant_context = self._plant.CreateDefaultContext()
        brick_body = self._plant.get_body(self._current_brick_body_index)
        actual_com_B = brick_body.CalcCenterOfMassInBodyFrame(plant_context)
        actual_com_W = X_WB.translation() + X_WB.rotation().matrix() @ actual_com_B
        
        self._meshcat.SetObject(
            f"brick{self._current_brick_body_index}_actual_com",
            Sphere(0.01),
            Rgba(1, 0, 0, 1)
        )
        self._meshcat.SetTransform(
            f"brick{self._current_brick_body_index}_actual_com",
            RigidTransform(actual_com_W)
        )

        print(f"Planned placement trajectory for {self._plant.get_body(self._current_brick_body_index).name()} at time {current_time}.")

    def EstimateCOM(self, context, body_poses, X_G: RigidTransform):
        print(f"Estimating COM for {self._plant.get_body(self._current_brick_body_index).name()}.")
        num_samples = len(self._collected_torques)
        if num_samples < 2:
            print("Not enough data to estimate COM.")
            self._estimated_com = np.zeros(3)
            self._estimated_mass = 0.5  # Default mass
            return

        problem = MathematicalProgram()
        d = problem.NewContinuousVariables(3, "d")  # COM offset in gripper frame
        m = problem.NewContinuousVariables(1, "m")  # mass

        # Set bounds on mass and COM offset
        problem.AddBoundingBoxConstraint(0.05, 0.45, m[0])

        problem.SetInitialGuess(d, np.zeros(3))
        problem.SetInitialGuess(m, [0.5])

        for i in range(num_samples):
            q_i = self._collected_positions[i]
            tau_measured_i = self._collected_torques[i]
            
            # Create context for this configuration
            sample_context = self._plant.CreateDefaultContext()
            self._plant.SetPositions(sample_context, self._iiwa_model, q_i)
            
            # Get brick pose
            X_WB = body_poses[self._current_brick_body_index]
            p_BC_B = d  # COM offset in brick frame
            p_WC_W = X_WB.translation() + X_WB.rotation().matrix() @ p_BC_B
            F_C_W = m[0] * np.array([0, 0, -9.81])
            
            for j, joint in enumerate(self._iiwa_joints):
                joint_frame = joint.frame_on_parent()
                X_WJ = self._plant.CalcRelativeTransform(
                    sample_context,
                    self._plant.world_frame(),
                    joint_frame
                )
                p_WJ = X_WJ.translation()
                r_JC_W = p_WC_W - p_WJ
                axis_W = X_WJ.rotation().matrix() @ joint.revolute_axis()

                expected_torque = axis_W.dot(np.cross(r_JC_W, F_C_W))
                torque_error = expected_torque - tau_measured_i[j]
                problem.AddCost(torque_error**2)

        result = Solve(problem)
        if result.is_success():
            d_sol = result.GetSolution(d)
            m_sol = result.GetSolution(m)[0]
            self._estimated_com = d_sol
            self._estimated_mass = m_sol
            self._estimated_masses[self._current_brick_body_index] = m_sol
            self._estimated_coms[self._current_brick_body_index] = d_sol

            # Now that we have numerical solutions, visualize the estimation process
            for i in range(num_samples):
                q_i = self._collected_positions[i]
                sample_context = self._plant.CreateDefaultContext()
                self._plant.SetPositions(sample_context, self._iiwa_model, q_i)
                
                X_WB = body_poses[self._current_brick_body_index]
                p_WC_W = X_WB.translation() + X_WB.rotation().matrix() @ d_sol
                F_C_W = m_sol * np.array([0, 0, -9.81])
                
                for i in range(num_samples):
                    q_i = self._collected_positions[i]
                    sample_context = self._plant.CreateDefaultContext()
                    self._plant.SetPositions(sample_context, self._iiwa_model, q_i)
                    
                    X_WB = body_poses[self._current_brick_body_index]
                    p_WC_W = X_WB.translation() + X_WB.rotation().matrix() @ d_sol
                    F_C_W = m_sol * np.array([0, 0, -9.81])
                    
                    for j, joint in enumerate(self._iiwa_joints):
                        joint_frame = joint.frame_on_parent()
                        X_WJ = self._plant.CalcRelativeTransform(
                            sample_context,
                            self._plant.world_frame(),
                            joint_frame
                        )
                        p_WJ = X_WJ.translation()
                        r_JC_W = p_WC_W - p_WJ
                        axis_W = X_WJ.rotation().matrix() @ joint.revolute_axis()
                        
                        # Visualize joint frame
                        # AddMeshcatTriad(
                        #     self._meshcat,
                        #     f"joint_{j}_frame",
                        #     length=0.1,
                        #     radius=0.004
                        # )
                        
                        # # Visualize lever arm - correct format for SetLine (3×N matrix)
                        # vertices_lever = np.array([p_WJ, p_WC_W]).T  # This makes it 3×2
                        # self._meshcat.SetLine(
                        #     f"lever_arm_{j}",
                        #     vertices_lever,
                        #     2.0,
                        #     Rgba(0, 0, 1, 0.8)  # Blue
                        # )
                        
                        # Visualize force
                        force_scale = 0.05
                        force_end = p_WC_W + F_C_W * force_scale
                        vertices_force = np.array([p_WC_W, force_end]).T  # 3×2 matrix
                        self._meshcat.SetLine(
                            f"force_{j}",
                            vertices_force,
                            2.0,
                            Rgba(1, 0, 0, 0.8)  # Red
                        )
                        
                        # Visualize joint axis
                        axis_length = 0.1
                        axis_end = p_WJ + axis_W * axis_length
                        vertices_axis = np.array([p_WJ, axis_end]).T  # 3×2 matrix
                        self._meshcat.SetLine(
                            f"joint_axis_{j}",
                            vertices_axis,
                            2.0,
                            Rgba(0, 1, 0, 0.8)  # Green
                        )

            # Visualize COM points
            plant_context = self._plant.CreateDefaultContext()
            brick_body = self._plant.get_body(self._current_brick_body_index)
            actual_com_B = brick_body.CalcCenterOfMassInBodyFrame(plant_context)
            
            # Transform to world frame
            estimated_com_W = X_WB.translation() + X_WB.rotation().matrix() @ d_sol
            actual_com_W = X_WB.translation() + X_WB.rotation().matrix() @ actual_com_B
            
            # Visualize estimated COM (green)
            self._meshcat.SetObject(
                f"brick{self._current_brick_body_index}_estimated_com",
                Sphere(0.01),
                Rgba(0, 1, 0, 1)
            )
            self._meshcat.SetTransform(
                f"brick{self._current_brick_body_index}_estimated_com",
                RigidTransform(estimated_com_W)
            )
            
            # Visualize actual COM (red)
            self._meshcat.SetObject(
                f"brick{self._current_brick_body_index}_actual_com",
                Sphere(0.01),
                Rgba(1, 0, 0, 1)
            )
            self._meshcat.SetTransform(
                f"brick{self._current_brick_body_index}_actual_com",
                RigidTransform(actual_com_W)
            )
            
            print(f"Estimated COM offset in body frame: {d_sol}")
            print(f"Actual COM offset in body frame: {actual_com_B}")
            print(f"Estimated mass: {m_sol}, Actual mass: {brick_body.default_mass()}")
        else:
            print("COM estimation failed")
            self._estimated_com = np.zeros(3)
            self._estimated_mass = 0.5

    def GoHome(self, context, state):
        print("Replanning due to large tracking error.")
        state.get_mutable_abstract_state(int(self._mode_index)).set_value(
            PlannerState.GO_HOME
        )
        q = self.get_input_port(self._iiwa_position_index).Eval(context)
        q0 = copy(context.get_discrete_state(self._q0_index).get_value())
        q0[0] = q[0]  # Safer to not reset the first joint.

        current_time = context.get_time()
        q_traj = PiecewisePolynomial.FirstOrderHold(
            [current_time, current_time + 5.0], np.vstack((q, q0)).T
        )
        state.get_mutable_abstract_state(int(self._traj_q_index)).set_value(q_traj)

    # Remaining methods (CalcGripperPose, CalcWsgPosition, etc.) are unchanged

    def CalcGripperPose(self, context, output):
        traj_X_G = context.get_abstract_state(int(self._traj_X_G_index)).get_value()
        if traj_X_G.get_number_of_segments() > 0 and traj_X_G.is_time_in_range(
            context.get_time()
        ):
            output.set_value(traj_X_G.GetPose(context.get_time()))
            return

        # Command the current position
        output.set_value(
            self.get_input_port(0).Eval(context)[int(self._gripper_body_index)]
        )

    def CalcWsgPosition(self, context, output):
        mode = context.get_abstract_state(int(self._mode_index)).get_value()
        opened = np.array([0.107])
        closed = np.array([0.0])

        if mode == PlannerState.GO_HOME:
            # Command the open position
            output.SetFromVector([closed])
            return

        traj_wsg = context.get_abstract_state(int(self._traj_wsg_index)).get_value()
        if traj_wsg.get_number_of_segments() > 0 and traj_wsg.is_time_in_range(
            context.get_time()
        ):
            output.SetFromVector([traj_wsg.value(context.get_time())[0]])
            return

        # Command the open position
        output.SetFromVector([closed])

    def CalcControlMode(self, context, output):
        mode = context.get_abstract_state(int(self._mode_index)).get_value()

        if mode == PlannerState.GO_HOME:
            output.set_value(InputPortIndex(2))  # Go Home
        else:
            output.set_value(InputPortIndex(1))  # Diff IK

    def CalcDiffIKReset(self, context, output):
        mode = context.get_abstract_state(int(self._mode_index)).get_value()

        if mode == PlannerState.GO_HOME:
            output.set_value(True)
        else:
            output.set_value(False)

    def CalcIiwaPosition(self, context, output):
        traj_q = context.get_abstract_state(int(self._traj_q_index)).get_value()
        if traj_q.is_time_in_range(context.get_time()):
            output.SetFromVector(traj_q.value(context.get_time()))
        else:
            output.SetFromVector(self.get_input_port(self._iiwa_position_index).Eval(context))

    def VisualizeCOMEstimation(self, context, joint_index, p_WJ, p_GoC_W, F_GC_W, r_JC_W, axis_W):
        """Visualize the COM estimation process for a single joint"""
        # Visualize joint frame
        AddMeshcatTriad(
            self._meshcat,
            f"joint_{joint_index}_frame",
            length=0.1,
            radius=0.004
        )
        
        # Visualize lever arm (vector from joint to COM)
        vertices_lever = np.column_stack([p_WJ, p_GoC_W])
        self._meshcat.SetLine(
            f"lever_arm_{joint_index}",
            vertices_lever,
            Rgba(0, 0, 1, 0.8),  # Blue
            2.0
        )
        
        # Visualize gravitational force
        force_scale = 0.05  # Scale factor for visualization
        force_end = p_GoC_W + F_GC_W * force_scale
        self._meshcat.SetLine(
            f"force_{joint_index}",
            [p_GoC_W, force_end],
            Rgba(1, 0, 0, 0.8),  # Red
            2.0
        )
        
        # Visualize joint axis
        axis_length = 0.1
        axis_end = p_WJ + axis_W * axis_length
        self._meshcat.SetLine(
            f"joint_axis_{joint_index}",
            [p_WJ, axis_end],
            Rgba(0, 1, 0, 0.8),  # Green
            2.0
        )


    def VisualizeStackStability(self, context):
        """
        Visualize the stack's stability metrics
        The visualization showing the stack's center of mass in the scene tree is yellow, and the support region is cyan.
        """
        # Get current stack properties
        if hasattr(self, '_cumulative_mass') and self._cumulative_mass > 0:
            # Visualize stack COM
            stack_com_height = TABLE_HEIGHT + BRICK_HEIGHT * len(self._picked_bricks) / 2
            stack_com_point = np.array([self._cumulative_com, -0.5, stack_com_height])
            
            # Stack COM marker (yellow sphere)
            self._meshcat.SetObject(
                "stack_com",
                Sphere(0.02),
                Rgba(1, 1, 0, 0.8)
            )
            self._meshcat.SetTransform(
                "stack_com",
                RigidTransform(stack_com_point)
            )
            
            # Support region (base brick)
            if len(self._placement_positions) > 0:
                base_x = self._placement_positions[0]
                points = np.array([
                    [base_x - BRICK_LENGTH/2, -0.5 - BRICK_WIDTH/2, TABLE_HEIGHT],
                    [base_x + BRICK_LENGTH/2, -0.5 - BRICK_WIDTH/2, TABLE_HEIGHT],
                    [base_x + BRICK_LENGTH/2, -0.5 + BRICK_WIDTH/2, TABLE_HEIGHT],
                    [base_x - BRICK_LENGTH/2, -0.5 + BRICK_WIDTH/2, TABLE_HEIGHT],
                    [base_x - BRICK_LENGTH/2, -0.5 - BRICK_WIDTH/2, TABLE_HEIGHT],  # Close the loop
                ]).T  # Make it 3×N
                
                self._meshcat.SetLine(
                    "support_region",
                    points,
                    2.0,
                    Rgba(0, 1, 1, 0.5)  # Cyan
                )


    def VisualizeOverhangCalculation(self, context, x_n, m_n, d_n):
        """Visualize the overhang calculation for the current brick"""
        if self._current_brick_body_index is not None:
            # Current brick position
            X_WB = self.get_input_port(self._body_poses_index).Eval(context)[
                self._current_brick_body_index
            ]
            
            # Visualize maximum theoretical position
            max_pos = np.array([x_n, -0.5, TABLE_HEIGHT + len(self._picked_bricks) * BRICK_HEIGHT])
            self._meshcat.SetObject(
                "max_overhang_marker",
                Sphere(0.01),
                Rgba(1, 0.5, 0, 0.8)  # Orange
            )
            self._meshcat.SetTransform(
                "max_overhang_marker",
                RigidTransform(max_pos)
            )
            
    def evaluate_estimation_accuracy(self):
        """
        Evaluate the accuracy of COM and mass estimation.
        Returns a dictionary with statistical metrics.
        """
        results = {
            'brick_id': [],
            'true_mass': [],
            'estimated_mass': [],
            'mass_error': [],
            'mass_error_percent': [],
            'true_com': [],
            'estimated_com': [],
            'com_error_euclidean': [],
            'com_error_by_axis': [],
        }
        
        for brick_idx in self._picked_bricks:
            brick_body = self._plant.get_body(brick_idx)
            plant_context = self._plant.CreateDefaultContext()
            
            # Get true values
            true_mass = brick_body.default_mass()
            true_com = brick_body.CalcCenterOfMassInBodyFrame(plant_context)
            
            # Get stored estimates
            estimated_mass = self._estimated_masses[brick_idx] if hasattr(self, '_estimated_masses') else 0
            estimated_com = self._estimated_coms[brick_idx] if hasattr(self, '_estimated_coms') else np.zeros(3)
            
            # Calculate errors
            mass_error = estimated_mass - true_mass
            mass_error_percent = (mass_error / true_mass) * 100
            com_error_euclidean = np.linalg.norm(estimated_com - true_com)
            com_error_by_axis = estimated_com - true_com
            
            # Store results
            results['brick_id'].append(brick_body.name())
            results['true_mass'].append(true_mass)
            results['estimated_mass'].append(estimated_mass)
            results['mass_error'].append(mass_error)
            results['mass_error_percent'].append(mass_error_percent)
            results['true_com'].append(true_com)
            results['estimated_com'].append(estimated_com)
            results['com_error_euclidean'].append(com_error_euclidean)
            results['com_error_by_axis'].append(com_error_by_axis)
        
        # Calculate summary statistics
        summary = {
            'mass': {
                'mean_error': np.mean(results['mass_error']),
                'std_error': np.std(results['mass_error']),
                'mean_percent_error': np.mean(results['mass_error_percent']),
                'rmse': np.sqrt(np.mean(np.array(results['mass_error'])**2)),
            },
            'com': {
                'mean_euclidean_error': np.mean(results['com_error_euclidean']),
                'std_euclidean_error': np.std(results['com_error_euclidean']),
                'rmse_by_axis': np.sqrt(np.mean(np.array(results['com_error_by_axis'])**2, axis=0)),
                'max_error': np.max(results['com_error_euclidean']),
            }
        }
        
        # Print detailed report
        print("\nEstimation Accuracy Report")
        print("=" * 50)
        print("\nPer-Brick Results:")
        print("-" * 30)
        for i in range(len(results['brick_id'])):
            print(f"\n{results['brick_id'][i]}:")
            print(f"Mass (kg):")
            print(f"  True: {results['true_mass'][i]:.4f}")
            print(f"  Estimated: {results['estimated_mass'][i]:.4f}")
            print(f"  Error: {results['mass_error'][i]:.4f} ({results['mass_error_percent'][i]:.2f}%)")
            print(f"COM (m):")
            print(f"  True: [{results['true_com'][i][0]:.4f}, {results['true_com'][i][1]:.4f}, {results['true_com'][i][2]:.4f}]")
            print(f"  Estimated: [{results['estimated_com'][i][0]:.4f}, {results['estimated_com'][i][1]:.4f}, {results['estimated_com'][i][2]:.4f}]")
            print(f"  Euclidean Error: {results['com_error_euclidean'][i]:.4f}")
        
        print("\nSummary Statistics:")
        print("-" * 30)
        print("\nMass Estimation:")
        print(f"Mean Error: {summary['mass']['mean_error']:.4f} kg")
        print(f"Standard Deviation of Error: {summary['mass']['std_error']:.4f} kg")
        print(f"Mean Percentage Error: {summary['mass']['mean_percent_error']:.2f}%")
        print(f"RMSE: {summary['mass']['rmse']:.4f} kg")
        
        print("\nCOM Estimation:")
        print(f"Mean Euclidean Error: {summary['com']['mean_euclidean_error']:.4f} m")
        print(f"Standard Deviation of Euclidean Error: {summary['com']['std_euclidean_error']:.4f} m")
        print(f"RMSE by axis [x, y, z]: [{summary['com']['rmse_by_axis'][0]:.4f}, {summary['com']['rmse_by_axis'][1]:.4f}, {summary['com']['rmse_by_axis'][2]:.4f}] m")
        print(f"Maximum Error: {summary['com']['max_error']:.4f} m")
        
        return results, summary
