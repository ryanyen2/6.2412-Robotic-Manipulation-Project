import logging
import numpy as np
import pandas as pd
from pydrake.all import (
    DiagramBuilder, MeshcatVisualizer, StartMeshcat, PortSwitch, ConstantVectorSource,
    Simulator, ModelInstanceIndex, RigidTransform, RotationMatrix
)
from manipulation import running_as_notebook
from manipulation.scenarios import AddIiwaDifferentialIK
from manipulation.meshcat_utils import AddMeshcatTriad, StopButton

from setup import StationSetup
from planner import Planner
from setup import BRICK_LENGTH, BRICK_WIDTH, BRICK_HEIGHT, TABLE_HEIGHT, rng


class NoDiffIKWarnings(logging.Filter):
    def filter(self, record):
        return not record.getMessage().startswith("Differential IK")

logging.getLogger("drake").addFilter(NoDiffIKWarnings())

def main():
    meshcat = StartMeshcat()
    
    builder = DiagramBuilder()
    station, plant, scene_graph = StationSetup(meshcat)
    station_system = builder.AddSystem(station)
    
    MeshcatVisualizer.AddToBuilder(
        builder,
        station_system.GetOutputPort("query_object"),
        meshcat
    )
    
    brick_bodies = []
    for index in range(plant.num_model_instances()):
        model_instance_index = ModelInstanceIndex(index)
        model_name = plant.GetModelInstanceName(model_instance_index)
        if model_name.startswith("brick"):
            brick_bodies.append(plant.GetBodyIndices(model_instance_index)[0])

    planner = builder.AddSystem(Planner(plant, brick_bodies, meshcat))
    # Connect the external torque
    builder.Connect(
        station_system.GetOutputPort("iiwa_torque_external"),
        planner.GetInputPort("external_torque")
    )

    # Connect the iiwa state (positions and velocities)
    builder.Connect(
        station_system.GetOutputPort("iiwa_state_estimated"),
        planner.GetInputPort("iiwa_state")
    )
    builder.Connect(
        station_system.GetOutputPort("body_poses"), planner.GetInputPort("body_poses")
    )
    builder.Connect(
        station_system.GetOutputPort("wsg_state_measured"),
        planner.GetInputPort("wsg_state"),
    )
    builder.Connect(station_system.GetOutputPort("iiwa_position_measured"),
                    planner.GetInputPort("iiwa_position"))

    robot = station.GetSubsystemByName("iiwa_controller").get_multibody_plant_for_control()

    # Set up differential inverse kinematics.
    diff_ik = AddIiwaDifferentialIK(builder, robot)
    builder.Connect(planner.GetOutputPort("X_WG"), diff_ik.get_input_port(0))
    builder.Connect(
        station_system.GetOutputPort("iiwa_state_estimated"),
        diff_ik.GetInputPort("robot_state"),
    )
    
    builder.Connect(
        planner.GetOutputPort("reset_diff_ik"),
        diff_ik.GetInputPort("use_robot_state"),
    )

    builder.Connect(
        planner.GetOutputPort("wsg_position"),
        station_system.GetInputPort("wsg_position"),
    )

    # The DiffIK and the direct position-control modes go through a PortSwitch
    switch = builder.AddSystem(PortSwitch(7))
    builder.Connect(diff_ik.get_output_port(), switch.DeclareInputPort("diff_ik"))
    builder.Connect(
        planner.GetOutputPort("iiwa_position_command"),
        switch.DeclareInputPort("position"),
    )
    builder.Connect(switch.get_output_port(), station_system.GetInputPort("iiwa_position"))
    builder.Connect(
        planner.GetOutputPort("control_mode"),
        switch.get_port_selector_input_port(),
    )
    
    
    # apply force limit to the wsg
    wsg_force_source = builder.AddNamedSystem(
        "wsg_force_limit", ConstantVectorSource([220.0])
    )

    builder.Connect(
        wsg_force_source.get_output_port(), station.GetInputPort("wsg_force_limit")
    )

    builder.AddSystem(StopButton(meshcat))
    diagram = builder.Build()
    simulator = Simulator(diagram)
    
    # Initialize simulation
    context = simulator.get_context()
    plant_context = plant.GetMyMutableContextFromRoot(context)

    z = BRICK_HEIGHT / 2
    positions = []
    for body_index in plant.GetFloatingBaseBodies():
        body = plant.get_body(body_index)
        if body.name().startswith("brick"):
            max_attempts = 10
            for attempt in range(max_attempts):
                tf = RigidTransform(
                    # UniformlyRandomRotationMatrix(generator),
                    RotationMatrix.Identity(),
                    [rng.uniform(0.38, 0.58), rng.uniform(-0.35, 0.3), z],
                )
                position = tf.translation()
                # Ensure the brick does not go too high
                # if position[2] > 1.0:
                #     continue
                # Check for overlap with existing bricks
                no_overlap = True
                for pos in positions:
                    dx = abs(position[0] - pos[0])
                    dy = abs(position[1] - pos[1])
                    dz = abs(position[2] - pos[2])
                    if dx < BRICK_WIDTH and dy < BRICK_LENGTH and dz < BRICK_HEIGHT:
                        no_overlap = False
                        break
                if no_overlap:
                    positions.append(position)
                    plant.SetFreeBodyPose(plant_context, body, tf)
                    z += 0.02
                    break
            else:
                print(f"Could not place brick {body.name()} without overlap after {max_attempts} attempts")
            

    simulator.AdvanceTo(0.1)
    meshcat.Flush()  # Wait for the large object meshes to get to meshcat.
    simulator.set_target_realtime_rate(1.0)
    simulator.AdvanceTo(np.inf)
    
    # Evaluate results
    results, summary = planner.evaluate_estimation_accuracy()
    df = pd.DataFrame({
        'Brick': results['brick_id'],
        'True Mass': results['true_mass'],
        'Estimated Mass': results['estimated_mass'],
        'Mass Error %': results['mass_error_percent'],
        'COM Error (m)': results['com_error_euclidean']
    })
    df.to_csv('estimation_results.csv', mode='a', header=False, index=False)

if __name__ == "__main__":
    main()
    
    
    