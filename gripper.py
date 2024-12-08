
from pydrake.all import RigidTransform, AngleAxis, PiecewisePolynomial, PiecewisePose
import numpy as np

def MakePickGripperFrames(X_G, t0=0):
    """
    Generates frames and times for the pick operation using X_G["initial"] and X_G["pick"].
    """
    # Pre-grasp position (negative y in the gripper frame)
    X_GgraspGpregrasp = RigidTransform([0, -0.2, 0])
    X_G["prepick"] = X_G["pick"] @ X_GgraspGpregrasp

    # Interpolate between initial and prepick
    X_GinitialGprepick = X_G["initial"].inverse() @ X_G["prepick"]
    angle_axis = X_GinitialGprepick.rotation().ToAngleAxis()
    X_GinitialGprepare = RigidTransform(
        AngleAxis(angle=angle_axis.angle() / 2.0, axis=angle_axis.axis()),
        X_GinitialGprepick.translation() / 2.0,
    )
    X_G["prepare"] = X_G["initial"] @ X_GinitialGprepare

    # Ensure the gripper is above a certain height to avoid obstacles
    p_G = np.array(X_G["prepare"].translation())
    p_G[2] = max(p_G[2], 0.5)
    X_G["prepare"].set_translation(p_G)

    # Set timing for each frame
    times = {"initial": t0}
    prepare_time = 4.0 * np.linalg.norm(X_G["prepare"].translation() - X_G["prepick"].translation())
    clearance_time = 5.0 * np.linalg.norm(X_G["prepick"].translation() - X_G["pick"].translation())
    
    times["prepare"] = times["initial"] + prepare_time
    times["prepick"] = times["prepare"] + prepare_time
    # Time allocated for gripper closing
    times["pick_start"] = times["prepick"] + 1.5
    times["pick_end"] = times["pick_start"] + 2.0
    X_G["pick_start"] = X_G["pick"]
    X_G["pick_end"] = X_G["pick"]
    times["postpick"] = times["pick_end"] + clearance_time
    X_G["postpick"] = X_G["prepick"]

    return X_G, times


def MakePlaceGripperFrames(X_G, t0=0):
    """
    Generates frames and times for the place operation using X_G["initial"] and X_G["place"].
    key frames: initial, prepare, preplace, place_start, place_end, postplace
    """
    # Pre-place position (negative y in the gripper frame)
    X_GgraspGpregrasp = RigidTransform([0, -0.2, 0])
    X_G["preplace"] = X_G["place"] @ X_GgraspGpregrasp

    # Interpolate between initial and preplace
    X_GinitialGpreplace = X_G["initial"].inverse() @ X_G["preplace"]
    angle_axis = X_GinitialGpreplace.rotation().ToAngleAxis()
    X_GinitialGprepare = RigidTransform(
        AngleAxis(angle=angle_axis.angle() / 2.0, axis=angle_axis.axis()),
        X_GinitialGpreplace.translation() / 2.0,
    )
    X_G["prepare"] = X_G["initial"] @ X_GinitialGprepare

    # Ensure the gripper is above a certain height to avoid obstacles
    p_G = np.array(X_G["prepare"].translation())
    p_G[2] = max(p_G[2], 0.5)
    X_G["prepare"].set_translation(p_G)

    # Set timing for each frame
    times = {"initial": t0}
    prepare_time = 7.0 * np.linalg.norm(X_G["prepare"].translation() - X_G["preplace"].translation())
    times["prepare"] = times["initial"] + prepare_time
    times["preplace"] = times["prepare"] + prepare_time
    # Time allocated for gripper opening
    times["place_start"] = times["preplace"] + 2.0
    times["place_end"] = times["place_start"] + 2.0
    X_G["place_start"] = X_G["place"]
    X_G["place_end"] = X_G["place"]
    times["postplace"] = times["place_end"] + 2.0
    X_G["postplace"] = X_G["preplace"]

    return X_G, times


def MakeGripperCommandTrajectoryPick(times):
    """
    Constructs gripper command trajectory for the pick operation.
    """
    sample_times = []
    positions = []
    open_position = [0.107]
    closed_position = [0.0]

    # Gripper starts open
    sample_times.append(times["initial"])
    positions.append(closed_position)
    sample_times.append(times["prepick"])
    positions.append(open_position)

    # Close gripper to grasp
    sample_times.append(times["pick_start"])
    positions.append(open_position)
    sample_times.append(times["pick_end"])
    positions.append(closed_position)

    # Keep gripper closed after picking
    sample_times.append(times["postpick"])
    positions.append(closed_position)

    return PiecewisePolynomial.FirstOrderHold(sample_times, np.vstack(positions).T)


def MakeGripperCommandTrajectoryPlace(times):
    """
    Constructs gripper command trajectory for the place operation.
    """
    sample_times = []
    positions = []
    closed_position = [0.0]
    open_position = [0.107]

    # Gripper starts closed
    sample_times.append(times["initial"])
    positions.append(closed_position)
    sample_times.append(times["preplace"])
    positions.append(closed_position)

    # Open gripper to release
    sample_times.append(times["place_start"])
    positions.append(closed_position)
    sample_times.append(times["place_end"])
    positions.append(open_position)

    # Keep gripper open after placing
    sample_times.append(times["postplace"])
    positions.append(open_position)

    return PiecewisePolynomial.FirstOrderHold(sample_times, np.vstack(positions).T)

def MakeGripperPoseTrajectoryCustom(X_G, times, keys):
    """
    Constructs a gripper pose trajectory from specified frames and times.
    """
    opened = np.array([0.107])
    closed = np.array([0.0])
    
    sample_times = [times[key] for key in keys]
    poses = [X_G[key] for key in keys]
    gripper_widths = [opened if "pick" in key else closed for key in keys]
    
    return PiecewisePose.MakeLinear(sample_times, poses)