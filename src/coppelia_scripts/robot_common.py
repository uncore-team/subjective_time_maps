"""
Common utilities for CoppeliaSim robot scripts.

This module contains shared functionality between robot_script_copp.py (normal mode)
and robot_script_pv_copp.py (test_map/test_path mode), reducing code duplication.

Functions:
    get_transform_stamped: Build a ROS2-style TransformStamped dictionary.
    apply_cmd_vel: Apply differential-drive velocity commands and return linear speed.
    initialize_robot_handles: Resolve and return common robot object handles.
    setup_ros2_comms: Create ROS2 subscriber and publisher objects.
    publish_ros2_tfs: Send the standard set of TF transforms via ROS2.
    normalize_angle: Normalize an angle to [-pi, pi].
    augment_base_poses: Extend base poses with extra yaw variants.
    build_world_poses_from_path_data: Sample (x, y, z, yaw) poses from a CoppeliaSim Path.
"""

import logging
import math
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


# -----------------------------
# ------- ROS2 HELPERS -------- 
# -----------------------------

def get_transform_stamped(
    sim: Any,
    simROS2: Any,
    obj_handle: int,
    name: str,
    rel_to: int,
    rel_to_name: str,
) -> Dict:
    """Build a ROS2-style TransformStamped dictionary for an object.

    Args:
        sim: CoppeliaSim API object.
        simROS2: CoppeliaSim ROS2 bridge object.
        obj_handle: Handle of the object whose transform we want.
        name: Child frame name.
        rel_to: Handle of the parent frame object (or -1 for world).
        rel_to_name: Parent frame name.

    Returns:
        A dictionary with the fields of a ROS2/TF TransformStamped message.
    """
    t = simROS2.getSimulationTime()
    p = sim.getObjectPosition(obj_handle, rel_to)
    o = sim.getObjectQuaternion(obj_handle, rel_to)
    return {
        "header": {
            "stamp": t,
            "frame_id": rel_to_name,
        },
        "child_frame_id": name,
        "transform": {
            "translation": {"x": p[0], "y": p[1], "z": p[2]},
            "rotation": {"x": o[0], "y": o[1], "z": o[2], "w": o[3]},
        },
    }


def setup_ros2_comms(
    simROS2: Any,
    robot_alias: str,
) -> Tuple[Any, Any, Any]:
    """Create ROS2 subscriber and publishers for odometry and ground truth.

    Args:
        simROS2: CoppeliaSim ROS2 bridge object.
        robot_alias: Robot name, used to build topic names.

    Returns:
        Tuple of (subscriber_twist, publisher_odometry, publisher_ground_truth).

    Raises:
        RuntimeError: If ROS2 is not available.
    """
    try:
        subscriber_twist = simROS2.createSubscription(
            robot_alias + "/cmd_vel", "geometry_msgs/msg/Twist", "cmd_vel"
        )
        publisher_odometry = simROS2.createPublisher(
            robot_alias + "/odom", "geometry_msgs/msg/Pose"
        )
        publisher_ground_truth = simROS2.createPublisher(
            robot_alias + "/ground_truth", "geometry_msgs/msg/Pose"
        )
    except Exception:
        raise RuntimeError(
            "[Robot] simROS2 not available. Unable to set Publishers and Subscribers."
        )
    return subscriber_twist, publisher_odometry, publisher_ground_truth


def publish_ros2_tfs(
    sim: Any,
    simROS2: Any,
    footprint_handle: int,
    robot_handle: int,
    laser_handle: int,
    robot_initial_pose_handle: int,
    robot_alias: str,
) -> None:
    """Send the standard set of TF transforms via ROS2.

    Publishes four transforms:
        1. footprint → map (ground truth)
        2. footprint → odom
        3. base_link → base_footprint
        4. laser_scan → base_link (static)

    Args:
        sim: CoppeliaSim API object.
        simROS2: CoppeliaSim ROS2 bridge object.
        footprint_handle: Handle of the robot base footprint.
        robot_handle: Handle of the robot body.
        laser_handle: Handle of the laser sensor.
        robot_initial_pose_handle: Handle of the initial pose dummy.
        robot_alias: Robot name used for frame naming.

    Raises:
        RuntimeError: If simROS2 is not available.
    """
    try:
        simROS2.sendTransform(
            get_transform_stamped(sim, simROS2, footprint_handle, "base_footprint_" + robot_alias, -1, "map")
        )
        simROS2.sendTransform(
            get_transform_stamped(sim, simROS2, footprint_handle, "base_footprint_" + robot_alias, robot_initial_pose_handle, "odom_" + robot_alias)
        )
        simROS2.sendTransform(
            get_transform_stamped(sim, simROS2, robot_handle, "base_link_" + robot_alias, footprint_handle, "base_footprint_" + robot_alias)
        )
        simROS2.sendTransform(
            get_transform_stamped(sim, simROS2, laser_handle, "laser_scan_" + robot_alias, robot_handle, "base_link_" + robot_alias)
        )
    except Exception:
        raise RuntimeError(
            "[Robot] simROS2 not available. Unable to publish TFs."
        )


# -----------------------------
# ------ GENERAL HELPERS ------
# -----------------------------

def initialize_robot_handles(
    sim: Any,
    robot_alias_param: str,
    robot_base_alias: str,
    laser_alias: str,
) -> Dict[str, Any]:
    """Resolve and return common robot object handles from the CoppeliaSim scene.

    Retrieves handles for the robot body, footprint, laser, and wheel joints.
    Warns if the alias in the scene does not match the one from the params file.

    Args:
        sim: CoppeliaSim API object.
        robot_alias_param: Expected robot alias (from params file), e.g. '/Burger'.
        robot_base_alias: Path alias of the robot base link, e.g. '/Burger/base_link_visual'.
        laser_alias: Path alias of the laser sensor, e.g. '/Burger/Laser'.

    Returns:
        Dictionary with keys:
            - 'robot_handle': int
            - 'robot_alias': str  (as found in the scene)
            - 'footprint_handle': int
            - 'laser_handle': int
            - 'motor_left': int
            - 'motor_right': int
    """
    # Obtain parent object to read the scene alias
    parent_handle = sim.getObject("..")
    robot_alias_scene = sim.getObjectAlias(parent_handle, 3)

    if f"/{robot_alias_scene}" != robot_alias_param:
        logging.warning(
            f"[Robot] Alias from params file '{robot_alias_param}' does not match "
            f"the scene alias '/{robot_alias_scene}'."
        )

    robot_handle = sim.getObject(robot_base_alias)
    footprint_handle = sim.getObject(robot_base_alias)
    laser_handle = sim.getObject(laser_alias)
    motor_left = sim.getObject("/wheel_left_joint")
    motor_right = sim.getObject("/wheel_right_joint")

    return {
        "robot_handle": robot_handle,
        "robot_alias": robot_alias_scene,
        "footprint_handle": footprint_handle,
        "laser_handle": laser_handle,
        "motor_left": motor_left,
        "motor_right": motor_right,
    }


def apply_cmd_vel(
    sim: Any,
    motor_left: int,
    motor_right: int,
    linear: float,
    angular: float,
    distance_between_wheels: float,
    wheel_radius: float,
) -> float:
    """Apply a differential-drive velocity command to the robot wheel joints.

    Converts linear and angular velocity commands into individual wheel joint
    speeds and sends them to the simulator. An optional noise term can be activated 
    for robustness testing.

    Args:
        sim: CoppeliaSim API object.
        motor_left: Handle of the left wheel joint.
        motor_right: Handle of the right wheel joint.
        linear: Desired linear velocity [m/s].
        angular: Desired angular velocity [rad/s].
        distance_between_wheels: Track width of the robot [m].
        wheel_radius: Radius of each wheel [m].

    Returns:
        The actual linear velocity applied (after optional noise) [m/s].
    """
    # Optional noise (set to 0 to disable)
    error_v = 0
    error_w = 0
    linear = linear + error_v
    angular = angular + error_w

    if angular != 0:
        r = linear / angular
        v_left = angular * (r - distance_between_wheels / 2)
        v_right = angular * (r + distance_between_wheels / 2)
    else:
        v_left = linear
        v_right = linear

    sim.setJointTargetVelocity(motor_left, v_left / wheel_radius)
    sim.setJointTargetVelocity(motor_right, v_right / wheel_radius)

    return linear


# ----------------------------------------
# ------ TEST MAP/PATH MODE HELPERS ------
# ----------------------------------------

def normalize_angle(a: float) -> float:
    """Normalize an angle to the range [-pi, pi].

    Args:
        a: Input angle in radians.

    Returns:
        Equivalent angle in [-pi, pi].
    """
    return math.atan2(math.sin(a), math.cos(a))


def augment_base_poses(
    base_poses: List,
    n_extra_poses: int = 0,
    delta_deg: float = 5.0,
    default_z: float = 0.0,
    default_yaw: float = 0.0,
) -> List[Tuple[float, float, float, float]]:
    """Augment base poses with additional yaw variants.

    Accepts each entry in 'base_poses' with shape (x, y), (x, y, z) or
    (x, y, z, yaw). Missing z/yaw values are filled with 'default_z' and
    'default_yaw' respectively. Yaw is normalized to [-pi, pi].

    For each base pose, the function appends the original pose followed by
    'n_extra_poses' variants shifted by +k*delta_deg and 'n_extra_poses'
    variants shifted by -k*delta_deg (for k = 1..n_extra_poses).

    Args:
        base_poses: Sequence of poses as (x,y), (x,y,z) or (x,y,z,yaw).
        n_extra_poses: Number of extra yaw variants per side (default: 0).
        delta_deg: Yaw step in degrees for augmentation (default: 5.0).
        default_z: Z coordinate used when not provided (default: 0.0).
        default_yaw: Yaw angle in degrees used when not provided (default: 0.0).

    Returns:
        List of (x, y, z, yaw) tuples with augmented poses.

    Raises:
        ValueError: If a pose has an unsupported length.
        TypeError: If a pose is not a sequence or contains non-numeric values.
    """
    n_extra = max(0, int(n_extra_poses))
    delta = math.radians(float(delta_deg))
    default_yaw_rads = math.radians(float(default_yaw))
    augmented = []

    for p in base_poses:
        if isinstance(p, (list, tuple)):
            if len(p) == 4:
                x, y, z, yaw = p
            elif len(p) == 3:
                x, y, z = p
                yaw = default_yaw_rads
            elif len(p) == 2:
                x, y = p
                z = default_z
                yaw = default_yaw_rads
            else:
                raise ValueError(
                    f"Base pose must be (x,y), (x,y,z) or (x,y,z,yaw); got length {len(p)}"
                )
        else:
            raise TypeError(f"Base pose must be a sequence, got {type(p)}")

        try:
            x = float(x)
            y = float(y)
            z = float(z)
            yaw = float(yaw)
        except Exception:
            raise TypeError(f"Pose contains non-numeric values: {p}")

        yaw = normalize_angle(yaw)
        augmented.append((x, y, z, yaw))
        for k in range(1, n_extra + 1):
            augmented.append((x, y, z, normalize_angle(yaw + k * delta)))
        for k in range(1, n_extra + 1):
            augmented.append((x, y, z, normalize_angle(yaw - k * delta)))

    return augmented


def build_world_poses_from_path_data(
    sim: Any,
    path_handle: int,
    n_samples: int,
    n_extra_poses: int = 0,
    delta_deg: float = 5.0,
) -> Tuple[List[Tuple[float, float, float, float]], List[Tuple[float, float, float, float]]]:
    """Sample (x, y, z, yaw) poses uniformly from a CoppeliaSim Path object.

    Reads the path position and quaternion data, picks 'n_samples' evenly
    spaced indices, extracts the yaw from each quaternion, and optionally
    augments each base pose with extra yaw variants via 'augment_base_poses'.

    Args:
        sim: CoppeliaSim API object (needed to read path buffer data).
        path_handle: Handle of the CoppeliaSim Path object.
        n_samples: Desired number of uniformly sampled poses.
        n_extra_poses: Extra yaw variants per side (+/-k*delta_deg). Default: 0.
        delta_deg: Angle step in degrees for yaw augmentation. Default: 5.0.

    Returns:
        Tuple of:
            - augmented_poses: List of (x, y, z, yaw) including yaw variants.
            - base_poses: List of (x, y, z, yaw) without augmentation.

    Raises:
        ValueError: If position and quaternion arrays are inconsistent.
    """

    def _yaw_from_quat(qx: float, qy: float, qz: float, qw: float) -> float:
        """Extract yaw (Z rotation) from a quaternion using ZYX convention."""
        s = 2.0 * (qw * qz + qx * qy)
        c = 1.0 - 2.0 * (qy * qy + qz * qz)
        return math.atan2(s, c)

    # Read path data from CoppeliaSim buffer
    path_data = sim.unpackDoubleTable(
        sim.getBufferProperty(path_handle, "customData.PATH")
    )
    m = np.array(path_data).reshape(len(path_data) // 7, 7)
    path_positions_flat = m[:, :3].flatten().tolist()
    path_quaternions_flat = m[:, 3:].flatten().tolist()

    n_pos = len(path_positions_flat) // 3
    n_quat = len(path_quaternions_flat) // 4
    if n_pos != n_quat or n_pos == 0:
        raise ValueError(
            f"Inconsistent path arrays: {n_pos} positions vs {n_quat} quaternions."
        )

    n_original = n_pos

    # Compute uniformly spaced indices
    if n_samples != n_original:
        if n_samples > n_original:
            print(
                f"[Path] Requested {n_samples} samples but path only has "
                f"{n_original}; using all available."
            )
        else:
            print(f"[Path] Resampling {n_original} → {n_samples} poses.")

        indices: List[int] = []
        for k in range(n_samples):
            idx = int(round(k * (n_original - 1) / (n_samples - 1)))
            if not indices or idx != indices[-1]:
                indices.append(idx)
        while len(indices) < n_samples and indices[-1] < n_original - 1:
            indices.append(indices[-1] + 1)
    else:
        indices = list(range(n_original))

    # Build base poses
    base_poses: List[Tuple[float, float, float, float]] = []
    for i in indices:
        x = float(path_positions_flat[3 * i])
        y = float(path_positions_flat[3 * i + 1])
        z = float(path_positions_flat[3 * i + 2])
        qx = float(path_quaternions_flat[4 * i])
        qy = float(path_quaternions_flat[4 * i + 1])
        qz = float(path_quaternions_flat[4 * i + 2])
        qw = float(path_quaternions_flat[4 * i + 3])
        yaw = normalize_angle(_yaw_from_quat(qx, qy, qz, qw))
        base_poses.append((x, y, z, yaw))

    augmented_poses = augment_base_poses(base_poses, n_extra_poses, delta_deg)
    return augmented_poses, base_poses
