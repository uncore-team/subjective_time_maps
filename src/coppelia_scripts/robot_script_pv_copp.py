import logging
import math
import numpy as np
from pathlib import Path
import sys
from typing import Any
import rl_coppelia

# Locate rl_coppelia source folder and add to sys.path
_pkg_file = Path(rl_coppelia.__file__).resolve()
_src_dir = _pkg_file.parent.parent
if str(_src_dir) not in sys.path:
    sys.path.append(str(_src_dir))

from coppelia_scripts import robot_common


# ----- Simulated self object for CoppeliaSim context -----
# CoppeliaSim provides 'self' internally; this prevents IDE errors
class _CoppeliaSimContext:
    """Mock object to simulate CoppeliaSim internal 'self' context."""

    # Sim API (provided by CoppeliaSim at runtime)
    sim: Any

    # ROS2
    simROS2_flag: bool
    subscriber_twist: Any
    publisher_odometry: Any
    publisher_ground_truth: Any

    # Robot state
    linealSpeed: float
    angularSpeed: float

    # Handles
    robotHandle: int
    robotAlias: str
    footprintHandle: int
    robot_initial_pose_handle: int
    motorLeft: int
    motorRight: int
    laserHandle: int

self: Any = _CoppeliaSimContext()  # type: ignore


# -----------------------------------------
# -- GLOBALS (received from RL side)
# -----------------------------------------
verbose = None
distance_between_wheels = None  # meters
wheel_radius = None             # meters
robot_alias = ""
robot_base_alias = ""
laser_alias = ""

# Uncomment next line for using ROS
# simROS2=require('simROS2')

# Self initialization (script-internal state)
self.simROS2_flag = False
self.linealSpeed = -1
self.angularSpeed = -1
self.robotHandle = -1
self.robotAlias = ""
self.footprintHandle = -1
self.robot_initial_pose_handle = -1
self.motorLeft = -1
self.motorRight = -1
self.laserHandle = -1
self.subscriber_twist = None
self.publisher_odometry = None
self.publisher_ground_truth = None

MAX_SAMPLES = 1000
MIN_SAMPLES = 10


# ----------------------------------------------------------------------
# ------- Path / transform utilities are in robot_common.py -----------
# ----------------------------------------------------------------------


# --------------------------------------------
# --------------- Exposed API ----------------
# --------------------------------------------


def rp_init(n_samples, n_extra_poses, path_name):
    """Initialize path sampling. 

    Args:
        n_samples (int): The initial number of samples to request.
        n_extra_poses (int): Number of additional poses to generate beyond the samples.
        path_name (str): The alias or name of the path object in the simulation.

    Returns:
        tuple: A tuple containing:
            - augmented_pos_samples (list): List of world poses including extra calculations.
            - base_pos_samples (list): List of the base sampled world poses.
    """
    if n_samples < MIN_SAMPLES:
        n_samples = 10
    elif n_samples > MAX_SAMPLES:
        n_samples = 1000
    print(f"Trying to sample the path using a path alias: {path_name}, with {n_samples} samples.")
    path_alias = path_name if path_name else "/RecordedPath"
    path_handle = self.sim.getObject(path_alias)
    augmented_pos_samples, base_pos_samples = robot_common.build_world_poses_from_path_data(
        self.sim, path_handle, n_samples, n_extra_poses
    )

    return augmented_pos_samples, base_pos_samples


def rp_tp(pose): 
    """Teleport the robot to a given pose.

    Args:
        pose: (x, y, yaw) or (x, y, z, yaw). z is ignored (hardcoded to robot height).

    Returns:
        outFloats:  [x, y, yaw] actually set
    """
    if len(pose) == 4:
        x, y, _, yaw = pose
    elif len(pose) == 3:
        x, y, yaw = pose
    else:
        raise ValueError(f"pose must have 3 or 4 elements, got {len(pose)}")
    self.sim.setObjectPosition(self.robotHandle, self.sim.handle_world, [x, y, 0.06969])
    self.sim.setObjectOrientation(self.robotHandle, self.sim.handle_world, [0.0, 0.0, yaw])

    return float(x), float(y), float(yaw)


def cmd_vel(linear, angular):
    """Differential-drive velocity command.

    Args:
        linear: Linear velocity [m/s].
        angular: Angular velocity [rad/s].
    """
    self.angularSpeed = angular
    self.linealSpeed = robot_common.apply_cmd_vel(
        self.sim, self.motorLeft, self.motorRight, linear, angular, distance_between_wheels, wheel_radius
    )


# -------------------------------
# ------- MAIN FUNCTIONS --------
# -------------------------------

def sysCall_init():
    self.sim = require('sim')    # type: ignore

    # HANDLES
    handles = robot_common.initialize_robot_handles(self.sim, robot_alias, robot_base_alias, laser_alias)
    self.robotHandle = handles["robot_handle"]
    self.robotAlias = handles["robot_alias"]
    self.footprintHandle = handles["footprint_handle"]
    self.laserHandle = handles["laser_handle"]
    self.motorLeft = handles["motor_left"]
    self.motorRight = handles["motor_right"]

    # ROS2 PUBLISHERS AND SUBSCRIBERS
    if self.simROS2_flag:
        self.subscriber_twist, self.publisher_odometry, self.publisher_ground_truth = robot_common.setup_ros2_comms(
            simROS2, self.robotAlias  # type: ignore
        )


def sysCall_actuation():
    if self.simROS2_flag:
        robot_common.publish_ros2_tfs(
            self.sim, simROS2, self.footprintHandle, self.robotHandle, self.laserHandle,  # type: ignore
            self.robot_initial_pose_handle, self.robotAlias
        )


def sysCall_sensing():
    pass


def sysCall_cleanup():
    # Do some clean-up here if needed
    pass


