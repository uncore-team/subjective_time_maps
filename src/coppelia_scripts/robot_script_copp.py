"""
Low-level robot control script for CoppeliaSim.

This script runs on the robot side and is responsible for:

    * Receiving velocity commands via the 'cmd_vel' function and converting
      them into wheel joint velocities (differential drive).
    * Optionally publishing TFs and odometry via ROS2 (when enabled).
    * Drawing the robot path in different colors depending on the current
      action, using 'draw_path'.
    * Logging wheel velocities and LAT vs distance in CoppeliaSim graphs
      when 'verbose == 3'.
"""

import logging
import math
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

    # Sim API (provided by CoppeliaSim)
    sim: Any

    # ROS2
    simROS2_flag: bool
    subscriber_twist: Any
    publisher_odometry: Any
    publisher_ground_truth: Any

    # Robot state
    linearSpeed: float
    angularSpeed: float

    # Handles
    robotHandle: int
    robotAlias: str
    footprintHandle: int
    robot_initial_pose_handle: int
    motorLeft: int
    motorRight: int
    laserHandle: int
    curve1Handle: int
    curve2Handle: int
    inner_target: int
    handle_laser_get_observation_script: Any

    # Graph handles and timing
    graph_vel: int
    graph_lat: int
    graphStartTime: float

    # Graph stream handles
    leftMotorCurveX: Any
    leftMotorCurveY: Any
    rightMotorCurveX: Any
    rightMotorCurveY: Any
    latX: Any
    latY: Any
    distanceX: Any
    distanceY: Any

    # Path drawing
    color_dict: list
    poseGtList: dict
    current_color_id: int

    # Testing laser
    laser_script_handle: Any
    poses: list
    pose_idx: int

self: Any = _CoppeliaSimContext()  # type: ignore


# -----------------------------------------
# ----- GLOBALS (received from RL side)----
# -----------------------------------------
verbose = None
distance_between_wheels = None  # meters
wheel_radius = None             # meters
robot_alias = ""
robot_base_alias = ""
laser_alias = ""

# Parameters dict injected by _send_params_dict
params_env = {}

# Uncomment next line for using ROS, and enable simROS2_flag
# simROS2=require('simROS2')

# Self initialization (script-internal state)
self.simROS2_flag = False
self.linearSpeed = -1
self.angularSpeed = -1
self.robotHandle = -1
self.robotAlias = ""
self.footprintHandle = -1
self.robot_initial_pose_handle = -1
self.motorLeft = -1
self.motorRight = -1
self.laserHandle = -1
self.curve1Handle = -1
self.curve2Handle = -1
self.inner_target = -1
self.handle_laser_get_observation_script = None
self.graph_vel = -1
self.graph_lat = -1
self.graphStartTime = 0
self.leftMotorCurveX = None
self.leftMotorCurveY = None
self.rightMotorCurveX = None
self.rightMotorCurveY = None
self.latX = None
self.latY = None
self.distanceX = None
self.distanceY = None
self.poseGtList = {}
self.current_color_id = 1
self.subscriber_twist = None
self.publisher_odometry = None
self.publisher_ground_truth = None
self.laser_script_handle = None
self.poses = []
self.pose_idx = 0
self.color_dict = [
    [1, 0, 0],        # Red
    [0.4, 0.0, 0.3],  # Purple
    [0, 1, 0],        # Green
    [0.6, 0.3, 0.0],  # Brown
    [0, 0, 1],        # Blue
    [1, 1, 0],        # Yellow
    [1, 0, 1],        # Magenta
    [0, 1, 1],        # Cyan
]


# -----------------------------------------
# -- UTILITIES 
# -----------------------------------------

def resetGraphWithOffset():
    """Reset both graphs and restart their time baseline."""
    self.sim.resetGraph(self.graph_vel)
    self.sim.resetGraph(self.graph_lat)


# Create streams and curves
def createCurves(
    graph,
    x1Label, y1Label, x1Units, y1Units,
    x2Label, y2Label, x2Units, y2Units,
    curve1Name, curve2Name
):
    """
    Create two curves with their streams in a CoppeliaSim graph.

    Args:
        graph: Graph object handle.
        x1Label, y1Label: Labels for the first curve axes.
        x1Units, y1Units: Units for the first curve axes.
        x2Label, y2Label: Labels for the second curve axes.
        x2Units, y2Units: Units for the second curve axes.
        curve1Name: Display name of the first curve.
        curve2Name: Display name of the second curve.

    Returns:
        Tuple (curve1X, curve1Y, curve2X, curve2Y) of stream handles.
    """
    # Initialize graphStartTime
    self.graphStartTime = self.sim.getSimulationTime()

    # Create streams for first curve:
    curve1X = self.sim.addGraphStream(graph, x1Label, x1Units, 1)
    curve1Y = self.sim.addGraphStream(graph, y1Label, y1Units, 1)

    # Create curve
    self.curve1Handle = self.sim.addGraphCurve(graph, curve1Name, 2, [curve1X, curve1Y], [0, 0], y1Units, 0, [1, 0, 0])

    # Create streams for second curve:
    curve2X = self.sim.addGraphStream(graph, x2Label, x2Units, 1)
    curve2Y = self.sim.addGraphStream(graph, y2Label, y2Units, 1)

    # Create curve for right motor: x = time, y = velocity
    self.curve2Handle = self.sim.addGraphCurve(graph, curve2Name, 2, [curve2X, curve2Y], [0, 0], y2Units, 0, [0, 1, 0])

    return curve1X, curve1Y, curve2X, curve2Y


# -----------------------------------------
# -- PUBLIC API
# -----------------------------------------


def cmd_vel(linear, angular):
    """
    Differential-drive velocity command.

    Converts linear and angular velocity commands into wheel joint speeds.

    Args:
        linear: Linear velocity [m/s].
        angular: Angular velocity [rad/s].
    """
    self.angularSpeed = angular
    self.linearSpeed = robot_common.apply_cmd_vel(
        self.sim, self.motorLeft, self.motorRight, linear, angular, distance_between_wheels, wheel_radius
    )


def draw_path(linear, angular, color_id):
    """
    Draw the robot path in different colors (on per timestep).

    Args:
        linear: Linear velocity command [m/s].
        angular: Angular velocity command [rad/s].
        color_id: Integer index to select a path color from 'color_dict'.

    Behavior:
        * If (linear == 0 and angular == 0): clears all stored paths and
          resets the graphs (used as a "reset" command).
        * Otherwise: appends the current robot pose to a color-specific
          cyclic drawing object, creating a colored trajectory.
    """
    # If a reset is received, it clears the previous path (all colors) and finishes the function
    if linear == 0 and angular == 0:
        for poseGt in self.poseGtList.values():
            self.sim.addDrawingObjectItem(poseGt, None)
            self.sim.removeDrawingObject(poseGt)
        self.poseGtList = {}  # Clear the list
        self.current_color_id = 1  # Reset the color
        resetGraphWithOffset()
        return

    # If there is movement, it draws it.
    # In case no color is received, it will choose the first one from the color dictionary
    color_id = color_id or 1
    # Start a new cycle in the dictionary if its capacity is exceeded
    color_id = (color_id - 1) % len(self.color_dict) + 1

    # If the drawing object for the current color doesn't exist, let's create it.
    if color_id not in self.poseGtList:
        color = self.color_dict[color_id - 1]
        self.poseGtList[color_id] = self.sim.addDrawingObject(self.sim.drawing_cyclic, 3, 0, -1, 5000, color, None, None, color)

    # Draw the current point in the trajectory
    p = self.sim.getObjectPosition(self.footprintHandle, -1)
    self.sim.addDrawingObjectItem(self.poseGtList[color_id], p)
    logging.debug('[draw_path] point:', p)

    self.current_color_id = color_id  # Update current color
    return


def get_virtual_observation(virtual_pose):
    """
    Builds an observation as if the robot was placed in virtual_pose.

    virtual_pose formats supported:
        - [x, y, yaw] 
        - [x, y, z, yaw]
        - [x, y, z, qx, qy, qz, qw]

    Returns:
    dict: Observation dictionary with the same keys and ordering
        as get_observation_space() ( = params_env["observation_names"]).
    """
    # Target pose
    target_pos = self.sim.getObjectPosition(self.inner_target, -1)

    # Unpack virtual pose of the robot
    if len(virtual_pose) == 3:
        # [x, y, yaw]
        x, y, yaw = virtual_pose
    elif len(virtual_pose) == 4:
        # [x, y, z, yaw]
        x, y, _z, yaw = virtual_pose
    elif len(virtual_pose) == 7:
        # [x, y, z, qx, qy, qz, qw]
        x, y, _z, qx, qy, qz, qw = virtual_pose
        # Convert quaternion -> Euler, take yaw
        m = self.sim.buildMatrixQ([0.0, 0.0, 0.0], [qx, qy, qz, qw])
        e = self.sim.getEulerAnglesFromMatrix(m)
        yaw = e[2]
    else:
        raise ValueError(
            "[Get VPose] virtual_pose must be [x,y,yaw], [x,y,z,yaw] "
            "or [x,y,z,qx,qy,qz,qw]."
        )

    # Distance between robot (in virtual pose) and target in XY plane (center-center)
    dx = target_pos[0] - x
    dy = target_pos[1] - y
    distance = math.hypot(dx, dy)

    # Relative angle
    angle = math.atan2(dy, dx) - yaw
    if angle > math.pi:
        angle -= 2 * math.pi
    elif angle < -math.pi:
        angle += 2 * math.pi

    # Fake laser measures
    if self.laserHandle is not None:
        pose_for_laser = [x, y, yaw]
        laser_obs = self.sim.callScriptFunction(
            'laser_get_observations_from_pose',
            self.handle_laser_get_observation_script,
            pose_for_laser
        )
    else:
        laser_obs = None
    
    # Build observation dict
    value_pool = {
        "distance": float(distance),
        "angle": float(angle),
    }
    if laser_obs is not None:
        for i, val in enumerate(laser_obs):
            value_pool[f"laser_obs{i}"] = float(val)
    else:
        expected_lasers = int(params_env.get("laser_observations", 0))
        for i in range(expected_lasers):
            value_pool[f"laser_obs{i}"] = 0.0

    names = params_env.get("observation_names", [])
    if not names:
        raise RuntimeError("[Get VPose] No observation_names configured in params_env.")

    obs = {}
    for name in names:
        if name in value_pool:
            obs[name] = value_pool[name]
        else:
            logging.warning(f"[Get VPose] Unknown obs name '{name}', filling 0.0")
            obs[name] = 0.0
    return obs


# -----------------------------------------
# -- Coppelia callbacks
# -----------------------------------------

def sysCall_init():
    """
    Initialization callback.

    Resolves object handles, sets up optional ROS2 publishers/subscribers and
    prepares graphs (if 'verbose == 3').
    """

    self.sim = require('sim')    # type: ignore

    # --------------------------------------------
    # -- Handles
    # --------------------------------------------
    handles = robot_common.initialize_robot_handles(self.sim, robot_alias, robot_base_alias, laser_alias)
    self.robotHandle = handles["robot_handle"]
    self.robotAlias = handles["robot_alias"]
    self.footprintHandle = handles["footprint_handle"]
    self.laserHandle = handles["laser_handle"]
    self.motorLeft = handles["motor_left"]
    self.motorRight = handles["motor_right"]
    self.handle_laser_get_observation_script = self.sim.getScript(1, self.laserHandle, "laser_get_observations")
    self.inner_target = self.sim.getObject("/Target/Inner_disk")

    # --------------------------------------------
    # -- ROS2 publishers and subscribers (optional)
    # --------------------------------------------
    if self.simROS2_flag:
        self.subscriber_twist, self.publisher_odometry, self.publisher_ground_truth = robot_common.setup_ros2_comms(
            simROS2, self.robotAlias  # type: ignore
        )

    # --------------------------------------------
    # -- Graphs if verbose
    # --------------------------------------------
    if verbose == 3:
        # Instantiate graph objects
        self.graph_vel = self.sim.getObject("/Velocity_graph")
        self.graph_lat = self.sim.getObject("/LAT_vs_Distance_graph")

        # Velocity graph
        self.leftMotorCurveX, self.leftMotorCurveY, self.rightMotorCurveX, self.rightMotorCurveY = createCurves(
            self.graph_vel, 'Time', 'Left Motor Velocity', 's', 'rad/s',
            'Time', 'Right Motor Velocity', 's', 'rad/s',
            'Left Motor', 'Right Motor'
        )

        # LAT graph
        self.latX, self.latY, self.distanceX, self.distanceY = createCurves(
            self.graph_lat, 'Time', 'LAT', 's', 's',
            'Distance Time', 'Traveled distance', 's', 'm',
            'LAT', 'Traveled distance'
        )

    # --------------------------------------------
    # -- For testing laser with virtual poses
    # --------------------------------------------

    # Note: Please comment for normal use until the end of sysCall_init function (also see sysCall_sensing)

    # # Get the associated child script handle
    # self.laser_script_handle = self.sim.getScript(self.sim.scripttype_childscript, self.laserHandle)

    # # Get robot pose to build some poses around it
    # robot_pos = self.sim.getObjectPosition(self.robotHandle, self.sim.handle_world)
    # robot_ori = self.sim.getObjectOrientation(self.robotHandle, self.sim.handle_world)
    # robot_yaw = robot_ori[2]

    # # Build a small list of virtual poses:
    # self.poses = []
    # self.poses.append([
    #     robot_pos[0] + 2.0 * math.cos(robot_yaw),
    #     robot_pos[1] + 2.0 * math.sin(robot_yaw),
    #     robot_yaw
    # ])
    # self.poses.append([
    #     robot_pos[0] + 2.0 * math.cos(robot_yaw + math.pi / 2),
    #     robot_pos[1] + 2.0 * math.sin(robot_yaw + math.pi / 2),
    #     robot_yaw + math.pi
    # ])
    # m_robot = self.sim.buildMatrix(robot_pos, robot_ori)
    # q_robot = self.sim.getQuaternionFromMatrix(m_robot)
    # self.poses.append([
    #     robot_pos[0], robot_pos[1], robot_pos[2],
    #     q_robot[0], q_robot[1], q_robot[2], q_robot[3]
    # ])
    # self.pose_idx = 0

    
def sysCall_actuation():
    """
    Actuation callback.

    Responsible for publishing TF transforms via ROS2 (if enabled).
    """
    if self.simROS2_flag:
        robot_common.publish_ros2_tfs(
            self.sim, simROS2, self.footprintHandle, self.robotHandle, self.laserHandle,  # type: ignore
            self.robot_initial_pose_handle, self.robotAlias
        )


def sysCall_sensing():
    """
    Sensing callback.

    Updates graphs (if enabled) and, for testing, calls the laser script
    with a sequence of virtual poses and prints the distances returned.
    """
    # --------------------------------------------
    # -- Update graphs if verbose
    # --------------------------------------------

    if verbose == 3 and wheel_radius is not None:
        # Get the current simulation time
        currentTime = self.sim.getSimulationTime()
        relativeTime = currentTime - self.graphStartTime

        # Get wheel angular velocities
        omegaLeft = self.sim.getJointVelocity(self.motorLeft)
        vLeft = omegaLeft * wheel_radius

        omegaRight = self.sim.getJointVelocity(self.motorRight)
        vRight = omegaRight * wheel_radius

        # Update the graph with the linear and angular velocities
        self.sim.setGraphStreamValue(self.graph_vel, self.leftMotorCurveX, relativeTime)
        self.sim.setGraphStreamValue(self.graph_vel, self.leftMotorCurveY, omegaLeft)
        self.sim.setGraphStreamValue(self.graph_vel, self.rightMotorCurveX, relativeTime)
        self.sim.setGraphStreamValue(self.graph_vel, self.rightMotorCurveY, omegaRight)

        # Get traveled distance according to last simulation LAT and linear speed
        latValue = self.sim.getFloatSignal('latValueSignal')

        if latValue is not None and self.linearSpeed != -1:
            distValue = self.linearSpeed * latValue

            # Update LAT vs Distance graph
            self.sim.setGraphStreamValue(self.graph_lat, self.latX, relativeTime)
            self.sim.setGraphStreamValue(self.graph_lat, self.latY, latValue)
            self.sim.setGraphStreamValue(self.graph_lat, self.distanceX, relativeTime)
            self.sim.setGraphStreamValue(self.graph_lat, self.distanceY, distValue)


def sysCall_cleanup():
    # Do some clean-up here if needed
    pass