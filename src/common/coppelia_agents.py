import csv
import json
import logging
import math
import os
import random
import sys
import time
from typing import Dict, Tuple
import numpy as np
import pandas as pd
from common import utils
from spindecoupler import AgentSide # type: ignore


class CoppeliaAgent:

    def _check_variable_timestep(self, params_env: Dict) -> None:
        """
        Check if the action names include 'timestep' to determine if variable timestep is used.
        Sets the _rltimestep and _variable_timestep attributes accordingly.
        Args:
            params_env (dict): Environment parameters loaded from a JSON file.
        """
        # Get list of action names from params
        action_names = params_env.get("action_names", [])

        # Set timestep duration for RL actions
        if "timestep" in action_names:
            # The timestep will be variable, provided in the action dict
            self._rltimestep = None 
            self.variable_timestep = True
            logging.info("Variable timestep detected: will use action value instead of fixed_actime.")
        else:
            # Default to fixed timestep
            self._rltimestep = params_env.get("fixed_actime", 0.75)
            self.variable_timestep = False
            self._validate_rltimestep()
            logging.info(f"Fixed timestep set to {self._rltimestep:.3f} s.")

    
    def _validate_rltimestep (self) -> None:
        """Validate that the RL timestep is strictly greater than control timestep.

        In variable-timestep mode, _rltimestep may be None before the first action;
        in that case the check is skipped.

        Raises:
            ValueError: If _rltimestep is set and not greater than _control_timestep.
        """
        # Skip if not yet set (variable-timestep before first action)
        if getattr(self, "_rltimestep", None) is None:
            return
        if self._rltimestep <= self._control_timestep:
            raise(ValueError("RL timestep must be > control timestep"))


    def _update_rltimestep(self, action: Dict[str, float]) -> None:
        """Update _rltimestep from the incoming action if in variable-timestep mode.

        Args:
            action (dict): Action dictionary received from RL.

        Raises:
            KeyError: If variable mode is enabled but the timestep key is missing.
            ValueError: If the extracted timestep is invalid (<= control timestep).
            TypeError: If the timestep value is not convertible to float.
        """
        if "timestep" not in action:
            raise KeyError("Missing 'timestep' key in action while variable-timestep mode is enabled.")
        
        # Convert and validate type
        val = action["timestep"]
        try:
            self._rltimestep = float(val)
        except (TypeError, ValueError):
            raise TypeError(f"Invalid 'timestep' value: {val!r} (must be numeric)")

        # Check if it is okey in relation with the control timestep of the simulator
        self._validate_rltimestep()


    def __init__(self, sim, params_scene, params_env, paths, file_id, verbose, ip_address, comms_port = 49054) -> None:
        """
        Custom agent for CoppeliaSim simulations of different robots.
        
        This agent interfaces with the RLSide class (from spindecoupler package) to
        receive actions and return observations in a reinforcement learning (RL) setup. The
        environment simulates the robot's movement in response to actions.
        Args:
            sim: The CoppeliaSim simulation instance.
            params_env (dict): Environment parameters loaded from a JSON file.
            paths (dict): Dictionary containing various paths for saving/loading data.
            file_id (str): Unique identifier for the current training/testing session.
            verbose (int): Verbosity level for logging.
                ip_address (str): IP address for communication with the RL side.
            comms_port (int, optional): Port for communication with the RL side. Defaults to 49054.
        Attributes:
            _commstoRL: Instance of AgentSide for communication with the RL side.
            _control_timestep (float): Timestep used by the simulation engine.
            _rltimestep (float): Timestep for RL actions.
            _waitingforrlcommands (bool): Flag indicating if the agent is waiting for RL commands.
            _lastaction (dict): The most recent action executed.
            _lastactiont0_sim (float): Simulation time when the last action was received.
            _lastactiont0_wall (float): Wall-clock time when the last action was received.
            lat_sim (float): Last Action Time (LAT) in simulation time.
            lat_wall (float): Last Action Time (LAT) in wall-clock time.
            current_sim_offset_time (float): Elapsed simulated time in the current episode.
            current_wall_offset_time (float): Elapsed wall-clock time in the current episode.
            verbose (int): Verbosity level for logging.
            reward (float): Reward for the current step.
            execute_cmd_vel (bool): Flag to indicate if cmd_vel should be executed.
            colorID (int): Color ID for visualization purposes.
            robot, robot_baselink, target, inner_targer: Handles for robot and target objects in CoppeliaSim.
            distance_line: Handle for the debug line showing distance between robot and target.
            handle_robot_scripts: Handle for robot script in CoppeliaSim.
            laser: Handle for laser sensor object, if any.
            generator: Handle for obstacle generator object, if any.
            handle_laser_get_observation_script: Handle for laser observation script, if any.
            handle_obstaclegenerators_script: Handle for obstacle generator script, if any.
            sim: The CoppeliaSim simulation instance.
            params_env (dict): Environment parameters loaded from a JSON file.
            initial_simTime (float): Initial simulation time when the agent starts its first movement.
            initial_realTime (float): Initial wall-clock time when the agent starts its first movement.
            paths (dict): Dictionary containing various paths for saving/loading data.
            first_reset_done (bool): Flag indicating if the first reset has been done.
            finish_rec (bool): Flag to indicate if Finish flag has been received.
            episode_start_time_sim (float): Simulation time when the current episode started.
            episode_start_time_wall (float): Wall-clock time when the current episode started.
            reset_flag (bool): Flag to indicate if a reset has been requested.
            crash_flag (bool): Flag to indicate if a crash has occurred.
            training_started (bool): Flag to indicate if training has started.
            scene_to_load_folder (str): Folder name for loading preconfigured scenes.
            id_obstacle (int): Counter for obstacle IDs.
            action_times (list): List of action times for loading scenes.
            tuples (list): List of tuples (action_time, target_id) for loading scenes.
            num_targets (int): Number of targets in the loaded scene.
            test_scene_mode (str): Mode for loading test scenes ("alternate_targets" or "alternate_action_times").
            df (pd.DataFrame): DataFrame containing scene configuration loaded from CSV.
            target_rows (pd.DataFrame): DataFrame containing only target rows from the scene configuration.
            save_scene (bool): Flag to indicate if the current scene should be saved.
            scene_configs_path (str): Path to the scene configurations directory.
            experiment_id (str): Unique identifier for the current training/testing session.
            episode_idx (int): Current episode index.
            trajectory (list): List to store the robot's trajectory for the current episode.
            save_scene_csv_folder (str): Folder path for saving scene CSV files.
            save_traj (bool): Flag to indicate if trajectories should be saved.
            model_ids (list): List of model IDs for naming trajectory files.
            save_trajs_path (str): Path to the directory for saving trajectory files.
            save_traj_csv_folder (str): Folder path for saving trajectory CSV files.
        Methods:
            get_observation(): Compute the current distance and angle from the robot to the target.
            get_observation_space(): Returns the observation space of the agent.
            generate_obs_from_csv(row): Generate an obstacle in the CoppeliaSim scene based on a row from the CSV file.
            get_random_object_pos(): Get a random robot/target position inside the container.
            is_position_valid(): Check if a random generated position is valid (no collisions with obstacles)
            reset_simulator(): Reset the simulator: position the robot and target, and reset the counters.
            agent_step(): A step of the agent. Process incoming instructions from the server side and execute actions accordingly.
        
        """

        sim.setFloatParam(sim.floatparam_simulation_time_step,0.05)
        self._control_timestep = sim.getSimulationTimeStep()

        # Set timestep duration for RL actions
        self._check_variable_timestep(params_env)
        
        self._waitingforrlcommands = True
        self._lastaction = None
        self._lastactiont0_sim = 0.0
        self._lastactiont0_wall = 0.0
        self.lat_sim = 0.0
        self.lat_wall = 0.0
        self.current_sim_offset_time = 0.0
        self.current_wall_offset_time = 0.0
        self.verbose = verbose

        self.reward = 0
        self.execute_cmd_vel = False
        self.colorID = 1
        
        self.robot_name = None
        self.robot = None
        self.robot_baselink = None
        self.distance_line = None
        self.laser = None

        self.target=sim.getObject("/Target")
        self.inner_target=sim.getObject("/Target/Inner_disk")
        self.container = sim.getObject('/ExternalWall')
        self.generator=sim.getObject('/ObstaclesGenerator')
        self.handle_laser_get_observation_script=None
        self.handle_robot_scripts = None
        self.handle_obstaclegenerators_script=sim.getScript(1,self.generator,'generate_obs')        

        self.sim = sim
        self.params_env = params_env
        self.params_scene = params_scene

        self.initial_simTime = 0
        self.initial_realTime = 0

        self.paths = paths
        self.first_reset_done = False

        # Communication
        self.ip_address = ip_address
        self.comms_port = comms_port
        self._commstoRL = None  # To be initialized externally after agent creation

        self._query_comm = None
        self.query_port = comms_port + 1
        
        # Process control variables
        self.finish_rec = False
        self.episode_start_time_sim = 0.0
        self.episode_start_time_wall = 0.0
        self.reset_flag = False
        self.crash_flag = False
        self.training_started = False

        # For loading a scene
        self.scene_to_load_folder = ""
        self.id_obstacle = 0
        self.action_times = []
        self.tuples = []
        self.num_targets = 0
        self.test_scene_mode = ""
        self.df = None
        self.target_rows = None

        # Needed for saving scenes
        self.save_scene = False
        self.scene_configs_path = self.paths["scene_configs"]
        self.experiment_id = file_id
        self.episode_idx = 0
        self.trajectory = []
        self.save_scene_csv_folder = os.path.join(
            self.scene_configs_path,
            self.experiment_id,
            "scene_episode"
        )
        
        # For saving trajectory
        self.save_traj = False
        self.model_ids = []
        self.save_trajs_path = self.paths["testing_metrics"]
        self.save_traj_csv_folder = ""

        # For saving obstacles objects generated
        self.obstacles_objs = None

        # For indicating that the lat reset have been done with the first reset of the scene
        self.lat_reset = False

        # For loading obstacles when training with fixed osbtacles:
        self.obstacles_csv_folder = ""

        # Path-version (pv) mode configuration
        self.pv_mode = False
        self.place_obstacles_flag = False
        self.random_target_flag = False
        self._pv_step_duration = 0.15  # Fixed action duration for pv) mode (seconds)

        # Path version script for building timesteps map
        self.current_trial_idx_pv = 0
        self.trials_per_sample = 0
        self.ts_received = False
        self.current_sample_idx_pv = 0
        self.path_pos_samples = []
        self.path_base_pos_samples = []
        self.path_handle = None
        self.perimeterRadius = 1.0  # meters
        self.new_robot_pose = (0.0, 0.0, 0.0, 0.0)
        self.grid_positions_flag = False    # True when the user sends a map image to get positions from a grid
        self.robot_target_ori = 0   # Orientation between the robot and the target (with respect to X axis)

        # For testing specific target/robot positions in a custom map
        self.target_pos_samples = []
        self.robot_pos_samples = []

        # test_map mode: pre-computed test cases with robot pose and target position
        self.test_map_mode = False
        self.test_cases = []
        self.fixed_target_pos = None
        self.current_test_case_idx = 0


    # ----------------------------------
    # ----- HELPERS: Communication -----
    # ----------------------------------

    def start_communication(self, retry_delay=5.0, max_retry_time=None):
        """Attempt to establish communication with the RL side, with retries.

        This function repeatedly attempts to create an 'AgentSide' communication
        object until the RL process becomes available. It uses real-time sleeps 
        ('time.sleep') between attempts to avoid blocking CoppeliaSim simulation 
        thread. Designed to be called from 'sysCall_thread()' in CoppeliaSim 
        embedded Python environment.

        Args:
            retry_delay (float): Number of real-time seconds to wait between
                connection attempts.
            max_retry_time (float or None): Maximum amount of real-time seconds to
                keep retrying. If None, retries indefinitely.

        Returns:
            bool: True if the connection is successfully established.
                False if the maximum retry time is exceeded or the simulation
                is stopping.

        Notes:
            - This function should be executed only inside 'sysCall_thread()', not
            inside 'sysCall_sensing' or other callbacks, as that would break the simulation.
        """
        start_time = time.time()
        attempt = 0

        while True:
            attempt += 1
            try:
                logging.info(
                    f"[comms] Attempt {attempt}: connecting to RL at "
                    f"{self.ip_address}:{self.comms_port}"
                )

                # Attempt to open RL-side communication
                self._commstoRL = AgentSide(self.ip_address, self.comms_port)

                logging.info(
                    f"[comms] Communication established with RL at "
                    f"{self.ip_address}:{self.comms_port}"
                )
                return True

            except Exception as e:
                logging.warning(
                    f"[comms] Failed to connect to RL (attempt {attempt}): "
                    f"{type(e).__name__}: {e}. Retrying in {retry_delay} seconds..."
                )

                # If the simulation is stopping, break cleanly
                if self.sim is not None and self.sim.getSimulationStopping():
                    logging.warning("[comms] Simulation is stopping. Aborting connection attempts.")
                    return False

                # Abort if a maximum retry time is defined and exceeded
                if max_retry_time is not None:
                    elapsed = time.time() - start_time
                    if elapsed > max_retry_time:
                        logging.error(
                            f"[comms] Unable to establish RL communication after "
                            f"{elapsed:.1f} seconds. Aborting."
                        )
                        if hasattr(self, "finish_rec"):
                            self.finish_rec = True  # The experiment is truncated
                        return False

                # Wait before retrying
                time.sleep(retry_delay)


    # ----------------------------------
    # ----- HELPERS: OBSERVATIONS ------
    # ----------------------------------

    def _get_observation(self):
        """
        Compute the current distance and angle from the robot to the target,
        and draw a debug line showing the measured distance.

        Note: the method 'sim.checkDistance' cannot be used here as it calculate the 
        nearest distance, not the center-to-center one.
        """

        # Robot position
        p1 = self.sim.getObjectPosition(self.robot_baselink, self.sim.handle_world)

        # Target position
        self.inner_target = self.sim.getObject("/Target/Inner_disk")    # The target is recreated when starting the scene, so refresh its handle here.
        p2 = self.sim.getObjectPosition(self.inner_target, self.sim.handle_world)

        # Distance center-center in XY plane
        dx = p2[0] - p1[0]
        dy = p2[1] - p1[1]
        distance = math.hypot(dx, dy)

        # DEBUG: draw line to visualize distance center-to-center
        if self.verbose ==3:
            try:
                # Delete previous debug drawing (if exists)
                if hasattr(self, "distance_line") and self.distance_line is not None:
                    self.sim.removeDrawingObject(self.distance_line)

                # Create a new line-drawing object (size=2, color=black)
                self.distance_line = self.sim.addDrawingObject(
                    self.sim.drawing_lines, 2.0, 0.0, -1, 1, [0, 0, 0]
                )

                # Add the two endpoints (world coordinates)
                line_data = [p1[0], p1[1], p1[2], p2[0], p2[1], p2[2]]
                self.sim.addDrawingObjectItem(self.distance_line, line_data[0:6])

            except Exception as e: 
                logging.info(f"[DEBUG] Could not draw distance line: {e}")

        # Compute relative angle   
        twist = self.sim.getObjectOrientation(self.robot_baselink, -1)
        angle = math.atan2(dy, dx) - twist[2]

        # Normalize the angle
        if angle > math.pi:
            angle -= 2 * math.pi
        elif angle < -math.pi:
            angle += 2 * math.pi

        # Add laser measurements to the observation space (optional)
        if self.laser is not None:
            lasers_obs = self.sim.callScriptFunction(
                'laser_get_observations', self.handle_laser_get_observation_script
            )
        else:
            lasers_obs = None

        return distance, angle, lasers_obs


    def get_observation_space(self):
        """Build the observation dict using names from params_env["observation_names"].

        This method calls '_get_observation()' (which returns distance, angle, and optionally
        laser observations) and then assembles a dictionary of observations in the exact
        order and naming specified by 'self.params_env["observation_names"]'.

        Behavior:
            - Known base signals: "distance", "angle".
            - Laser signals are exposed as "laser_obs{i}" (e.g., laser_obs0, laser_obs1, ...).
            - If lasers are not available but the config includes laser entries, those keys
            will be filled with a default numeric value (0.0) to keep the shape stable.

        Returns:
            dict: Observation dictionary keyed by names in params_env["observation_names"].
        """
        # Get raw measurements from the simulator
        distance, angle, laser_obs = self._get_observation()

        # Build a value pool from available signals
        # (distance, angle are always present; lasers may be None)
        value_pool = {
            "distance": float(distance),
            "angle": float(angle),
        }

        if laser_obs is not None:
            # Expose laser beams as laser_obs0..N-1
            for i, val in enumerate(laser_obs):
                value_pool[f"laser_obs{i}"] = float(val)
        else:
            # If config expects lasers but we don't have them now, fill with defaults.
            # This keeps the observation shape consistent with the Box space.
            logging.warning("[Get obs] No laser observations obtained from _get_observation(). They will be set to 0.")
            expected_lasers = int(self.params_env.get("laser_observations", 0))
            for i in range(expected_lasers):
                value_pool[f"laser_obs{i}"] = 0.0

        # Desired names and order come from the params file
        names = self.params_env.get("observation_names")

        if names:
            obs = {}
            for name in names:
                if name in value_pool:
                    obs[name] = value_pool[name]
                else:
                    # Unknown name in config: keep numeric output stable and warn once
                    logging.warning(f"[Get obs] Unknown observation name in config: '{name}'. Filling 0.0")
                    obs[name] = 0.0
            logging.info(f"[Get obs] Observation stored as: { {key: round(value, 3) for key, value in obs.items()} }")

            return obs

        else:
            raise RuntimeError("No observation names in params json file. Please check it.")


    # -------------------------------------
    # ----- HELPERS: SCENE GENERATION -----
    # -------------------------------------


    def generate_obs_from_csv(self, row):
        '''
        Generate an obstacle in the CoppeliaSim scene based on a row from a CSV file.
        Args: 
            row (pd.Series): A row from the CSV file containing 'x' and 'y' coordinates for the obstacle.
        Returns:
            None
        '''
        logging.debug(f"Generating obstacles from csv file")
        height_obstacles = self.params_scene["height_obstacles"]
        diam_obstacles = self.params_scene["diam_obstacles"]

        x, y = row["x"], row["y"]
        logging.debug(f"Placing obstacle at x: {x} and y: {y}")
        obs = self.sim.createPrimitiveShape(5, [diam_obstacles, diam_obstacles, height_obstacles])
        self.sim.setObjectPosition(obs, self.sim.handle_world, [x, y, height_obstacles / 2])
        self.sim.setObjectAlias(obs, f"Obstacle_csv_{self.id_obstacle}")
        self.sim.setObjectParent(obs, self.generator, True)
        self.sim.setObjectSpecialProperty(obs, self.sim.objectspecialproperty_collidable |
                                        self.sim.objectspecialproperty_measurable |
                                        self.sim.objectspecialproperty_detectable)
        self.sim.setObjectInt32Param(obs, self.sim.shapeintparam_respondable, 1)
        self.sim.setObjectInt32Param(obs, self.sim.shapeintparam_static, 0)
        self.sim.setShapeMass(obs, 1000)
        self.sim.resetDynamicObject(obs)
        
        return


    def get_random_object_pos (self, object_type):
        '''
        Get a random object position inside the container, taking into account the object radius and the container dimensions.
        It is used for locating the robot and the target in standard modes (not test_map).
        
        Args:
            object_type (string): String to indicate if it is the 'robot' or the 'target'.
        Returns:
            tuple: x and y coordinates of the object position.
        '''
        # Get wall info from its customization script
        raw_container = self.sim.readCustomBufferData(self.container, '__config__')
        cfg_wall   = self.sim.unpackTable(raw_container) if raw_container else {}
        containerSideX    = cfg_wall.get('scene_x_dim', None)
        containerSideY    = cfg_wall.get('scene_y_dim', None)

        # Define a tolerance to ensure the object is not placed too close to the walls.
        tolerance = 0.04  # meters

        # Calculate min distance between objects and wall
        threshold = utils.get_min_objet_dist (self, object_type, tolerance)

        if containerSideX is None or containerSideY is None or threshold is None:
            raise ValueError("[CoppeliaAgent - Get random pos] Missing scene dimensions or object radius; cannot sample a valid position.")

        objectPosX = random.uniform(-containerSideX/2 + threshold, containerSideX/2 - threshold)
        objectPosY = random.uniform(-containerSideY/2 + threshold, containerSideY/2 - threshold)

        return objectPosX, objectPosY 

    
    def get_target_relative_to_robot(
        self,
        robot_pose_world: Tuple[float, float, float, float],
        distance: float,
        robot_target_ori: float
    ) -> Tuple[float, float]:
        """
        Compute a target position at a given distance and relative orientation.

        The robot forward axis is assumed to be the +X axis of its local frame
        rotated by the yaw angle in world coordinates (standard 2D planar assumption).

        Args:
            robot_pose_world (tuple[float, float, float, float]): Robot pose as a tuple/list of 4 floats: (x, y, z, yaw).
                If you pass only 3 elements (x, y, yaw), z is assumed 0.0.
                The last element is interpreted as yaw (heading).
            distance (float): Distance in meters. Positive places the target in front.
            robot_target_ori (float): Relative orientation in degrees from the robot X axis.
    
        Returns:
            (tx, ty) (tuple[float, float]): Target position in world coordinates.

        Notes:
        - Forward direction is computed from (yaw + relative_orientation) in world XY.
        - This function does not clamp to map bounds nor check collisions.
        """

        if len(robot_pose_world) < 3:
            raise ValueError("robot_pose_world must have at least (x, y, yaw).")
        x = float(robot_pose_world[0])
        y = float(robot_pose_world[1])

        if len(robot_pose_world) == 3:
            robot_yaw = float(robot_pose_world[2])
        else:
            robot_yaw = float(robot_pose_world[-1])

        # Direction defined by the robot heading plus the relative target orientation.
        rel_angle = math.radians(robot_target_ori)
        abs_angle = robot_yaw + rel_angle
        fx = math.cos(abs_angle)
        fy = math.sin(abs_angle)

        tx = x + distance * fx
        ty = y + distance * fy
        return tx, ty


    def get_target_in_path_pos(
        self,
        robot_pose_world: Tuple[float, float, float, float],
        radius: float,
    ):
        """Compute a target point on the path using a robot-centered lookahead circle.

        The algorithm:
        1) Reads the densified path samples from 'customData.PATH' (local Path frame).
        2) Transforms path points into world frame using the Path pose.
        3) Computes 2D intersections (XY) between the path segments and a circle of
            radius 'radius' centered at the robot (x, y).
        4) Selects the nearest intersection that is ahead of the robot (dot(h, p-c) > 0),
            where h = [cos(yaw), sin(yaw)] and c = robot XY.
        5) If the ir no ahead intersection, falls back to a point D=radius straight
            ahead of the robot along +X (its heading).
        6) Returns the chosen (x, y) in world frame and, if 'self.target' is
            available, places the target there.

        Args:
            robot_pose_world (tuple[float, float, float, float]): (x, y, z, yaw) of the robot in world frame.
            radius (float): Lookahead circle radius in meters. Also used as forward distance D for the fallback.

        Returns:
            tuple[float, float]: Target world position (x, y).

        Notes:
            - Requires the Path object to have 'customData.PATH'.
            - Intersections are computed in XY; z is linearly interpolated across the segment.
            - If multiple ahead intersections exist, the closest to the robot center is chosen.
        """

        # -------------------
        # ----- Helpers -----
        # -------------------

        def _q_normalize(q):
            """Normalize quaternion (x, y, z, w)."""
            x, y, z, w = q
            n = math.sqrt(x * x + y * y + z * z + w * w)
            if n < 1e-16:
                return (0.0, 0.0, 0.0, 1.0)
            return (x / n, y / n, z / n, w / n)

        def _q_rotate_point(q, p):
            """Rotate point p=(x,y,z) by quaternion q=(x,y,z,w)."""
            x, y, z = p
            qx, qy, qz, qw = q
            ix = qw * x + qy * z - qz * y
            iy = qw * y + qz * x - qx * z
            iz = qw * z + qx * y - qy * x
            iw = -qx * x - qy * y - qz * z
            rx = ix * qw + iw * -qx + iy * -qz - iz * -qy
            ry = iy * qw + iw * -qy + iz * -qx - ix * -qz
            rz = iz * qw + iw * -qz + ix * -qy - iy * -qx
            return (rx, ry, rz)
        
    
        def _segment_circle_intersections(a, b, center_xy, radius_):
            """Return list of (t, x, y, z) with t in [0,1] for segment a->b, circle in XY."""
            ax, ay, az = a
            bx, by, bz = b
            cx_, cy_ = center_xy
            dx, dy = (bx - ax), (by - ay)

            A = dx * dx + dy * dy
            if A <= 1e-16:
                return []  # degenerate segment in XY

            fx, fy = (ax - cx_), (ay - cy_)
            B = 2.0 * (dx * fx + dy * fy)
            C = fx * fx + fy * fy - radius_ * radius_
            disc = B * B - 4.0 * A * C
            if disc < -1e-12:
                return []

            out = []
            disc = max(disc, 0.0)  # clamp small negatives due to precision
            sqrt_disc = math.sqrt(disc)
            for s in (-1.0, 1.0):
                t = (-B + s * sqrt_disc) / (2.0 * A)
                if -1e-12 <= t <= 1.0 + 1e-12:
                    t = min(1.0, max(0.0, t))
                    x = ax + t * (bx - ax)
                    y = ay + t * (by - ay)
                    z = az + t * (bz - az)  # linear z interpolation
                    out.append((t, x, y, z))
            return out


        # Read & unpack densified path (local frame) 
        buf = self.sim.getBufferProperty(self.path_handle, 'customData.PATH', {'noError': True})
        if not buf:
            self.sim.addLog(self.sim.verbosity_scripterrors, 'customData.PATH not found on the Path.')
            return None
        data = self.sim.unpackDoubleTable(buf)  # [x,y,z,qx,qy,qz,qw, ...]
        if len(data) % 7 != 0:
            self.sim.addLog(self.sim.verbosity_scripterrors, 'customData.PATH size is not a multiple of 7.')
            return None

        n = len(data) // 7
        if n < 2:
            self.sim.addLog(self.sim.verbosity_scripterrors, 'Path has fewer than 2 samples.')
            return None

        # Keep only local positions (we ignore local quaternions for intersections)
        loc_pts = [(data[7 * i + 0], data[7 * i + 1], data[7 * i + 2]) for i in range(n)]

        # Transform local path points -> world 
        p_path = self.sim.getObjectPosition(self.path_handle, -1)     # world
        q_path = self.sim.getObjectQuaternion(self.path_handle, -1)   # world
        q_path = _q_normalize((q_path[0], q_path[1], q_path[2], q_path[3]))

        w_pts = []
        append_wp = w_pts.append
        px, py, pz = p_path
        for (lx, ly, lz) in loc_pts:
            wx, wy, wz = _q_rotate_point(q_path, (lx, ly, lz))
            append_wp((wx + px, wy + py, wz + pz))

        # Unpack robot pose & heading 
        cx, cy, cz, yaw = robot_pose_world
        r = float(radius)
        hx, hy = math.cos(yaw), math.sin(yaw)  # heading unit vector


        # Circle–segment intersections in XY
        candidates_ahead = []
        for i in range(len(w_pts) - 1):
            a = w_pts[i]
            b = w_pts[i + 1]
            for (t, x, y, z) in _segment_circle_intersections(a, b, (cx, cy), r):
                vx, vy = (x - cx), (y - cy)
                if (hx * vx + hy * vy) > 0.0:  # ahead test
                    dist2 = vx * vx + vy * vy
                    candidates_ahead.append((dist2, x, y, z, i, t))

        # Choose best ahead intersection or fallback 
        if candidates_ahead:
            # Nearest ahead
            _, x, y, z, seg_idx, t = min(candidates_ahead, key=lambda c: c[0])
        else:
            # Fallback: place D=radius straight ahead along robot +X
            x = cx + r * hx
            y = cy + r * hy
            z = cz 

        # Apply to target (if available) and return
        if getattr(self, 'target_handle', None) is not None:
            self.sim.setObjectPosition(self.target_handle, [x, y, z], -1)

        return x, y


    def is_position_valid(self, object_type, posObjectX, posObjectY):
        '''
        Check if the object position is valid (not colliding with any obstacle).
        
        Args:
            object_type (string): String to indicate if it is the 'robot' or the 'target'.
            posObjectX (float): x coordinate of the target position.
            posObjectY (float): y coordinate of the target position.
        Returns:
            bool: True if the position is valid, False otherwise.
        '''
        objs = self.sim.getObjectsInTree(self.obstacles_objs, self.sim.handle_all, 1) or []

        # Calculate the distance between the object proposed position and each obstacle
        for obj in objs:
            pos_obstacle = self.sim.getObjectPosition(obj, self.sim.handle_world)  # [x, y, z]
            dx, dy = (posObjectX - pos_obstacle[0]), (posObjectY - pos_obstacle[1])
            dist = math.hypot(dx, dy)

            # Get the threshold for each case
            threshold = utils.get_min_objet_dist(self, object_type)
            
            # Always add the obstacle radius, as distance is calculated center to center
            threshold = threshold + self.params_scene["diam_obstacles"]/2

            # Check if the distance does not respect the minimum threshold
            if dist < threshold:
                return False
        return True


    # --------------------------
    # ----- RESET FUNCTION -----
    # --------------------------


    def reset_simulator(self):
        """
        Reset the simulator: position the robot and target, and reset the counters.
        If there are obstacles, remove them and create new ones.
        """

        # Set speed to 0. It's important to do this before setting the position and orientation
        # of the robot, to avoid bugs with Coppelia simulation
        self.sim.callScriptFunction('cmd_vel',self.handle_robot_scripts,0,0)
        if self.verbose == 3:
            self.sim.callScriptFunction('draw_path', self.handle_robot_scripts, 0,0, self.colorID)

        # Calculate lat:
        self.current_sim_offset_time = self.sim.getSimulationTime()-self.episode_start_time_sim
        self.current_wall_offset_time = self.sim.getSystemTime()-self.episode_start_time_wall

        self.lat_sim= self.current_sim_offset_time-self._lastactiont0_sim
        self.lat_wall= self.current_wall_offset_time-self._lastactiont0_wall

        self._lastactiont0_sim = self.current_sim_offset_time
        self._lastactiont0_wall = self.current_wall_offset_time

        # Reset colorID counter
        self.colorID = 1

        # Save trajectory at the beggining of the reset (last episode traj)
        if self.save_traj:
            if self.trajectory != []:
                if self.model_ids is not None and self.model_ids != []:
                    traj_output_path = os.path.join(self.save_traj_csv_folder, f"trajectory_{self.episode_idx}_{self.model_ids[self.episode_idx-1]}.csv")
                else:
                    traj_output_path = os.path.join(self.save_traj_csv_folder, f"trajectory_{self.episode_idx}.csv")
                with open(traj_output_path, mode='w', newline='') as f:
                    writer = csv.DictWriter(f, fieldnames=["x", "y"])
                    writer.writeheader()
                    writer.writerows(self.trajectory)
                self.trajectory = []
            
                logging.info(f"Trajectory saved in CSV: {traj_output_path}")

        
        # If 'fixed_obs' flag is not set, remove old obstacles before creating new ones
        if not self.params_scene["fixed_obs"]:
            logging.info("Resetting simulator changing obstacles positions")

            # Always remove old obstacles before creating new ones
            if self.generator is not None:
                # Remove old obstacles
                last_obstacles=self.sim.getObjectsInTree(self.generator,self.sim.handle_all,1) 
                if len(last_obstacles) > 0:
                    self.sim.removeObjects(last_obstacles)
        else:
            logging.info("Resetting simulator with fixed obstacles")
        
        # Just place the scene objects at random positions and call 'generate_obs'
        if self.scene_to_load_folder == "" or self.scene_to_load_folder is None:
            # Reset positions and orientation
            current_position = self.sim.getObjectPosition(self.robot_baselink, -1)
            
            # If obstacles are not fixed (so they are randomized after each episode, alternated between some possible 
            # locations sent by RL side, or we jsut have no obstacles at all):
            if not self.params_scene["fixed_obs"]:

                # Default case: just place the robot at the center of the scene
                if self.robot_pos_samples == []:
                    if current_position != [0, 0, 0.06969]:
                        self.sim.setObjectPosition(self.robot_baselink,-1, [0, 0, 0.06969])

                # If agent has received a list of specific positions, a random one will be taken and robot 
                # will be palced there
                else:
                    # Use predefined robot positions
                    idx = np.random.randint(0, len(self.robot_pos_samples))
                    robot_pos = self.robot_pos_samples[idx]
                    logging.info(f"Using predefined robot position: {robot_pos}")
                    self.sim.setObjectPosition(self.robot_baselink,-1, [robot_pos[0], robot_pos[1], 0.06969])
            
            # If the obstacles are palced in fixed locations, then set a random position for the robot 
            else:
                if not self.first_reset_done:
                    # Read obstacles csv
                    csv_folder = os.path.join(self.paths["scene_configs"], self.obstacles_csv_folder)  
                    scene_path = utils.find_scene_csv_in_dir(csv_folder)
                    if not os.path.exists(scene_path):
                        logging.error(f"[ERROR] CSV scene file not found: {scene_path}")
                        sys.exit()

                    logging.info(f"Reading obstacles from: {scene_path}")
                    self.df = pd.read_csv(scene_path)
                    coords = []
                    for _, row in self.df.iterrows():
                        if row['type'] == 'obstacle':
                            coords.append((float(row['x']), float(row['y'])))
                    
                    # self.generate_obs_from_csv(row)
                    logging.info(f"Obstacles coords: {coords}")
                    self.obstacles_objs = self.sim.callScriptFunction('generate_obs',self.handle_obstaclegenerators_script, coords, [])

                while True:
                    posX, posY = self.get_random_object_pos('robot')
                    if self.is_position_valid('robot', posX, posY):
                        break
                logging.info(f"Robot new position: {posX}, {posY}")
                self.sim.setObjectPosition(self.robot_baselink,-1, [posX, posY, 0.06969])

            # The orientation will always be randomized
            random_ori = random.uniform(-math.pi, math.pi)
            self.sim.setObjectOrientation(self.robot_baselink,-1,[0,0,random_ori])

            # Randomize target position
            if self.target_pos_samples == []:
                current_target_position = self.sim.getObjectPosition(self.target, -1)
                if current_target_position != [0, 0, 0]:
                    if not self.params_scene["fixed_obs"]:
                        posX, posY = self.get_random_object_pos('target')
                        self.sim.setObjectPosition(self.target, -1, [posX, posY, 0])
                    # As the obstacles are the same as in the previous episode, we need to check that the
                    # new target position is not colliding with any obstacle
                    else:
                        while True:
                            posX, posY = self.get_random_object_pos('target')
                            if self.is_position_valid('target', posX, posY):
                                break
                        logging.info(f"Target new position: {posX}, {posY}")
                        self.sim.setObjectPosition(self.target, -1, [posX, posY, 0])
            else:
                # Use predefined target positions
                idx = np.random.randint(0, len(self.target_pos_samples))
                target_pos = self.target_pos_samples[idx]
                logging.info(f"Using predefined target position: {target_pos}")
                self.sim.setObjectPosition(self.target, -1, [target_pos[0], target_pos[1], 0])

            # If obstacles are not fixed, generate new ones
            if not self.params_scene["fixed_obs"]:
                if self.generator is not None:
                    # Generate new obstacles
                    logging.info(f"Regenerating new obstacles")
                    self.obstacles_objs = self.sim.callScriptFunction('generate_obs',self.handle_obstaclegenerators_script, [], [])
            logging.info("Environment RST done")


        # --- LOAD A PRECONFIGURED SCENE FOR TESTING ---
        else:
            if self.episode_idx < len(self.action_times):
                self.id_obstacle = 0

                # Load the CSV file and get all the tuples (action_time, target_id) just once
                if self.episode_idx==0:

                    # CSV path
                    csv_folder = os.path.join(self.paths["scene_configs"], self.scene_to_load_folder)  
                    scene_path = utils.find_scene_csv_in_dir(csv_folder)
                    if not os.path.exists(scene_path):
                        logging.error(f"[ERROR] CSV scene file not found: {scene_path}")
                        sys.exit()

                    self.df = pd.read_csv(scene_path)

                    # Get all rows that contain targets
                    self.target_rows = self.df[self.df['type'] == 'target'].reset_index(drop=True)
                    self.num_targets = len(self.target_rows)

                    if self.num_targets == 0:
                        logging.error("No targets found in the scene CSV.")
                        sys.exit()

                    # Get block size (number of episodes per target)
                    block_size = len(self.action_times) // self.num_targets 

                    # Get unique values of action times
                    unique_times = sorted(set(self.action_times))

                    # Calculate how many times each unique action time is repeated
                    reps = self.action_times.count(unique_times[0]) // self.num_targets

                    # Get a list (tuples) with (action_time, target_id) for each episode
                    # Mode A: alternate targets
                    if self.test_scene_mode == "alternate_targets":
                        logging.info("Test scene mode: alternate_targets")
                        for t in unique_times:
                            for target_id in range(self.num_targets):
                                self.tuples.extend([(t, target_id)] * reps)
                    # Mode B (default): alternate action times
                    else:
                        logging.info("Test scene mode: alternate_action_times")
                        for idx, t in enumerate(self.action_times):
                            target_id = idx // block_size
                            self.tuples.append((t, target_id))
                    logging.debug("Tuples (action_time, target_id) for each episode:", self.tuples)
                # Get what target will be used for each episode
                target_idx = min(self.tuples[self.episode_idx][1], self.num_targets - 1)  

                # Set action time for the episode new episode
                if self.action_times != [] and self.action_times is not None:
                    if self.episode_idx < len(self.action_times):
                        self._rltimestep = self.tuples[self.episode_idx][0]
                        logging.info(f"Action time set to {self._rltimestep}")

                # Initialize target counter
                current_target_idx = 0

                for _, row in self.df.iterrows():
                    x, y = row['x'], row['y']
                    z = 0.06969 if row['type'] == "robot" else 0.0  # Set height for placing the robot

                    if row['type'] == 'robot':
                        self.sim.setObjectPosition(self.robot_baselink, -1, [x, y, z])
                        theta = float(row['theta']) if 'theta' in row and not pd.isna(row['theta']) else 0
                        self.sim.setObjectOrientation(self.robot_baselink, -1, [0, 0, theta])

                    elif row['type'] == 'target':
                        if current_target_idx == target_idx:
                            self.sim.setObjectPosition(self.target, -1, [x, y, 0])
                        current_target_idx += 1

                    elif row['type'] == 'obstacle':
                        self.id_obstacle += 1
                        self.generate_obs_from_csv(row)

                logging.info(f"Scene recreated with {self.id_obstacle} obstacles.")
                logging.info(f"Episode {self.episode_idx}: Using target #{target_idx} with position {self.target_rows.iloc[target_idx][['x','y']].tolist()}")


        # --- SAVE CURRENT SCENE CONFIGURATION FOR FURTHER TESTING ---
        if self.save_scene:            
            # Create list to save all the elements
            scene_elements = []

            # Get and save the position and orientation of the robot
            robot_pos = self.sim.getObjectPosition(self.robot_baselink, -1)
            robot_ori = self.sim.getObjectOrientation(self.robot_baselink, -1)
            scene_elements.append(["robot", robot_pos[0], robot_pos[1], robot_ori[2]]) 

            # Get and save target position
            target_pos = self.sim.getObjectPosition(self.target, -1)
            scene_elements.append(["target", target_pos[0], target_pos[1]])

            # Get obstacles (assuming that they are located under self.generator object in Coppelia scene)
            obstacles = self.sim.getObjectsInTree(self.generator, self.sim.handle_all, 1)
            for obs_handle in obstacles:
                obs_pos = self.sim.getObjectPosition(obs_handle, -1)
                scene_elements.append(["obstacle", obs_pos[0], obs_pos[1]])
            
            csv_path = os.path.join(self.save_scene_csv_folder, f"scene_{self.episode_idx}.csv")
                
            # Save CSV file
            with open(csv_path, mode="w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(["type", "x", "y", "theta"])
                writer.writerows(scene_elements)

            logging.info(f"Scene saved in CSV: {csv_path}")

        self.episode_idx = self.episode_idx + 1

        # Set first reset flag to True
        if (self.episode_idx<=1):
            self.first_reset_done = True


    # --------------------------------
    # ----- HELPERS: AGENT STEP ------
    # --------------------------------

    def _compute_and_send_lat(self) -> None:
        """Compute Last Action Time (LAT) and send it to the RL side.

        Called when a new step is received, before executing it. Hanlde first-reset case also.
        """
        if self.first_reset_done and not self.lat_reset:
            self._lastactiont0_sim = 0.0
            self._lastactiont0_wall = 0.0
            self.lat_sim = 0.0
            self.lat_wall = 0.0
            self.lat_reset = True

        if self.reset_flag:
            logging.info(f"[Compute LAT] LAT sim: {round(self.lat_sim, 4)}. LAT wall: {round(self.lat_wall, 4)}")
            self._commstoRL.stepSendLastActDur(self.lat_sim, self.lat_wall)
            logging.debug("[Compute LAT] LAT sent")

            self.episode_start_time_sim = self.sim.getSimulationTime()
            self.episode_start_time_wall = self.sim.getSystemTime()

            self.current_sim_offset_time = self.sim.getSimulationTime() - self.episode_start_time_sim
            self.current_wall_offset_time = self.sim.getSystemTime() - self.episode_start_time_wall

            self._lastactiont0_sim = self.current_sim_offset_time
            self._lastactiont0_wall = self.current_wall_offset_time
            self.reset_flag = False

        else:
            self.current_sim_offset_time = self.sim.getSimulationTime() - self.episode_start_time_sim
            self.current_wall_offset_time = self.sim.getSystemTime() - self.episode_start_time_wall

            self.lat_sim = self.current_sim_offset_time - self._lastactiont0_sim
            self.lat_wall = self.current_wall_offset_time - self._lastactiont0_wall

            self._lastactiont0_sim = self.current_sim_offset_time
            self._lastactiont0_wall = self.current_wall_offset_time

            logging.info(f"[Compute LAT]LAT sim: {round(self.lat_sim, 4)}. LAT wall: {round(self.lat_wall, 4)}")
            self._commstoRL.stepSendLastActDur(self.lat_sim, self.lat_wall)
            logging.debug("[Compute LAT] LAT sent")


    def _handle_reset_standard(self) -> None:
        """Handle Reset in standard mode: randomize scene and send observation."""
        self.reset_simulator()

        observation = self.get_observation_space()
        logging.info(f"[Reset (standard)] Obs send RESET: { {key: round(value, 3) for key, value in observation.items()} }")

        simTime = self.sim.getSimulationTime() - self.initial_simTime
        self._commstoRL.resetSendObs(observation, simTime)


    def _handle_reset_pv(self) -> None:
        """Handle Reset in path-version mode: teleport robot and place target."""
        # test_map mode: use predefined test cases with specific robot and target positions
        if self.test_map_mode and self.test_cases:
            tc_idx = self.current_test_case_idx
            if tc_idx < len(self.test_cases):
                test_case = self.test_cases[tc_idx]

                robot_x, robot_y, robot_yaw = test_case["robot_pose"]
                target_x, target_y = test_case["target_pos"]

                self.new_robot_pose = (robot_x, robot_y, 0.0, robot_yaw)
                self.sim.callScriptFunction('rp_tp', self.handle_robot_scripts, self.new_robot_pose)
                logging.info(f"[Reset (pv)] Robot teleported to: ({robot_x:.2f}, {robot_y:.2f}, yaw={robot_yaw:.3f})")

                self.sim.setObjectPosition(self.target, -1, [target_x, target_y, 0])
                logging.info(f"[Reset (pv)] Target placed at: ({target_x:.2f}, {target_y:.2f})")

                self.current_test_case_idx = tc_idx + 1
                self.first_reset_done = True

            observation = self.get_observation_space()
            logging.info(f"[Reset (pv)] Obs send RESET: { {key: round(value, 3) for key, value in observation.items()} }")

            simTime = self.sim.getSimulationTime() - self.initial_simTime
            info = {"posX": float(self.new_robot_pose[0]), "posY": float(self.new_robot_pose[1])}
            self._commstoRL.resetSendObs(observation, simTime, info)

        # test_path mode: teleport the robot along the path and place the target in front of it, alternating between predefined path positions
        else:
            # Reset obstacles on the first reset
            if self.place_obstacles_flag and not self.first_reset_done:
                if self.generator is not None:
                    logging.info("Regenerating new obstacles as it's the first RESET...")
                    self.obstacles_objs = self.sim.callScriptFunction(
                        'generate_obs', self.handle_obstaclegenerators_script,
                        [], self.path_base_pos_samples
                    )

            # Advance robot position if all trials for this sample are done
            if not self.first_reset_done or self.current_trial_idx_pv == self.trials_per_sample - 1:
                if not self.first_reset_done:
                    self.first_reset_done = True
                else:
                    self.current_sample_idx_pv += 1
                    self.current_trial_idx_pv = 0

                sample = self.path_pos_samples[self.current_sample_idx_pv]
                if isinstance(sample, dict):
                    rp = sample['robot_pose']
                    self._current_sample_target_pos = sample.get('target_pos', None)
                else:
                    rp = sample
                    self._current_sample_target_pos = None
                # Normalize to 4-element tuple (x, y, z, yaw)
                if len(rp) == 3:
                    self.new_robot_pose = (float(rp[0]), float(rp[1]), 0.0, float(rp[2]))
                elif len(rp) == 4:
                    self.new_robot_pose = (float(rp[0]), float(rp[1]), float(rp[2]), float(rp[3]))
                else:
                    raise ValueError(f"robot_pose must have 3 or 4 elements, got {len(rp)}")
                print(self.new_robot_pose)
                self.sim.callScriptFunction('rp_tp', self.handle_robot_scripts, self.new_robot_pose)
                logging.info(f"[Reset (pv)] Robot teleported to new pos: {self.new_robot_pose}")
            else:
                self.current_trial_idx_pv += 1

            # Reset the target
            posX, posY = None, None

            # If the sample dict provided a target_pos, use it directly
            if getattr(self, '_current_sample_target_pos', None) is not None:
                posX, posY = float(self._current_sample_target_pos[0]), float(self._current_sample_target_pos[1])
            elif self.random_target_flag:
                while True:
                    posX, posY = self.get_random_object_pos('target')
                    if self.is_position_valid('target', posX, posY):
                        break
            elif self.current_trial_idx_pv == 0:
                if self.grid_positions_flag:
                    posX, posY = self.get_target_relative_to_robot(
                        self.new_robot_pose, self.perimeterRadius, self.robot_target_ori
                    )
                else:
                    posX, posY = self.get_target_in_path_pos(self.new_robot_pose, self.perimeterRadius)

            if posX is not None and posY is not None:
                logging.info(f"[Reset (pv)] Target new position: {posX}, {posY}")
                self.sim.setObjectPosition(self.target, -1, [posX, posY, 0])

            observation = self.get_observation_space()
            logging.info(f"[Reset (pv)] Obs send RESET: { {key: round(value, 3) for key, value in observation.items()} }")

            simTime = self.sim.getSimulationTime() - self.initial_simTime
            info = {"posX": float(self.new_robot_pose[0]), "posY": float(self.new_robot_pose[1])}
            self._commstoRL.resetSendObs(observation, simTime, info)


    def _check_collision_during_movement(self):
        '''
        Check if there has been a collision during the robot movement.
        Returns:
            bool: True if there was a collision, False otherwise.
        '''
        if self.laser is not None:
            # Get laser readings once
            laser_obs = self.sim.callScriptFunction(
                'laser_get_observations',
                self.handle_laser_get_observation_script
            )
            logging.debug(f"[Collision check] Laser values during movement: {laser_obs}")

            # Rename for simplicity
            p = self.params_env
            crit = p["max_crash_dist_critical"]
            norm = p["max_crash_dist"]

            crashed = False

            # Turtlebot case -> Off-center configuration: We have a different distance threshold for lateral measurements, 
            # as the sensor is not placed at the robot center but at the front, so the lateral measurements need a bigger 
            # threshold to avoid collisions

            if self.robot_name == "TurtleBot":
                logging.debug(f"[Collision check] Off-centered Laser configuration. Thresholds: Crit: {crit}, Norm: {norm}")
                # 4-beam layout: 0,3 -> critical; 1,2 -> normal
                if (
                    laser_obs[0] < crit
                    or laser_obs[3] < crit
                    or any(laser_obs[i] < norm for i in (1, 2))
                ):
                    crashed = True

            # BurgerBot case or any other setup by default -> Same distance threshold for all the laser measurements
            else:
                logging.debug(f"[Collision check] Centered Laser configuration. Threshold: {crit}")
                if any(d < crit for d in laser_obs):
                    crashed = True

            return crashed
        else:
            return False    


    # ----------------------------
    # ------- AGENT STEP ---------
    # ----------------------------

    def agent_step(self):
        """A step of the agent: process RL instructions and execute actions.

        In standard mode ('pv_mode=False'), resets use 'reset_simulator()' and
        collision detection is active during action execution.  In path-version mode
        ('pv_mode=True'), resets teleport the robot along sampled path positions
        and actions are shared via CoppeliaSim string signals.

        Returns:
            dict: The action to be executed, or '{}' on FINISH.
        """
        action = self._lastaction

        # --- WAITING STATE: executing last action ---
        if not self._waitingforrlcommands:
            self.current_sim_offset_time = self.sim.getSimulationTime() - self.episode_start_time_sim
            action_duration = self._pv_step_duration if self.pv_mode else self._rltimestep

            if self.current_sim_offset_time - self._lastactiont0_sim >= action_duration:
                logging.info("[Main comms loop] Action time completed.")
                self.execute_cmd_vel = False
                self.ts_received = False

                observation = self.get_observation_space()
                logging.info(f"[Main comms loop] Obs send STEP: { {key: round(value, 3) for key, value in observation.items()} }")

                simTime = self.sim.getSimulationTime() - self.initial_simTime
                crash = self.crash_flag if not self.pv_mode else False
                self._commstoRL.stepSendObs(observation, simTime, crash)
                self.crash_flag = False
                self._waitingforrlcommands = True

            # Collision detection during movement (standard mode only)
            elif not self.pv_mode:
                if not self.crash_flag:
                    if self._check_collision_during_movement():
                        logging.info("[Collision check] Collision detected during movement.")
                        self.crash_flag = True

            action = self._lastaction

        # --- READING STATE: waiting for new RL command ---
        else:
            rl_instruction = self._commstoRL.readWhatToDo()

            if rl_instruction is not None:
                self.training_started = True

                # STEP received
                if rl_instruction[0] == AgentSide.WhatToDo.REC_ACTION_SEND_OBS:
                    logging.info("Received: REC_ACTION_SEND_OBS")

                    action = rl_instruction[1]
                    logging.info(f"[Main comms loop] Action rec: { {key: round(value, 3) for key, value in action.items()} }")

                    # Expose action to other CoppeliaSim scripts (pv mode)
                    if self.pv_mode:
                        self.sim.setStringSignal("agent_shared_data", json.dumps(action))

                    if self.variable_timestep:
                        self._update_rltimestep(action)

                    self._compute_and_send_lat()

                    # Publish LAT signal for robot-side graphs (standard mode)
                    if not self.pv_mode:
                        self.sim.setFloatSignal('latValueSignal', float(self.lat_sim))
                        logging.debug(f"[Main comms loop] Signal latValueSignal set to {self.lat_sim}")

                    self._waitingforrlcommands = False
                    self.execute_cmd_vel = True
                    self.ts_received = True

                # RESET received
                elif rl_instruction[0] == AgentSide.WhatToDo.RESET_SEND_OBS:
                    logging.info("Received: RESET_SEND_OBS")

                    if self.pv_mode:
                        self._handle_reset_pv()
                    else:
                        self._handle_reset_standard()

                    self.reset_flag = True
                    action = None

                # FINISH received
                elif rl_instruction[0] == AgentSide.WhatToDo.FINISH:
                    logging.info("Received: FINISH")
                    self.finish_rec = True
                    return {}

                else:
                    logging.error(f"Received unexpected instruction: {rl_instruction}")

        self._lastaction = action
        return action


# -------------------------------
# ------ HARDCODED CLASSES ------
# -------------------------------

class BurgerBotAgent(CoppeliaAgent):
    def __init__(self, sim, params_scene, params_env, paths, file_id, verbose, ip_address, comms_port=49054):
        """
        Custom agent for the BurgerBot robot simulation in CoppeliaSim, inherited from CoppeliaAgent class.

        Args:
            sim: Coppelia object for handling the scene's objects.
            params_scene (dict): Dictionary of parameters specific to the scene.
            params_env (dict): Dictionary of parameters for configuring the agent.
            paths (dict): Dictionary of paths for saving/loading data.
            file_id (str): Identifier for the current run, used in saving files.
            verbose (int): Verbosity level for logging.
            ip_address (str): IP address for communication with the RL side.
            comms_port (int, optional): The port to be used for communication with the agent system. Defaults to 49054.
            
        Attributes:
            robot_name (str): Name of the robot.
            robot (CoppeliaObject): Robot object in CoppeliaSim scene.
            robot_baselink (CoppeliaObject): Object of the robot's basein CoppeliaSim scene.
            laser (CoppeliaObject): Lase object in CoppeliaSim scene.
            handle_laser_get_observation_script (int): Handle of the script function to get laser observations.
            handle_robot_scripts (int): Handle of the robot main script, used for calling functions like 'rp_tp' to teleport the robot.
        """
        super(BurgerBotAgent, self).__init__(sim, params_scene, params_env, paths, file_id, verbose, ip_address, comms_port)

        self.robot_name = "BurgerBot"
        self.robot = sim.getObject("/Burger")
        self.robot_baselink = self.robot
        self.laser=sim.getObject('/Burger/Laser')
        self.handle_laser_get_observation_script=sim.getScript(1,self.laser,'laser_get_observations')
        self.handle_robot_scripts = sim.getScript(1, self.robot)

        logging.info(f"BurgerBot Agent created successfully using port {comms_port}.")


class TurtleBotAgent(CoppeliaAgent):
    def __init__(self, sim, params_scene, params_env, paths, file_id, verbose, ip_address, comms_port=49054):
        """
        Custom agent for the TurtleBot robot simulation in CoppeliaSim, inherited from CoppeliaAgent class.

        Args:
            sim: Coppelia object for handling the scene's objects.
            params_scene (dict): Dictionary of parameters specific to the scene.
            params_env (dict): Dictionary of parameters for configuring the agent.
            paths (dict): Dictionary of paths for saving/loading data.
            file_id (str): Identifier for the current run, used in saving files.
            verbose (int): Verbosity level for logging.
            ip_address (str): IP address for communication with the RL side.
            comms_port (int, optional): The port to be used for communication with the agent system. Defaults to 49054.

        Attributes:
            robot_name (str): Name of the robot.
            robot (CoppeliaObject): Robot object in CoppeliaSim scene.
            robot_baselink (CoppeliaObject): Object of the robot's basein CoppeliaSim scene.
            laser (CoppeliaObject): Lase object in CoppeliaSim scene.
            handle_laser_get_observation_script (int): Handle of the script function to get laser observations.
            handle_robot_scripts (int): Handle of the robot main script, used for calling functions like 'rp_tp' to teleport the robot.

        """
        super(TurtleBotAgent, self).__init__(sim, params_scene, params_env, paths, file_id, verbose, ip_address, comms_port)

        self.robot_name = "TurtleBot"
        self.robot=sim.getObject('/Turtlebot2')
        self.robot_baselink=sim.getObject('/Turtlebot2/base_link_respondable')
        self.laser=sim.getObject('/Turtlebot2/fastHokuyo_ROS2')
        self.handle_laser_get_observation_script=sim.getScript(1,self.laser,'laser_get_observations')
        self.handle_robot_scripts = sim.getScript(1, self.robot)
        
        logging.info(f"TurtleBot Agent created successfully using port {comms_port}.")
