# Agent_Script: minimal version for 'test_map.py' and 'test_path.py' functionalities.
# -------------------------------------------------------
# Responsibilities:
#   - Create the Agent (TurtleBot/BurgerBot or plugin).
#   - Open the RL socket (spindecoupler).
#   - Initialize the path sampling once (delegated to Robot_Script.rp_init).
#   - Wait for RL instructions:
#       * RESET: prepare to deliver the first observation.
#       * STEP: TP to current path sample, randomize target, wait N frames,
#               read observation via Agent.get_observation(), send it to RL,
#               advance (trial -> sample).
#       * FINISH: stop simulation.
#
# Notes:
#   - No RemoteAPI stepping is used; we count frames in sysCall_sensing().
#   - Target randomization and observation reuse the Agent's own methods.
#   - Path sampling and teleport are implemented in Robot_Script (rp_init/rp_tp).


import logging
from pathlib import Path
import sys
from typing import Any
import rl_coppelia
import numpy as np  # DO NOT REMOVE

# Append rl_coppelia source folder to sys.path
pkg_file = Path(rl_coppelia.__file__).resolve()
pkg_dir = pkg_file.parent
src_dir = pkg_dir.parent
sys.path.append(str(src_dir))

from coppelia_scripts import agent_common


# ----- Simulated self object for CoppeliaSim context -----
# CoppeliaSim provides 'self' internally; this prevents IDE errors
class _CoppeliaSimContext:
    """Mock object to simulate CoppeliaSim internal 'self' context."""   
    # Agent and initialization flags
    agent: Any
    init_done: bool
    comm_init_done: bool

    # Paths and experiment ID
    paths: dict
    file_id: str

    # Sim API (provided by CoppeliaSim)
    sim: Any

self: Any = _CoppeliaSimContext()  # type: ignore


# Global variables (values received from RL side via utils.py)
# These will be overwritten by CoppeliaSim with actual values
robot_name = ""
model_name = None
model_ids = None
comms_port = 49054
ip_address = ""
base_path = ""
comm_side = "agent"
params_scene = {}
params_env = {}
verbose = 1
scene_to_load_folder = ""
obstacles_csv_folder = ""
save_scene = None
save_traj = None
action_times = None

# Path-version specific variables
path_alias = ""
sample_step_m = None
trials_per_sample = None
n_samples = None
n_extra_poses = None
delta_deg = None
robot_world_ori = None
robot_target_ori = None
place_obstacles_flag = None
random_target_flag = None
base_pos_samples = None
test_cases = None
fixed_target_pos = None
test_map_mode = False


# -------------------------------
# ------- MAIN FUNCTIONS --------
# -------------------------------


def sysCall_init():
    """
    Called at the beginning of the simulation to configure logging and path setup. 
    It also receives variable data from the RL side.
    """
    self.sim = require('sim')    # type: ignore

    # Setup paths and logging
    self.paths, self.file_id = agent_common.setup_paths_and_logging(
        self.sim, base_path, robot_name, model_name, verbose, comm_side
    )

    # Load agent plugins and create agent
    agent_common.autoload_agent_plugins(base_path)
    self.agent = agent_common.initialize_agent(
        self.sim, robot_name, params_scene, params_env, self.paths, self.file_id, verbose, ip_address, comms_port
    )
    
    self.init_done = True


def sysCall_thread():
    """
    This is executed once after sysCall_init ends the initialization.
    """
    if not self.init_done: 
        return

    # Start communication with RL process
    self.comm_init_done = agent_common.start_agent_communication(self.agent)


    # Configure scene and trajectory behavior 
    self.agent.pv_mode = True
    self.agent.place_obstacles_flag = place_obstacles_flag
    self.agent.random_target_flag = random_target_flag
    self.agent.trials_per_sample = trials_per_sample

    # Use common configuration function
    agent_common.configure_scene_and_trajectory(
        self.agent, model_name, scene_to_load_folder, obstacles_csv_folder,
        save_scene, save_traj, model_ids, action_times
    )

    # Path sampling: test_path/test_map mode
    _robot_script = self.agent.handle_robot_scripts
    
    # test_map/test_path mode: use pre-computed test cases from RL side
    if test_map_mode and test_cases is not None:
        logging.info(f"[test_path_v2 mode] Using {len(test_cases)} pre-computed test cases from RL side.")
        self.agent.test_map_mode = True
        self.agent.test_cases = test_cases
        self.agent.fixed_target_pos = fixed_target_pos
        self.agent.current_test_case_idx = 0
        
        # Extract unique robot positions for compatibility
        unique_positions = []
        seen = set()
        for tc in test_cases:
            pos = (tc["robot_pose"][0], tc["robot_pose"][1])
            if pos not in seen:
                seen.add(pos)
                unique_positions.append(pos)
        self.agent.path_base_pos_samples = unique_positions
        self.agent.path_pos_samples = test_cases # Full test cases
        self.agent.trials_per_sample = 1  # Each test case is a single trial
        
    elif base_pos_samples is None or base_pos_samples==[]:
        self.agent.path_handle = self.sim.getObject(path_alias)
        self.agent.path_pos_samples, self.agent.path_base_pos_samples = self.agent.sim.callScriptFunction('rp_init', _robot_script, n_samples, n_extra_poses, path_alias)
    
    else:
        logging.info(f"Positions have been provided by RL.")
        self.agent.grid_positions_flag = True
        self.agent.robot_target_ori = robot_target_ori

        # Valid positions
        self.agent.path_base_pos_samples = base_pos_samples
        logging.info(f"Total number of grid positions: {len(self.agent.path_pos_samples)}.")

        # We augment them by changing the orientation of the robot
        self.agent.path_pos_samples = self.agent.sim.callScriptFunction('augment_base_poses', _robot_script, self.agent.path_base_pos_samples, n_extra_poses, delta_deg, 0, robot_world_ori)
    
    logging.info(f"Total number of scenarios to test: {len(self.agent.path_pos_samples)}.")
    logging.info(" ----- START EXPERIMENT ----- ")


def sysCall_sensing():
    """
    Called at each simulation step. Executes the agent action and logs trajectories.
    """
    if self.agent and self.comm_init_done and not self.agent.finish_rec:
        # Loop for processing instructions from RL continuously until the agent receives a FINISH command.
        action = self.agent.agent_step()

        # If an action is received, execute it
        if action is not None:
            # With this check we avoid calling cmd_vel script repeteadly for the same action
            if self.agent.ts_received:
                if len(action)>0:
                    logging.info(f"Sample idx: {self.agent.current_sample_idx_pv}. Trial idx: {self.agent.current_trial_idx_pv}")
                    logging.info("Action has been predicted. Changing target and/or robot after action finishes")
            self.agent.ts_received = False

    # FINISH command --> Finish the experiment
    if self.agent and self.agent.finish_rec:
        # Stop the simulation
        logging.info(" ----- END OF EXPERIMENT ----- ")
        self.sim.stopSimulation() 

    # Time tracking setup
    if self.agent and not self.agent.training_started:
        self.agent.initial_realTime = self.sim.getSystemTime()
        self.agent.initial_simTime = self.sim.getSimulationTime()
    