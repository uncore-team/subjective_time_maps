"""
Simulation-side handler for the RL-Coppelia integration.

This script runs inside CoppeliaSim and acts as the "environment side" of
the RL-Coppelia framework. It is responsible for:

    * Initializing the agent based on the robot type.
    * Handling communication with the external RL process.
    * Driving the robot by forwarding actions to the robot-side script.
    * Saving scene configurations and robot trajectories (optional).
    * Tracking simulation and real-time during training.

Callbacks expected by CoppeliaSim (threaded script):

    * sysCall_init():
        - One-time initialization when the script is created.
        - Sets up logging, paths and default parameters.

    * sysCall_thread():
        - Called once when the simulation starts.
        - Creates and configures the agent, and starts communication.

    * sysCall_sensing():
        - Called at each simulation step.
        - Queries the agent for actions, executes them, and logs trajectories.
"""

import logging
from pathlib import Path
import sys
from typing import Any
import rl_coppelia

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
    # Agent
    agent: Any

    # Paths and experiment ID
    paths: dict
    file_id: str

    # Sim API (provided by CoppeliaSim)
    sim: Any

self: Any = _CoppeliaSimContext()  # type: ignore


# -----------------------------------------
# -- GLOBALS 
# -----------------------------------------

# Variables to receive from RL side (via utils.py)
comm_side = "agent"
robot_name = ""
model_name = None
model_ids = None
comms_port = 49054
ip_address = ""
base_path = ""
params_scene = {}
params_env = {}
params_train = {}
verbose = 1
scene_to_load_folder = ""
obstacles_csv_folder = ""
fixed_vlin = None
save_scene = None
save_traj = None
action_times = None
target_pos_samples = None   # available target locations
robot_pos_samples = None    # available robot locations


# -----------------------------------------
# -- Coppelia callbacks
# -----------------------------------------

def sysCall_init():
    """
    Called at the beginning of the simulation to configure logging and path setup.
    """
    self.sim = require('sim')    # type: ignore

    # Setup paths and logging
    self.paths, self.file_id = agent_common.setup_paths_and_logging(
        self.sim, base_path, robot_name, model_name, verbose, comm_side
    )

    # Load agent plugins
    agent_common.autoload_agent_plugins(base_path)

    logging.info(" ----- START EXPERIMENT ----- ")


def sysCall_thread():
    """
    Create the agent and configure scene-dependent paths.

    This callback is executed once after the simulation starts (threaded
    context). It is responsible for:

        * Instantiating the agent (via plugin or built-in fallback).
        * Starting the communication with the external RL process.
        * Wiring scene / trajectory saving options into the agent.
    """
    # Create and initialize agent
    self.agent = agent_common.initialize_agent(
        self.sim, robot_name, params_scene, params_env, self.paths, self.file_id, verbose, ip_address, comms_port
    )
    
    # Start communication with RL process
    agent_common.start_agent_communication(self.agent, retry_delay=10, max_retry_time=None)

    # Scene / trajectory configuration
    agent_common.configure_scene_and_trajectory(
        self.agent, model_name, scene_to_load_folder, obstacles_csv_folder,
        save_scene, save_traj, model_ids, action_times
    )

    # Set optional sample positions for testing
    agent_common.set_optional_sample_positions(self.agent, target_pos_samples, robot_pos_samples)



def sysCall_sensing():
    """
    Main simulation step: query, execute and log agent actions.

    This callback is invoked at each simulation step. It performs:

        * One agent step (receive next action from RL side).
        * Execution of the action via the robot-side 'cmd_vel' script.
        * Optional path drawing via 'draw_path'.
        * Trajectory logging (robot base link positions).
        * Handling of the FINISH command to stop the experiment.
        * Basic tracking of simulation vs real time once training starts.
    """   
    # -----------------------------------------------
    # -- Action loop while the experiment is active
    # -----------------------------------------------
    if self.agent and not self.agent.finish_rec:
        # Loop for processing instructions from RL continuously until the agent receives a FINISH command.
        action = self.agent.agent_step()

        # If an action is received, execute it
        if action is not None:
            # With this check we avoid calling cmd_vel script repeteadly for the same action
            if self.agent.execute_cmd_vel and len(action)>0:
                logging.info("[Main comms loop] Execute action from the RL")
                if "linear" in action:
                    self.agent.sim.callScriptFunction('cmd_vel', self.agent.handle_robot_scripts, action["linear"], action["angular"])
                else:   # Fallback to fixed linear speed
                    self.agent.sim.callScriptFunction('cmd_vel', self.agent.handle_robot_scripts, fixed_vlin, action["angular"])

            if verbose == 3:
                if len(action)>0:
                    if "linear" in action:
                        self.agent.sim.callScriptFunction('draw_path', self.agent.handle_robot_scripts, action["linear"], action["angular"], self.agent.colorID)
                    else:
                        self.agent.sim.callScriptFunction('draw_path', self.agent.handle_robot_scripts, fixed_vlin, action["angular"], self.agent.colorID)
                    if self.agent._waitingforrlcommands:
                        self.agent.colorID +=1
            self.agent.execute_cmd_vel = False
        
        # Save current robot position for later saving it csv file
        if self.agent.episode_idx >=1 and self.agent.save_traj:
            position = self.agent.sim.getObjectPosition(self.agent.robot_baselink, -1)
            self.agent.trajectory.append({"x": position[0], "y": position[1]})
            logging.debug(f"x pos: {position[0]}; y pos: {position[1]}")

    # ----------------------------------------------
    # -- FINISH command --> Finish the experiment
    # ----------------------------------------------
    if self.agent and self.agent.finish_rec:
        # Stop the robot
        logging.info("[FINISH] Reset speed to 0")
        self.sim.callScriptFunction('cmd_vel',self.agent.handle_robot_scripts,0,0)
        if verbose == 3:
            self.sim.callScriptFunction('draw_path', self.agent.handle_robot_scripts, 0,0, self.agent.colorID)
        logging.info(" ----- END OF EXPERIMENT ----- ") 
        
        # Stop the simulation
        self.sim.stopSimulation()

    # ----------------------------------------------
    # -- Time tracking setup
    # ----------------------------------------------
    if self.agent and not self.agent.training_started:
        self.agent.initial_realTime = self.sim.getSystemTime()
        self.agent.initial_simTime = self.sim.getSimulationTime()
    
    elif self.agent and self.agent.training_started:
        simTime = self.sim.getSimulationTime() - self.agent.initial_simTime
        logging.debug("SIM Time:", simTime)
        realTime = self.sim.getSystemTime() - self.agent.initial_realTime
        logging.debug("REAL Time:", realTime)

    