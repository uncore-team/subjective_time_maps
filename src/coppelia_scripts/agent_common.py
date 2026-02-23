"""
Common utilities for CoppeliaSim agent scripts.

This module provides shared functionality between rl_script_copp.py (normal mode)
and rl_script_pv_copp.py (path-probe mode), reducing code duplication while
keeping each script's specific logic separate.

Functions:
    autoload_agent_plugins: Dynamically imports agent plugin modules.
    initialize_agent: Creates and configures the agent instance.
    configure_scene_and_trajectory: Sets up scene loading and trajectory saving paths.
    setup_paths_and_logging: Configures logging and filesystem paths.
"""

import importlib
import logging
import os
import pkgutil
import sys
import traceback
from typing import Any, Dict, Optional, Tuple

from common import utils
from common.coppelia_agents import BurgerBotAgent, TurtleBotAgent
from plugins.agents import get_agent_factory


def autoload_agent_plugins(base_path: str) -> None:
    """
    Import all agent plugin modules for self-registration.

    Dynamically imports Python modules from the 'plugins.agents' package,
    allowing each module to register its agent factory via get_agent_factory().

    Args:
        base_path: Root folder of the RL-Coppelia project. Used to extend
            sys.path for importing 'plugins.*' and 'agents.*' modules.

    Note:
        Errors during individual module imports are logged but do not stop
        the loading process.
    """
    src_dir = os.path.join(base_path, "src")
    if os.path.isdir(src_dir) and src_dir not in sys.path:
        sys.path.insert(0, src_dir)  # For 'plugins.*'

    if base_path not in sys.path:
        sys.path.insert(0, base_path)  # For 'agents.*'

    try:
        pkg = importlib.import_module("plugins.agents")
        for finder, name, ispkg in pkgutil.iter_modules(pkg.__path__, "plugins.agents."):
            try:
                importlib.import_module(name)
                logging.info(f"[plugins] Imported: {name}")
            except Exception:
                logging.error(f"[plugins] Failed to import {name}")
                logging.debug(traceback.format_exc())
    except Exception:
        logging.error(f"Agent plugins autoload failed:\n{traceback.format_exc()}")


def initialize_agent(
    sim: Any,
    robot_name: str,
    params_scene: Dict,
    params_env: Dict,
    paths: Dict,
    file_id: str,
    verbose: int,
    ip_address: str,
    comms_port: int
) -> Any:
    """
    Create and configure the agent instance.

    Attempts to instantiate the agent using a registered plugin factory first.
    If no plugin is found, falls back to hardcoded agent classes (TurtleBotAgent
    or BurgerBotAgent).

    Args:
        sim: CoppeliaSim API object.
        robot_name: Name of the robot (e.g., "turtleBot", "burgerBot").
        params_scene: Scene configuration parameters.
        params_env: Environment configuration parameters.
        paths: Dictionary containing filesystem paths (logs, models, etc.).
        file_id: Unique identifier for this experiment session.
        verbose: Logging verbosity level (0-3).
        ip_address: IP address for RL-Coppelia communication.
        comms_port: Port number for RL-Coppelia communication.

    Returns:
        Configured agent instance (TurtleBotAgent, BurgerBotAgent, or plugin agent).

    Raises:
        ValueError: If robot_name is unknown and no plugin factory is registered.
    """
    factory = get_agent_factory(robot_name)
    
    if factory is not None:
        agent = factory(sim, params_scene, params_env, paths, file_id, verbose, ip_address, comms_port=comms_port)
        logging.info(
            f"[plugins] Agent created via plugin for '{robot_name}'. "
            f"IP address: {ip_address}. "
            f"Comms port: {comms_port}"
        )
        return agent

    # Fallback to hardcoded agents
    logging.info(f"[plugins] No agent plugin found for '{robot_name}'. Using hardcoded class.")
    
    if robot_name == "turtleBot":
        agent = TurtleBotAgent(sim, params_scene, params_env, paths, file_id, verbose, ip_address, comms_port)
    elif robot_name == "burgerBot":
        agent = BurgerBotAgent(sim, params_scene, params_env, paths, file_id, verbose, ip_address, comms_port)
        agent.robot_baselink = agent.robot
    else:
        raise ValueError(f"Unknown robot name '{robot_name}' and no plugin found.")
    
    logging.info(f"Agent created via hardcoded class for '{robot_name}'. IP: {ip_address}. Comms port: {comms_port}.")
    return agent


def configure_scene_and_trajectory(
    agent: Any,
    model_name: Optional[str],
    scene_to_load_folder: str,
    obstacles_csv_folder: str,
    save_scene: bool,
    save_traj: bool,
    model_ids: Optional[list],
    action_times: Optional[list]
) -> None:
    """
    Configure scene loading and trajectory saving behavior.

    Sets up paths for scene configurations and trajectory CSVs, creates
    necessary directories, and configures the agent's saving flags.

    Args:
        agent: Agent instance to configure.
        model_name: Name of the model being tested (None for training).
        scene_to_load_folder: Folder containing predefined scene configurations.
        obstacles_csv_folder: Folder containing obstacle placement CSVs.
        save_scene: Whether to save scene configurations.
        save_traj: Whether to save robot trajectories.
        model_ids: List of model identifiers for multi-model testing.
        action_times: List of action timing configurations.

    Note:
        If a predefined scene is loaded, trajectory saving is automatically enabled.
    """
    agent.scene_to_load_folder = scene_to_load_folder
    agent.obstacles_csv_folder = obstacles_csv_folder
    agent.save_scene = save_scene
    agent.model_ids = model_ids
    agent.action_times = action_times

    # Set trajectory save folder based on test mode
    if model_name is None:
        agent.save_traj_csv_folder = os.path.join(
            agent.scene_configs_path,
            agent.scene_to_load_folder,
            "trajs"
        )
    else:
        agent.save_traj_csv_folder = os.path.join(
            agent.save_trajs_path,
            f"{model_name}_testing",
            "trajs"
        )

    # Create scene configuration directory if needed
    if agent.save_scene:
        os.makedirs(agent.save_scene_csv_folder, exist_ok=True)
        logging.info(f"Scene configurations will be saved in: {agent.save_scene_csv_folder}.")

    # Trajectory saving logic: force-enable when loading predefined scenes
    if scene_to_load_folder == "" or scene_to_load_folder is None:
        agent.save_traj = save_traj
        if agent.save_traj:
            os.makedirs(agent.save_traj_csv_folder, exist_ok=True)
            logging.info(f"Trajectories will be saved in: {agent.save_traj_csv_folder}.")
    else:
        agent.save_traj = True
        os.makedirs(agent.save_traj_csv_folder, exist_ok=True)
        logging.info(
            f"Scene configuration inside {agent.scene_to_load_folder} will be loaded "
            f"and trajectory will be saved inside it."
        )


def setup_paths_and_logging(
    sim: Any,
    base_path: str,
    robot_name: str,
    model_name: Optional[str],
    verbose: int,
    comm_side: str = "agent"
) -> Tuple[Dict, str]:
    """
    Configure filesystem paths and logging for the experiment.

    Generates standardized directory structure for logs and TensorBoard files,
    determines the next available experiment ID, and configures Python logging.

    Args:
        sim: CoppeliaSim API object (currently unused, for future compatibility).
        base_path: Root directory of the RL-Coppelia project.
        robot_name: Name of the robot being used.
        model_name: Name of the model (None for training mode).
        verbose: Logging verbosity level (0-3).
        comm_side: Communication side identifier (default: "agent").

    Returns:
        Tuple containing:
            - paths (dict): Dictionary with keys 'script_logs', 'tf_logs', 'models', etc.
            - file_id (str): Unique experiment identifier for file naming.
    """
    paths = utils.get_robot_paths(base_path, robot_name, agent_logs=True)
    file_id = utils.get_file_index(model_name, paths["tf_logs"], robot_name)
    utils.logging_config(
        paths["script_logs"],
        comm_side,
        robot_name,
        file_id,
        log_level=logging.INFO,
        verbose=verbose
    )
    return paths, file_id


def start_agent_communication(
    agent: Any,
    retry_delay: int = 10,
    max_retry_time: Optional[int] = None
) -> bool:
    """
    Establish communication with the external RL process.

    Attempts to connect the agent to the RL-side server, retrying on failure
    with configurable delay and timeout.

    Args:
        agent: Agent instance with communication capabilities.
        retry_delay: Time in seconds between connection attempts.
        max_retry_time: Maximum total time to retry (None for infinite retries).

    Returns:
        True if communication was successfully established, False otherwise.
    """
    success = agent.start_communication(retry_delay=retry_delay, max_retry_time=max_retry_time)
    if success:
        logging.info("Agent initialized and communication with RL side established.")
    else:
        logging.error("Failed to establish communication with RL side.")
    return success


def set_optional_sample_positions(
    agent: Any,
    target_pos_samples: Optional[list],
    robot_pos_samples: Optional[list]
) -> None:
    """
    Configure optional position samples for testing.

    Sets predefined target and robot positions on the agent if provided,
    typically used for systematic testing across a map.

    Args:
        agent: Agent instance to configure.
        target_pos_samples: List of [x, y] target positions (None if not used).
        robot_pos_samples: List of [x, y] robot starting positions (None if not used).
    """
    if target_pos_samples is not None and target_pos_samples != []:
        agent.target_pos_samples = target_pos_samples
        logging.info(f"Configured {len(target_pos_samples)} target position samples.")

    if robot_pos_samples is not None and robot_pos_samples != []:
        agent.robot_pos_samples = robot_pos_samples
        logging.info(f"Configured {len(robot_pos_samples)} robot position samples.")