import csv
import logging
import math
import re
from typing import Any, Dict, List, Sequence
import numpy as np
import gymnasium as gym
from gymnasium import spaces
from abc import ABC, abstractmethod

from common import utils


class CoppeliaEnv(gym.Env, ABC):

    def __init__(self, params_scene, params_env):
        """
        Custom environment for simulation agents in CoppeliaSim, inherited from gym.Env.
        
        This environment interfaces with the AgentSide class (from spindecoupler package) to
        send actions and receive observations in a reinforcement learning setup. The environment 
        simulates the robot's movement in response to linear and angular speed commands using a 
        CoppeliaSim scene.
        
        Args:
            params_env (dict): Dictionary of parameters for configuring the environment
        
        Attributes:
            action_space (gym.spaces.Box): Action space of the environment.
            observation_space (gym.spaces.Box): Observation space of the environment.
            _commstoagent (RLSide): Object to interact with the agent system (it will be used by the baseline).
            count (int): Step counter to track the number of steps taken in the episode.
            time_elapsed (float): Time counter to track the duration of the episode.
            n_ep (int): Episode counter.
            terminated (bool): Flag indicating if the current episode reached a terminal state.
            truncated (bool): Flag indicating if the current episode was truncated.
            reward (float): Reward for the current step.
            action_dic (dic): Dictionary of actions to be sent to CoppeliaSim agent.
        
        Methods:
            step(action): Executes a single step in the environment based on the provided action.
            reset(): Resets the environment to its initial state and returns the initial observation.
            _calculate_reward(distance): Calculates the reward based on the current distance.
        """
        super(CoppeliaEnv, self).__init__()

        self._commstoagent = None  # To be initialized externally after environment creation

        # Extract the parameters for the environment located inside the coresponding params file json.
        # Make params_env accessible for the methods of the class
        self.params_scene = params_scene
        self.params_env = params_env

        # LAT variables
        self.lat_sim = 0
        self.lat_wall = 0
        
        # Process control variables
        self.count=0
        self.time_elapsed=0
        self.total_time_elapsed = 0
        self.n_ep=0
        self.ato = 0
        self.terminated = False
        self.truncated = False
        self.collision_flag = False
        self.max_achieved = False
        self.reward = 0
        self.target_zone = 0
        self.action_dic = {}
        self.tol_lat = 0.3
        self.crash_flag = False
        self.initial_ato=0
        self.reset_flag = False
        self.initial_target_distance = 0

        # For logging LATs during training
        self.csv_lats_path = None  
        self.csv_lats_file = None
        self.csv_lats_writer = None
        self.csv_lats_opened = False
    
    
    # --------------------
    # ----- HELPERS ------
    # --------------------


    def _laser_sort_key(self, name: str) -> int:
        """Sort key for 'laser_obsN' names. Non-num suffix -> +inf to ir al final."""
        m = re.match(r"^laser_obs(\d+)$", name)
        return int(m.group(1)) if m else 10**9  # big number pushes weird names to the end


    def _extract_laser_from_obs_dict(self, obs_dict: Dict[str, Any]) -> List[float]:
        """Return laser readings from a dict observation using params_env["observation_names"].
        Fallback to get keys starting with "laser_obs" from 'obs_dict'.

        Args:
            obs_dict (Dict[str, Any]): The observation dictionary received from the agent system.
        
        Returns:
            list[float]: Laser distances en orden consistente.
        """
        # Prefer order specified in params_env
        names_from_params: Sequence[str] = self.params_env.get("observation_names", []) or []
        if names_from_params:
            laser_names = [n for n in names_from_params if isinstance(n, str) and n.startswith("laser_obs")]
            laser_names = sorted(laser_names, key=self._laser_sort_key)
        else:
            # Fallback: get keys starting with "laser_obs" from obs_dict and sort them
            laser_names = sorted(
                [k for k in obs_dict.keys() if isinstance(k, str) and k.startswith("laser_obs")],
                key=self._laser_sort_key
            )

        # Filtrate the laser names to only those that are in obs_dict, and return their values
        return [float(obs_dict[n]) for n in laser_names if n in obs_dict]
    

    def _compute_adjusted_reward(self, max_reward):
        """
        Compute the adjusted reward based on the time elapsed.
        
        The reward is reduced based on how much time has passed, where the maximum 
        reduction is half of the maximum reward. The function scales the reduction 
        linearly from 0 to max_reward/2 as time_elapsed goes from 0 to max_time.
        
        Args: 
            max_reward (float): The maximum reward.
            
        Returns:
            float: The adjusted reward after applying the time-based discount.
        """
        
        # Calculate the discount based on time_elapsed. It scales between 0 and max_reward/2.
        discount = (self.time_elapsed / self.params_env["max_time"]) * (max_reward / 2)
        
        # Ensure the discount does not exceed half of the maximum reward (max_reward/2).
        discount = min(discount, max_reward / 2)
        
        # Adjust the reward by subtracting the calculated discount.
        adjusted_reward = max_reward - discount
        
        return adjusted_reward


    def compute_reward(self):
        """Compute the reward based on the current observation, action, and info.

        Args: None

        Returns:
            reward (float): The computed reward.
        """
        laser_obs = self._extract_laser_from_obs_dict(self.observation)
        distance = self.observation["distance"]
        p_env = self.params_env
        p_scene = self.params_scene

        # --- Crash detected during the movement by the environment ---
        if self.crash_flag:
            logging.info("Crashed detected during the movement")
            self.collision_flag = True
            self.terminated=True
            self.target_zone = 0
            return p_env["crash_penalty"]

        # --- Collision check ---
        crashed = False
        if p_env["laser_observations"] == 4:
            if (
                laser_obs[0] < p_env["max_crash_dist_critical"]
                or laser_obs[3] < p_env["max_crash_dist_critical"]
                or any(laser_obs[i] < p_env["max_crash_dist"] for i in (1, 2))
            ):
                crashed = True
        else:
            if any(d < p_env["max_crash_dist_critical"] for d in laser_obs):
                crashed = True

        self.collision_flag = crashed
        if crashed:
            logging.info("Crashed")
            self.terminated = True
            self.target_zone = 0
            return p_env["crash_penalty"]
        
        # --- Distance-based rewards ---
        if distance < p_scene["inner_disk_rad"]:
            self.target_zone = 3
            self.terminated = True
            return self._compute_adjusted_reward(p_env["reward_3"])
        elif distance < p_scene["middle_disk_rad"]:
            self.target_zone = 2
            self.terminated = True
            return self._compute_adjusted_reward(p_env["reward_2"])
        elif distance < p_scene["outer_disk_rad"]:
            self.target_zone = 1
            self.terminated = True
            return self._compute_adjusted_reward(p_env["reward_1"])
        elif distance > p_env["max_dist"] or self.time_elapsed > p_env["max_time"]:
            self.terminated = True
            logging.info("Max dist or max time achieved")
            self.max_achieved = True
            self.target_zone = 0
            return p_env["overlimit_penalty"]
        else:
            self.terminated = False
            self.truncated = False
            self.max_achieved = False
            self.collision_flag = False
            self.target_zone = 0
            return 0


    # ---------------------------
    # ----- MAIN FUNCTIONS ------
    # ---------------------------


    def get_last_info(self):
        return self.info


    def step(self, action):
        """
        Execute one time step within the environment.

        Args:
            action (np.array): Action vector.

        Returns:
            observation (np.array): The new state.
            reward (float): Reward obtained.
            terminated (bool): Whether the episode has terminated.
            truncated (bool): Whether the episode was truncated.
            info (dict): Additional information.
        """

        logging.info("STEP Call")

        # Open csv lat file if it's not opened yet
        if self.csv_lats_path is not None:
            if not self.csv_lats_opened:
                self.csv_lats_file = open(self.csv_lats_path, "w", newline="")
                self.csv_lats_writer = csv.writer(self.csv_lats_file)
                self.csv_lats_writer.writerow(["lat_sim", "lat_wall"])  # Header
                self.csv_lats_opened = True


        # Get initial simulation time when the first step occurs. We cannot do this inside the reset, because between the reset
        if self.reset_flag:
            self.reset_flag = False
            self.initial_ato = self.ato

        # Make sure that action is a numpy array of 1D, because when testing it can be 2D
        action = action.flatten()   

        self.action_dic = utils.map_action_vector(self, action)

        # Send action to agent and receive an observation.
        logging.info(f"Send act to agent: { {key: round(value, 3) for key, value in self.action_dic.items()} }.")
        self.lat_sim, self.lat_wall, self.observation, self.crash_flag, self.ato = self._commstoagent.stepSendActGetObs(self.action_dic, timeout = 3000.0)
        logging.info(f"Obs rec STEP: { {key: round(value, 3) for key, value in self.observation.items()} }")
        logging.debug(f"REC: crash flag: {self.crash_flag}, ato: {self.ato}")

        # Save lats in csv file
        if self.csv_lats_path is not None:
            self.csv_lats_writer.writerow([
                float(self.lat_sim),
                float(self.lat_wall),
            ])

        # Update counters
        self.count=self.count+1 
        self.time_elapsed=self.ato-self.initial_ato
        self.truncated = False

        # Calculate reward
        self.reward = self.compute_reward()
        logging.info(f"LAT sim: {round(self.lat_sim,4)}. LAT wall: {round(self.lat_wall,4)}. RW: {round(self.reward,4)}")

        # Update episode
        if self.reward !=0:
            logging.debug(f"Episode {self.n_ep} is finished")
            logging.info(f"Time elapsed (sim time): {round(self.time_elapsed, 3)}")
            self.n_ep=self.n_ep+1
        
        # Observation conversion for consistency
        self.observation = np.array(list(self.observation.values()), dtype=np.float32)

        # Add additional information (optional)
        self.info = {
            "terminated": self.terminated, 
            "truncated": self.truncated, 
            "lat_sim":self.lat_sim,
            "lat_wall":self.lat_wall,
            "actions":self.action_dic
            }

        return self.observation, self.reward, self.terminated, self.truncated, self.info
    

    


    def reset(self, seed=None):
        """
        Reset the environment to an initial state and return the initial observation.

        Args: None

        Returns:
            observation (np.array): The initial state.
            info (dict): Additional information.
        """
        logging.info("RESET Call")

        # Get the initial observation after resetting the environment
        self.observation, self.ato, self.info_obs = self._commstoagent.resetGetObs(timeout = 3000.0)

        # Reset counters and termination flags
        self.terminated = False
        self.truncated = False 
        self.count=0
        self.total_time_elapsed = self.total_time_elapsed + self.time_elapsed
        self.time_elapsed=0
        self.crash_flag = False
        self.reset_flag = True

        logging.info(f"Obs rec RESET: { {key: round(value, 2) for key, value in self.observation.items()} }")
        
        # Convert observation to numpy array
        self.observation = np.array(list(self.observation.values()), dtype=np.float32)

        # Add additional information (optional)
        # self.info_obs = {}

        # SB3 learn method needs a tuple
        return (self.observation, self.info_obs) 
    

# -----------------------------------------------
# -------------- Hardcoded classes --------------
# -----------------------------------------------
        
class BurgerBotEnv(CoppeliaEnv):
    def __init__(self, params_scene, params_env):
        """
        Custom environment for the BurgerBot robot simulation in CoppeliaSim, inherited from CoppeliaEnv class.

        Args:
            params_scene (dict): Dictionary of parameters for configuring the scene.
            params_env (dict): Dictionary of parameters for configuring the environment

        Attributes:
            observation_space (gym.spaces.Box): Observation space of the environment.
                - distance: [0, 5] m.
                - angle: [-pi,pi] rads.
                - time_elapsed: [0, 100] seconds of time consumed by the agent. Optional.
        """
        super(BurgerBotEnv, self).__init__(params_scene, params_env)

        # Define action space
        low = np.asarray(params_env["action_bottom_limits"], dtype=np.float32)
        high = np.asarray(params_env["action_upper_limits"], dtype=np.float32)
        self.action_space = spaces.Box(low=low, high=high, dtype=np.float32)

        # Define observation space
        low = np.asarray(params_env["observation_bottom_limits"], dtype=np.float32)
        high = np.asarray(params_env["observation_upper_limits"], dtype=np.float32)
        self.observation_space = spaces.Box(low=low, high=high, dtype=np.float32)


class TurtleBotEnv(CoppeliaEnv):
    def __init__(self, params_scene, params_env):
        """
        Custom environment for the TurtleBot robot simulation in CoppeliaSim, inherited from CoppeliaEnv class.

        Args:
            params_scene (dict): Dictionary of parameters for configuring the scene.
            params_env (dict): Dictionary of parameters for configuring the environment

        Attributes:
            observation_space (gym.spaces.Box): Observation space of the environment.
                - distance: [0, 5] m.
                - angle: [-pi,pi] rads.
                - laser_obs: 4 floats in the range [0,4] representing the distance in m. to the closest obstacle.
                - time_elapsed: [0, 100] seconds of time consumed by the agent. Optional.
        """
        super(TurtleBotEnv, self).__init__(params_scene, params_env)

        # Define action space
        low = np.asarray(params_env["action_bottom_limits"], dtype=np.float32)
        high = np.asarray(params_env["action_upper_limits"], dtype=np.float32)
        self.action_space = spaces.Box(low=low, high=high, dtype=np.float32)

        # Define observation space
        low = np.asarray(params_env["observation_bottom_limits"], dtype=np.float32)
        high = np.asarray(params_env["observation_upper_limits"], dtype=np.float32)
        self.observation_space = spaces.Box(low=low, high=high, dtype=np.float32)