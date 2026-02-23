"""
Project: Robot Training and Testing RL Algorithms in CoppeliaSim
Author: Adrián Bañuls Arias
Version: 2.0
Date: 2025-03-25
License: GNU General Public License v3.0

Description:
    This script performs path-based evaluation of trained reinforcement learning policies.
    It supports two evaluation modes:
    
    1. Path mode: Samples positions along a recorded path (defined by control points in the scene).
    2. Grid mode: Tests the robot at all valid positions extracted from an occupancy map.
    
    For each sampled robot pose, the script generates orientation variants, places random targets,
    and predicts RL actions to analyze policy behavior across spatial and angular conditions.
    Results are logged to CSV for comprehensive performance analysis and temporal mapping.

Usage:
    uncore_rl test_path --model_name <model_name>
                        [--scene_path <path_to_scene_file>]
                        [--path_alias <path_alias>]
                        [--trials_per_sample <int>]
                        [--n_samples <int>]
                        [--n_extra_poses <int>]
                        [--delta_deg <float>]
                        [--robot_world_ori <float>]
                        [--robot_target_ori <float>]
                        [--map_name <map_file>]
                        [--place_obstacles_flag]
                        [--random_target_flag]
                        [--robot_name <robot_name>]
                        [--dis_parallel_mode]
                        [--no_gui]
                        [--params_file <path_to_config_file>]
                        [--save_scene]
                        [--save_traj]
                        [--obstacles_csv_folder <folder>]
                        [--set KEY=VALUE]
                        [--timestamp <timestamp>]
                        [--verbose <0|1|2|3>]

Features:
    - Dual evaluation modes: recorded path sampling or grid-based coverage testing.
    - Robot orientation augmentation with configurable angular increments.
    - Random target placement for robustness evaluation.
    - Deterministic action prediction for reproducible results.
    - Detailed CSV logging of robot states, actions, and observations.
    - Supports headless execution for automated testing pipelines.
    - Compatible with all Stable-Baselines3 algorithms.
    - Loads configuration from parameter files with CLI overrides via --set.
    - Enables temporal mapping and spatial policy analysis.
"""

import os
import csv
import logging
import stable_baselines3
from tqdm.auto import tqdm

from common import utils
from common.rl_coppelia_manager import RLCoppeliaManager


# ----------------------
# -------- MAIN --------
# ----------------------

def main(args):
    """Drive the path probe and save per-point timestep distributions to CSV."""

    # --- Start/attach to Coppelia/SB3 ---
    rl_copp = RLCoppeliaManager(args)

    # --- Check input map and get robot positions if a map has been provided
    # Ensure that used did not forget to specify the map on purpose
    utils.resolve_map_path_interactive(rl_copp)

    # If a map has been provided, get valid robot positions on it. If not, use n_samples from args.
    if rl_copp.args.map_name:
        map_path = os.path.join(rl_copp.base_path, "custom_maps", rl_copp.args.map_name)
        logging.info(f"Map provided: {map_path}.")
        rl_copp.base_pos_samples = utils.get_positions_on_map(rl_copp, object_type="robot")

        # Valid positions (not augmented yet, that will be done inside Coppelia scene)
        logging.info(f"{len(rl_copp.base_pos_samples)} positions will be sent to the Agent side to be tested.")
    
    # Create environment (skip if it's just agent side)
    if not args.agent_side:
        rl_copp.create_env()
    
    # Start CoppeliaSim (skip if it's just RL side)
    if not args.rl_side:
        rl_copp.start_coppelia_sim("TestPath", path_version=True)

    # --- Testing loop over sampled path positions ---
    if not args.agent_side:
        # Start communication RL - CoppeliaSim
        rl_copp.start_communication()

        # Get paths
        models_path = rl_copp.paths["models"]
        testing_path = rl_copp.paths["testing_metrics"]
        training_metrics_path = rl_copp.paths["training_metrics"]

        # Build whole model name path
        rl_copp.args.model_name = os.path.join(models_path, rl_copp.args.model_name)
        logging.info(f"Model used for the testing {rl_copp.args.model_name}")

        # Assure that the algorithm used for testing a model is the same than the one used for training it
        model_name = os.path.splitext(os.path.basename(rl_copp.args.model_name))[0] # Get the model name from the model file path.
        train_records_csv_name = os.path.join(training_metrics_path,"train_records.csv")    # Name of the train records csv to search the algorithm used
        try:
            rl_copp.params_test["sb3_algorithm"] = utils.get_data_from_training_csv(model_name, train_records_csv_name, "Algorithm")
        except:
            rl_copp.params_test["sb3_algorithm"] = rl_copp.params_train["sb3_algorithm"]
        ModelClass = getattr(stable_baselines3, rl_copp.params_test["sb3_algorithm"])

        # Load model
        model = ModelClass.load(rl_copp.args.model_name, rl_copp.env)

        # Output CSV
        testing_folder = os.path.join(testing_path, f"{model_name}_testing")
        os.makedirs(testing_folder, exist_ok=True)

        _experiment_csv_name, experiment_csv_path= utils.get_output_csv(model_name, testing_folder, f"path_data_RW{args.robot_world_ori}_RT{args.robot_target_ori}")
    
        # CSV header
        observation_names = rl_copp.env.envs[0].unwrapped.params_env.get("observation_names", [])
        id_headers = ["Position idx"] + ["Scenario idx"]  + ["Trial idx"] 
        position_info_headers = ["Pos X"] + ["Pos Y"] 
        headers = id_headers + position_info_headers + ["Timestep"] + observation_names
        
        if rl_copp.base_pos_samples ==[]:
            n_samples = args.n_samples
        else:
            n_samples = len(rl_copp.base_pos_samples)

        logging.info(
            f" ----- Testing the robot {rl_copp.robot_name} with its model {model_name} -----\n"
            f"          --- Path will be sampled in: {n_samples}. ---\n"
            f"          --- Each sample will be tested with {args.n_extra_poses*2+1} scenarios. ---\n"
            f"          --- Each scenario will be repeated {args.trials_per_sample} times. ---\n")

        with open(experiment_csv_path, mode="w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(headers)

            # Iterate sampled points trials_per_sample times each point
            total_robot_poses = n_samples*(args.n_extra_poses*2+1)

            for position_idx in tqdm(range(n_samples), desc="Testing positions", unit="position"):

                for scenario_idx in range(args.n_extra_poses*2+1):

                    # For each sampled point of the path, test different target scenarios
                    for trial_idx in range(args.trials_per_sample):
                        
                        observation, info_obs = rl_copp.env.envs[0].reset()
                        logging.info(f"Position idx: {position_idx}. Scenario idx: {scenario_idx}. Trial idx: {trial_idx}")

                        # Predict action based on last observation
                        action, _states = model.predict(observation, deterministic=True)

                        # Send a step to the agent jsut to confirm that a new action has been predicted successfully
                        observation, _, terminated, truncated, info = rl_copp.env.envs[0].step(action)
                        
                        # Save the predicted timestep
                        ts_value = float(info["actions"]["timestep"])

                        obs_values = [round(float(v), 4) for v in observation.tolist()]

                        # Format obtained info
                        logging.info(f"Extra info in the observation: {info_obs}")
                        info_obs = [info_obs["posX"], info_obs["posY"]]

                        row = [position_idx] + [scenario_idx] + [trial_idx] + info_obs + [ts_value] + obs_values
                        writer.writerow(row)

            logging.info(f"[test_path] Saved results to: {experiment_csv_path}")

        rl_copp.env.envs[0].unwrapped._commstoagent.stepExpFinished()
        logging.info("Testing path has finished")
        
        # Optionally close the simulation here
        # rl_copp.stop_coppelia_sim()


if __name__ == "__main__":
    main()
