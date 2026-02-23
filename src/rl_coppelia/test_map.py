"""
Project: Robot Training and Testing RL Algorithms in CoppeliaSim
Author: Adrián Bañuls Arias
Version: 2.0
Date: 2025-03-25
License: GNU General Public License v3.0

Description:
    This script performs grid-based evaluation of trained reinforcement learning policies
    with a fixed target position. The robot is teleported to each valid grid cell and tested
    with multiple orientations and target-position noise samples.
    
    The target position is fixed throughout the experiment (selected by user at start).
    The robot is teleported to each valid position in a grid extracted from the map.
    For each robot position, the base orientation is computed to face the target, or face away.
    Additional orientations are generated using n_extra_poses and delta_deg parameters.
    For each augmented orientation, the target position is perturbed with Gaussian noise
    target_trials times.
    
    The result is a CSV dataset containing robot positions, target positions, orientations,
    predicted timesteps, and observation values for comprehensive policy analysis.

Usage:
    uncore_rl test_map --model_name <model_name>
                       [--map_name <map_file>]
                       [--n_extra_poses <int>]
                       [--delta_deg <float>]
                       [--target_noise_std <float>]
                       [--target_trials <int>]
                       [--trials_per_sample <int>]
                       [--face_away]
                       [--scene_path <path_to_scene_file>]
                       [--robot_name <robot_name>]
                       [--dis_parallel_mode]
                       [--no_gui]
                       [--params_file <path_to_config_file>]
                       [--set KEY=VALUE]
                       [--timestamp <timestamp>]
                       [--verbose <0|1|2|3>]

Features:
    - Grid-based coverage testing with fixed target position selected interactively.
    - Robot orientation augmentation with configurable angular increments.
    - Target position noise sampling for robustness evaluation.
    - Supports headless execution for automated testing pipelines.
    - Integrates with CoppeliaSim for realistic simulation environment.
    - Detailed CSV logging of robot states, actions, and observations.
    - Compatible with all Stable-Baselines3 algorithms.
    - Loads configuration from parameter files with CLI overrides via --set.
    - Multiple verbosity levels for debugging and progress tracking.
"""

import os
import csv
import json
import math
import logging
import numpy as np
import stable_baselines3
from tqdm.auto import tqdm
from typing import Tuple, List
import matplotlib.pyplot as plt
from PIL import Image
import yaml

from common import utils
from common.rl_coppelia_manager import RLCoppeliaManager


# ----------------------
# ------- HELPERS ------
# ----------------------

def select_target_on_map(
    map_png_path: str,
    m_per_px: float = 0.02013,
    origin_xy: Tuple[float, float] = (-10.5, -6.0),
    origin_is_lower_left: bool = False,
    title: str = "Click to select the TARGET position. Press Enter to confirm."
) -> Tuple[float, float]:
    """
    Display the map and let the user click to select a single target position,
    or allow manual input via terminal.
    
    Args:
        map_png_path: Path to the map PNG image.
        m_per_px: Meters per pixel in the map image.
        origin_xy: Origin coordinates (x, y) of the map in world frame.
        origin_is_lower_left: Whether the origin is at the lower left of the image.
        title: Window title with instructions.
    
    Returns:
        (x, y) coordinates of the selected target in world frame (meters).
    """
    # Check if user wants to input target coordinates manually
    print("\nTarget selection options:")
    print("- Type 'M' and press Enter to open the map and click to select target position")
    print("- Type 'T' and press Enter to input target coordinates manually")
    
    user_choice = input("Your choice: ").strip().upper()
    
    if user_choice == 'T':
        # Manual input mode
        print("\nManual target coordinate input:")
        while True:
            try:
                target_input = input("Target coordinates (x y, e.g., -5.2 3.1): ").strip()
                x_str, y_str = target_input.split()
                target_x, target_y = float(x_str), float(y_str)
                
                # Show the coordinates on the map for visual confirmation
                print(f"Target coordinates entered: ({target_x:.3f}, {target_y:.3f})")
                print("Opening map to display the selected position...")
                
                # Load and display map with the entered position
                img = Image.open(map_png_path).convert("RGB")
                w_px, h_px = img.size
                x0, y0 = origin_xy
                x1 = x0 + w_px * m_per_px
                y1 = y0 + h_px * m_per_px
                origin_kw = "lower" if origin_is_lower_left else "upper"
                
                fig, ax = plt.subplots(figsize=(10, 8))
                ax.imshow(img, extent=[x0, x1, y0, y1], origin=origin_kw)
                ax.set_xlabel("X [m]")
                ax.set_ylabel("Y [m]")
                ax.scatter([target_x], [target_y], s=200, c='red', marker='X', 
                          zorder=10, edgecolors='white', linewidths=2)
                ax.set_title(f"Target position: ({target_x:.3f}, {target_y:.3f}). Press Enter to confirm or Escape to re-enter.")
                
                # State for key handling
                confirmed = [None]  # None=pending, True=confirmed, False=re-enter
                
                def on_key_manual(event):
                    if event.key == 'enter':
                        confirmed[0] = True
                        plt.close(fig)
                    elif event.key == 'escape':
                        confirmed[0] = False
                        plt.close(fig)
                
                fig.canvas.mpl_connect('key_press_event', on_key_manual)
                plt.show()
                
                if confirmed[0] is True:
                    return (target_x, target_y)
                else:
                    print("Position not confirmed. Please enter the coordinates again:")
                    
            except ValueError:
                print("Invalid input. Please enter two numeric values separated by space.")
    # Graphical selection mode (default)
    print("Opening map window for graphical target selection...")
    
    # Load and display map
    img = Image.open(map_png_path).convert("RGB")
    w_px, h_px = img.size
    x0, y0 = origin_xy
    x1 = x0 + w_px * m_per_px
    y1 = y0 + h_px * m_per_px
    origin_kw = "lower" if origin_is_lower_left else "upper"

    fig, ax = plt.subplots(figsize=(10, 8))
    ax.imshow(img, extent=[x0, x1, y0, y1], origin=origin_kw)
    ax.set_xlabel("X [m]")
    ax.set_ylabel("Y [m]")
    ax.set_title(title)

    # State
    selected_point = [None]  # Use list to allow modification in nested function
    marker = None

    def on_click(event):
        nonlocal marker
        if event.inaxes != ax or event.button != 1:
            return
        
        x, y = event.xdata, event.ydata
        selected_point[0] = (x, y)
        
        # Update marker visualization
        if marker is not None:
            marker.remove()
        marker = ax.scatter([x], [y], s=200, c='red', marker='X', zorder=10, edgecolors='white', linewidths=2)
        ax.set_title(f"Target selected at ({x:.2f}, {y:.2f}). Press Enter to confirm or click again to change.")
        fig.canvas.draw()

    def on_key(event):
        if event.key == 'enter' and selected_point[0] is not None:
            plt.close(fig)
        elif event.key == 'escape':
            selected_point[0] = None
            plt.close(fig)

    fig.canvas.mpl_connect('button_press_event', on_click)
    fig.canvas.mpl_connect('key_press_event', on_key)

    logging.info("--- Map window opened: click to select target position ---")
    plt.show()  # Blocks until window is closed

    if selected_point[0] is None:
        raise ValueError("No target position was selected. Aborting.")

    return selected_point[0]


def compute_orientation_to_target(
    robot_x: float,
    robot_y: float,
    target_x: float,
    target_y: float,
    face_away: bool = False
) -> float:
    """
    Compute the yaw angle (in radians) for the robot to face the target.
    
    Args:
        robot_x, robot_y: Robot position in world frame.
        target_x, target_y: Target position in world frame.
        face_away: If True, robot faces away from target (opposite direction).
    
    Returns:
        Yaw angle in radians, normalized to [-pi, pi].
    """
    dx = target_x - robot_x
    dy = target_y - robot_y
    yaw = math.atan2(dy, dx)
    
    # If face_away, rotate 180 degrees
    if face_away:
        yaw = yaw + math.pi
        # Normalize to [-pi, pi]
        if yaw > math.pi:
            yaw -= 2 * math.pi
    
    return yaw


def augment_orientations(
    base_yaw: float,
    n_extra_poses: int,
    delta_deg: float
) -> List[float]:
    """
    Generate a list of orientations: base_yaw, then +k*delta, then -k*delta for k=1..n_extra.
    
    Args:
        base_yaw: Base orientation in radians.
        n_extra_poses: Number of extra orientations on each side.
        delta_deg: Angular increment in degrees.
    
    Returns:
        List of yaw angles in radians (length = 1 + 2*n_extra_poses).
    """
    delta_rad = math.radians(delta_deg)
    orientations = [base_yaw]
    for k in range(1, n_extra_poses + 1):
        orientations.append(base_yaw + k * delta_rad)
        orientations.append(base_yaw - k * delta_rad)
    # Normalize all to [-pi, pi]
    orientations = [math.atan2(math.sin(y), math.cos(y)) for y in orientations]
    return orientations


def generate_noisy_targets(
    target_x: float,
    target_y: float,
    noise_std: float,
    n_samples: int,
    rng: np.random.Generator = None
) -> List[Tuple[float, float]]:
    """
    Generate noisy target positions around a fixed target using Gaussian noise.
    
    Args:
        target_x, target_y: Base target position.
        noise_std: Standard deviation of Gaussian noise in meters.
        n_samples: Number of noisy samples to generate.
        rng: Numpy random generator (optional).
    
    Returns:
        List of (x, y) noisy target positions.
    """
    if rng is None:
        rng = np.random.default_rng()
    
    noisy_targets = []
    for _ in range(n_samples):
        nx = target_x + rng.normal(0, noise_std)
        ny = target_y + rng.normal(0, noise_std)
        noisy_targets.append((nx, ny))
    
    return noisy_targets


def build_all_test_cases(
    robot_positions: List[Tuple[float, float]],
    fixed_target: Tuple[float, float],
    n_extra_poses: int,
    delta_deg: float,
    target_noise_std: float,
    target_trials: int,
    rng: np.random.Generator = None,
    face_away: bool = False
) -> List[dict]:
    """
    Pre-build all test cases with robot poses and noisy target positions.
    
    Each test case contains:
        - position_idx: Index of robot grid position
        - orientation_idx: Index of augmented orientation  
        - trial_idx: Index of target noise sample
        - robot_pose: (x, y, yaw)
        - target_pos: (tx, ty) with noise applied
    
    Returns:
        List of test case dictionaries.
    """
    if rng is None:
        rng = np.random.default_rng(seed=42)
    
    test_cases = []
    fixed_tx, fixed_ty = fixed_target
    
    for pos_idx, robot_pos in enumerate(robot_positions):
        robot_x, robot_y = robot_pos[0], robot_pos[1]
        
        # Compute base orientation facing target (or away from target if face_away=True)
        base_yaw = compute_orientation_to_target(robot_x, robot_y, fixed_tx, fixed_ty, face_away)
        
        # Generate augmented orientations
        orientations = augment_orientations(base_yaw, n_extra_poses, delta_deg)
        
        for ori_idx, robot_yaw in enumerate(orientations):
            # Generate noisy targets for this orientation
            noisy_targets = generate_noisy_targets(
                fixed_tx, fixed_ty, target_noise_std, target_trials, rng
            )
            
            for trial_idx, (noisy_tx, noisy_ty) in enumerate(noisy_targets):
                test_cases.append({
                    "position_idx": pos_idx,
                    "orientation_idx": ori_idx,
                    "trial_idx": trial_idx,
                    "robot_pose": (robot_x, robot_y, robot_yaw),
                    "target_pos": (noisy_tx, noisy_ty),
                })
    
    return test_cases

def extract_map_parameters(
    map_png_path: str
) -> Tuple[float, Tuple[float, float]]:
    """ 
    Extract map parameters (m_per_px and origin) from the corresponding YAML file.
    Parameter file is in yaml format, and it's located under the same directory as the provided map
    The YAML file has the same base name as the PNG/PGM but with .yaml extension
    Args:
        map_png_path: Path to the map PNG image.
    Returns:
        m_per_px: Meters per pixel.
        origin_xy: Origin coordinates (x, y) of the map in world frame.
    """
    map_dir = os.path.dirname(map_png_path)
    map_basename = os.path.splitext(os.path.basename(map_png_path))[0]
    yaml_path = os.path.join(map_dir, f"{map_basename}.yaml")
    
    if os.path.exists(yaml_path):
        
        with open(yaml_path, 'r') as yf:
            map_params = yaml.safe_load(yf)
        m_per_px = map_params.get("resolution", 0.02013)
        origin_list = map_params.get("origin", [-10.5, -6.0, 0])
        origin_xy = (origin_list[0], origin_list[1])
        logging.info(f"Map parameters loaded from {yaml_path}: resolution={m_per_px}, origin={origin_xy}")
    else:
        logging.warning(f"YAML file not found at {yaml_path}.")
        print(f"\nMap parameters file '{yaml_path}' not found.")
        print("Please enter the map parameters manually:")
        
        # Ask for resolution (meters per pixel)
        while True:
            try:
                m_per_px_input = input("Resolution (meters per pixel, e.g., 0.05): ").strip()
                m_per_px = float(m_per_px_input)
                if m_per_px > 0:
                    break
                else:
                    print("Resolution must be positive. Try again.")
            except ValueError:
                print("Invalid input. Please enter a numeric value.")
        
        # Ask for origin coordinates
        while True:
            try:
                origin_input = input("Origin coordinates (x y, e.g., -10.5 -6.0): ").strip()
                x_str, y_str = origin_input.split()
                origin_xy = (float(x_str), float(y_str))
                break
            except ValueError:
                print("Invalid input. Please enter two numeric values separated by space.")
        
        logging.info(f"Manual map parameters entered: resolution={m_per_px}, origin={origin_xy}")
    return m_per_px, origin_xy


# -------------------------
# --------- MAIN ---------- 
# -------------------------

def main(args):
    """Query the RL policy from each grid position with fixed target and save results to CSV."""

    # --- Start/attach to Coppelia/SB3 ---
    rl_copp = RLCoppeliaManager(args)

    # --- Check input map and get robot positions if a map has been provided
    # Ensure that user did not forget to specify the map on purpose
    utils.resolve_map_path_interactive(rl_copp, map_is_mandatory=True)

    # Build map path
    map_path = os.path.join(rl_copp.base_path, "custom_maps", rl_copp.args.map_name)

    # --- Extract m_per_px and origin from map parameters if available ---
    m_per_px, origin_xy = extract_map_parameters(map_path)

    # --- Select target position interactively ---
    logging.info("Please select the fixed target position on the map (graphically or via terminal input)...")
    fixed_target_x, fixed_target_y = select_target_on_map(
        map_png_path=map_path,
        m_per_px=m_per_px,
        origin_xy=origin_xy,
        origin_is_lower_left=False,
        title="Click to select the fixed target position. Press Enter to confirm."
    )
    logging.info(f"Fixed target selected at: ({fixed_target_x:.3f}, {fixed_target_y:.3f})")

    # --- Get robot positions from map ---
    logging.info(f"Map provided: {map_path}.")
    rl_copp.base_pos_samples = utils.get_positions_on_map(
        rl_copp, 
        object_type="robot",
        m_per_px=m_per_px,
        origin_xy=origin_xy,
        origin_is_lower_left=False,
        obstacle_threshold = 50,
        grid_step_m = 0.25,
        )
    n_positions = len(rl_copp.base_pos_samples)
    logging.info(f"{n_positions} robot positions will be tested.")

    # --- Pre-build all test cases ---
    rng = np.random.default_rng(seed=42)
    n_repetitions = getattr(args, 'trials_per_sample', 1)
    face_away = getattr(args, 'face_away', False)
    
    base_test_cases = build_all_test_cases(
        robot_positions=rl_copp.base_pos_samples,
        fixed_target=(fixed_target_x, fixed_target_y),
        n_extra_poses=args.n_extra_poses,
        delta_deg=args.delta_deg,
        target_noise_std=args.target_noise_std,
        target_trials=args.target_trials,
        rng=rng,
        face_away=face_away
    )
    
    # Expand test cases to include repetitions
    # Each base test case is repeated n_repetitions times
    test_cases = []
    for base_case in base_test_cases:
        for rep_idx in range(n_repetitions):
            expanded_case = base_case.copy()
            expanded_case["repetition_idx"] = rep_idx
            test_cases.append(expanded_case)
    
    # Store test cases in rl_copp for agent side to use
    rl_copp.test_cases = test_cases
    rl_copp.fixed_target_pos = (fixed_target_x, fixed_target_y)
    rl_copp.test_map_mode = True

    # Create environment (skip if it's just agent side)
    if not args.agent_side:
        rl_copp.create_env()

    # Start CoppeliaSim (skip if it's just RL side)
    if not args.rl_side:
        rl_copp.start_coppelia_sim("Test_Map", path_version=True)

    # --- Testing loop ---
    if not args.agent_side:
        # Start communication RL - CoppeliaSim
        rl_copp.start_communication()

        # Get paths
        models_path = rl_copp.paths["models"]
        testing_path = rl_copp.paths["testing_metrics"]
        training_metrics_path = rl_copp.paths["training_metrics"]

        # Build whole model name path
        rl_copp.args.model_name = os.path.join(models_path, rl_copp.args.model_name)
        logging.info(f"Model used for the testing: {rl_copp.args.model_name}")

        # Get algorithm used for training
        model_name = os.path.splitext(os.path.basename(rl_copp.args.model_name))[0]
        train_records_csv_name = os.path.join(training_metrics_path, "train_records.csv")
        try:
            rl_copp.params_test["sb3_algorithm"] = utils.get_data_from_training_csv(
                model_name, train_records_csv_name, "Algorithm"
            )
        except:
            rl_copp.params_test["sb3_algorithm"] = rl_copp.params_train["sb3_algorithm"]
        ModelClass = getattr(stable_baselines3, rl_copp.params_test["sb3_algorithm"])

        # Load model
        model = ModelClass.load(rl_copp.args.model_name, rl_copp.env)

        # Output CSV
        testing_folder = os.path.join(testing_path, f"{model_name}_testing")
        os.makedirs(testing_folder, exist_ok=True)

        # Experiment csv name and path       
        _experiment_csv_name, experiment_csv_path = utils.get_output_csv(
            model_name, 
            testing_folder, 
            f"fixedTarget_path_data"
        )

        # Save experiment configuration to JSON
        config_folder = os.path.join(testing_folder, "test_path_configs")
        os.makedirs(config_folder, exist_ok=True)
        
        # JSON config file has the same base name as the CSV
        config_filename = os.path.splitext(os.path.basename(experiment_csv_path))[0] + ".json"
        config_path = os.path.join(config_folder, config_filename)
        
        experiment_config = {
            "experiment_type": "test_map",
            "model_name": model_name,
            "map_name": rl_copp.args.map_name,
            "fixed_target": {
                "x": fixed_target_x,
                "y": fixed_target_y
            },
            "face_away": face_away,
            "n_positions": n_positions,
            "n_extra_poses": args.n_extra_poses,
            "delta_deg": args.delta_deg,
            "target_noise_std": args.target_noise_std,
            "target_trials": args.target_trials,
            "trials_per_sample": args.trials_per_sample,
            "total_iterations": len(test_cases),
            "map_parameters": {
                "m_per_px": m_per_px,
                "origin_xy": list(origin_xy)
            }
        }
        
        with open(config_path, 'w') as config_file:
            json.dump(experiment_config, config_file, indent=4)
        logging.info(f"Experiment config saved to: {config_path}")

        # CSV header
        observation_names = rl_copp.env.envs[0].unwrapped.params_env.get("observation_names", [])
        id_headers = ["Position idx", "Orientation idx", "Target trial idx", "Repetition idx"]
        robot_headers = ["Robot Pos X", "Robot Pos Y", "Robot Yaw"]
        target_headers = ["Target Pos X", "Target Pos Y"]
        action_headers = ["Timestep", "Angular Vel"]  
        headers = id_headers + robot_headers + target_headers + action_headers + observation_names

        n_orientations = 1 + 2 * args.n_extra_poses
        n_target_trials = args.target_trials
        n_repetitions = getattr(args, 'trials_per_sample', 1)
        total_iterations = len(test_cases)  # test_cases already includes repetitions
        face_away_str = "AWAY from target" if face_away else "TOWARDS target"

        logging.info(
            f" ----- Testing robot {rl_copp.robot_name} with model {model_name} -----\n"
            f"          --- Fixed target at: ({fixed_target_x:.2f}, {fixed_target_y:.2f}) ---\n"
            f"          --- Robot base orientation: {face_away_str} ---\n"
            f"          --- Robot grid positions: {n_positions} ---\n"
            f"          --- Orientations per position: {n_orientations} (base + {args.n_extra_poses}*2) ---\n"
            f"          --- Target noise trials per orientation: {n_target_trials} ---\n"
            f"          --- Repetitions per sample (trials_per_sample): {n_repetitions} ---\n"
            f"          --- Target noise std: {args.target_noise_std} m ---\n"
            f"          --- Total iterations: {total_iterations} ---\n"
        )

        with open(experiment_csv_path, mode="w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(headers)

            pbar = tqdm(total=total_iterations, desc="Testing grid positions", unit="iter")

            for case_idx, test_case in enumerate(test_cases):
                pos_idx = test_case["position_idx"]
                ori_idx = test_case["orientation_idx"]
                trial_idx = test_case["trial_idx"]
                rep_idx = test_case.get("repetition_idx", 0)
                robot_x, robot_y, robot_yaw = test_case["robot_pose"]
                noisy_tx, noisy_ty = test_case["target_pos"]

                # Reset environment (teleports robot and places target)
                # The agent side manages the test case index internally
                observation, info_obs = rl_copp.env.envs[0].reset()

                logging.debug(
                    f"Case {case_idx}: Pos={pos_idx}, Ori={ori_idx}, Trial={trial_idx}, Rep={rep_idx} | "
                    f"Robot: ({robot_x:.2f}, {robot_y:.2f}, {math.degrees(robot_yaw):.1f}°) | "
                    f"Target: ({noisy_tx:.2f}, {noisy_ty:.2f})"
                )

                # Predict action
                action, _states = model.predict(observation, deterministic=True)

                # Step to confirm action
                observation, _, terminated, truncated, info = rl_copp.env.envs[0].step(action)

                # Extract timestep and angular velocity from action info
                ts_value = float(info["actions"].get("timestep", 0.0))
                av_value = float(info["actions"].get("angular", 0.0))

                # Format observation values
                obs_values = [round(float(v), 4) for v in observation.tolist()]

                # Write row
                row = (
                    [pos_idx, ori_idx, trial_idx, rep_idx] +
                    [round(robot_x, 4), round(robot_y, 4), round(robot_yaw, 4)] +
                    [round(noisy_tx, 4), round(noisy_ty, 4)] +
                    [ts_value, av_value] +
                    obs_values
                )
                writer.writerow(row)

                pbar.update(1)

            pbar.close()
            logging.info(f"[test_path_v2] Results saved to: {experiment_csv_path}")

        # Signal experiment finished
        rl_copp.env.envs[0].unwrapped._commstoagent.stepExpFinished()
        logging.info("Testing (test_path_v2) has finished.")


if __name__ == "__main__":
    main()
