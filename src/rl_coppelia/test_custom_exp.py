"""
Project: Robot Training and Testing RL Algorithms in CoppeliaSim
Author: Adrián Bañuls Arias
Version: 2.0
Date: 2026-02-05
License: GNU General Public License v3.0

Description:
    Custom experiment evaluator for trained RL policies.

    This module allows the user to define and run preconfigured evaluation
    scenarios (experiments) on one or more maps, collecting per-episode
    metrics and generating a final summary.

    Three canonical experiment types are provided out-of-the-box:

      • RL_1 - Short-range: random target within a user-defined radius
             (default 2 m) from the robot.  General policy validation.
      • RL_2 - Long-range: random target between a minimum and maximum
             distance (default 2-6 m) from the robot.
      • RL_3 - Near-obstacle: targets placed close to obstacles.
             Evaluates agent confidence in high-risk areas.

    The user interacts with the map to define:
      - Allowed zones (polygon) for robot and/or target placement.
      - Forbidden zones (e.g., behind long walls) that could lead to
        the "lost robot" problem.
      - Near-obstacle zones (for RL_3 specifically).

    Each experiment is run for N episodes (default 500) on each selected
    map, and the results are saved in CSV files under the robot's
    testing_metrics folder.

Usage:
    uncore_rl test_custom_exp --model_name <model_name>
                              --robot_name <robot_name>
                              [--experiments RL_1 RL_2 RL_3]
                              [--maps map1.pgm map2.pgm map3.pgm]
                              [--episodes 500]
                              [--params_file <path>]
                              [--no_gui] [--verbose 0]
                              [other standard flags]
"""

import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import csv
import json
import logging
import time
import numpy as np
import stable_baselines3
import math
from common import utils
from common.rl_coppelia_manager import RLCoppeliaManager
from tqdm.auto import tqdm




# --------------------------------
# ----- Map → Scene mapping ------
# --------------------------------
#  Each .pgm map file is paired with a CoppeliaSim .ttt scene.
#  Add new entries here when new maps / scenes are created.

MAP_TO_SCENE = {
    "ts1.png":       "burgerBot_real_scene1.ttt",
    "ts2.pgm":     "burgerBot_real_scene2.ttt",
    "ts3.pgm":   "burgerBot_real_scene3.ttt",
}

# ------------------------------
# ---------- HELPERS -----------
# ------------------------------


def _resolve_scene_for_map(map_name: str) -> str:
    """Return the scene filename associated with 'map_namE'.

    If no mapping is found the user is asked to type the scene name
    manually so the program never crashes silently.
    """
    scene = MAP_TO_SCENE.get(map_name)
    if scene:
        return scene

    # Also try without extension variants (.pgm / .png / .jpg)
    base = os.path.splitext(map_name)[0]
    for key, val in MAP_TO_SCENE.items():
        if os.path.splitext(key)[0] == base:
            return val

    # Interactive fallback
    print(f"\n[WARNING] No scene mapping found for map '{map_name}'.")
    scene = input(">> Please enter the scene file name (e.g. burgerBot_real_scene1.ttt): ").strip()
    if not scene:
        raise ValueError(f"No scene provided for map '{map_name}'.")
    return scene if scene.endswith(".ttt") else f"{scene}.ttt"


EXPERIMENT_CATALOG = {
    "RL_1": {
        "description": (
            "Short-range validation: random target within {max_dist} m "
            "from the robot, excluding forbidden zones."
        ),
        "defaults": {
            "min_target_dist": 0.3,   # minimum distance (m) robot-target
            "max_target_dist": 2.0,   # maximum distance (m) robot-target
            "random_robot_pos": True, # randomise robot start position
            "use_forbidden_zones": True,  # let user draw forbidden zones on map
            "near_obstacle_mode": False,
        },
    },
    "RL_2": {
        "description": (
            "Long-range evaluation: random target between {min_dist} and "
            "{max_dist} m from the robot, excluding forbidden zones."
        ),
        "defaults": {
            "min_target_dist": 2.0,
            "max_target_dist": 5.0,
            "random_robot_pos": True,
            "use_forbidden_zones": True,
            "near_obstacle_mode": False,
        },
    },
    "RL_3": {
        "description": (
            "Manual target placement: the user clicks specific target "
            "positions directly on the map.  Robot placed via mask; "
            "distance constraints still apply."
        ),
        "defaults": {
            "min_target_dist": 0.3,
            "max_target_dist": 3.0,
            "random_robot_pos": True,
            "use_forbidden_zones": False,
            "near_obstacle_mode": False,
            "manual_target_selection": True,
        },
    },
}


def _select_maps_interactive(maps_dir: str, preselected: list) -> list:
    """Return a list of map file names, asking the user if needed.

    If 'preselected' already contains entries they are validated and
    returned directly.  Otherwise we list the contents of 'maps_dir'
    and let the user pick one or more.
    """
    valid_ext = (".png", ".pgm", ".jpg", ".jpeg")

    if preselected:
        # Validate that every file exists
        validated = []
        for m in preselected:
            full = os.path.join(maps_dir, m) if not os.path.isabs(m) else m
            if os.path.isfile(full):
                validated.append(os.path.basename(full))
            else:
                logging.warning(f"Map file not found, skipping: {full}")
        if validated:
            return validated
        logging.warning("None of the provided maps were found.  Falling back to interactive selection.")

    # Interactive selection — allow multiple picks
    available = sorted(
        f for f in os.listdir(maps_dir)
        if os.path.isfile(os.path.join(maps_dir, f)) and f.lower().endswith(valid_ext)
    )
    if not available:
        raise FileNotFoundError(f"No map files ({valid_ext}) found in {maps_dir}")

    print("\n[test_custom_exp] Available maps:")
    for i, name in enumerate(available):
        print(f"  [{i}] {name}")

    chosen = []
    while True:
        sel = input(
            "\n>> Enter map number to add (or 'done' to finish, 'all' for every map): "
        ).strip().lower()
        if sel == "done":
            break
        if sel == "all":
            chosen = list(available)
            break
        if sel.isdigit() and 0 <= int(sel) < len(available):
            name = available[int(sel)]
            if name not in chosen:
                chosen.append(name)
                print(f"   ✓ Added '{name}'  (selected so far: {len(chosen)})")
            else:
                print(f"   '{name}' already selected.")
        else:
            print("   Invalid input.")

    if not chosen:
        raise ValueError("[test_custom_exp] No maps selected — aborting.")
    return chosen


def _configure_experiment_interactive(exp_name: str, defaults: dict) -> dict:
    """Let the user review / override the default parameters of an experiment.

    Returns the final parameter dictionary for the experiment.
    """
    print(f"\n{'='*60}")
    print(f"  Configuring experiment: {exp_name}")
    print(f"{'='*60}")
    desc = EXPERIMENT_CATALOG[exp_name]["description"]
    print(f"  {desc}\n")
    print("  Current parameters:")
    for k, v in defaults.items():
        print(f"    {k} = {v}")

    modify = input("\n>> Do you want to modify any parameter? [y/N]: ").strip().lower()
    if modify != "y":
        return dict(defaults)

    cfg = dict(defaults)
    for key, val in defaults.items():
        raw = input(f"   {key} [{val}]: ").strip()
        if raw == "":
            continue
        # Cast to the same type as the default
        if isinstance(val, bool):
            cfg[key] = raw.lower() in ("true", "1", "yes", "y")
        elif isinstance(val, float):
            cfg[key] = float(raw)
        elif isinstance(val, int):
            cfg[key] = int(raw)
        else:
            cfg[key] = raw
    return cfg


def _draw_forbidden_zones(map_path, m_per_px, origin_xy, origin_is_lower_left):
    """Let the user draw one or more forbidden polygons on the map.

    Returns a list of np.ndarray polygons (each shape (N,2) in world metres).
    """
    forbidden = []
    print("\n[Forbidden zones] You can draw polygons on the map to mark areas")
    print("where the TARGET should NEVER be placed (e.g., behind long walls).")

    while True:
        draw = input(">> Draw a forbidden zone? [y/N]: ").strip().lower()
        if draw != "y":
            break
        poly = utils.interactive_polygon_on_map_live(
            map_path,
            m_per_px=m_per_px,
            origin_xy=origin_xy,
            origin_is_lower_left=origin_is_lower_left,
            title="Draw FORBIDDEN zone (Click=vertex, Enter=finish, Backspace=undo)"
        )
        if poly is not None and len(poly) >= 3:
            forbidden.append(poly)
            print(f"Forbidden zone added ({len(poly)} vertices).  Total: {len(forbidden)}")
        else:
            print("Cancelled / not enough vertices.")
    return forbidden


def _draw_near_obstacle_zone(map_path, m_per_px, origin_xy, origin_is_lower_left):
    """Draw a single polygon identifying the near-obstacle region for RL_3.

    Returns np.ndarray (N,2) or None.
    """
    print("\n[Near-obstacle zone] Draw the region where targets should be")
    print("placed close to obstacles (RL_3 scenario).")
    poly = utils.interactive_polygon_on_map_live(
        map_path,
        m_per_px=m_per_px,
        origin_xy=origin_xy,
        origin_is_lower_left=origin_is_lower_left,
        title="Draw NEAR-OBSTACLE target zone (Click=vertex, Enter=finish)"
    )
    if poly is not None and len(poly) >= 3:
        print(f"Near-obstacle zone set ({len(poly)} vertices).")
        return poly
    print("No valid zone drawn — near-obstacle filtering disabled for this map.")
    return None


def _get_manual_targets_path(maps_dir: str, map_name: str, exp_name: str) -> str:
    """Return the path where manual target positions are saved for a given map and experiment."""
    map_base = os.path.splitext(map_name)[0]
    return os.path.join(maps_dir, f"manual_targets_{exp_name}_{map_base}.npy")


def _save_manual_targets(filepath: str, targets: np.ndarray) -> None:
    """Save manual target positions to a .npy file."""
    np.save(filepath, targets)
    print(f"   ✓ Manual targets saved to '{os.path.basename(filepath)}'")
    logging.info(f"Manual targets saved → {filepath}")


def _load_manual_targets(filepath: str) -> np.ndarray:
    """Load manual target positions from a .npy file."""
    targets = np.load(filepath)
    print(f"   ✓ Loaded {len(targets)} manual target(s) from '{os.path.basename(filepath)}'")
    logging.info(f"Manual targets loaded ← {filepath}")
    return targets


def _visualize_targets_on_map(map_path, m_per_px, origin_xy,
                              origin_is_lower_left, targets, occ_mask=None):
    """Show the map with the loaded target positions and wait for the user to press Enter.

    Parameters
    ----------
    map_path : str
        Path to the map image file.
    m_per_px : float
        Metres per pixel.
    origin_xy : tuple (x0, y0)
        World-coordinate origin of the map image.
    origin_is_lower_left : bool
        Whether the image origin is at the lower-left corner.
    targets : np.ndarray (N, 2)
        Target positions in world coordinates.
    occ_mask : np.ndarray or None
        Optional occupancy mask overlay.
    """
    from PIL import Image
    import matplotlib.pyplot as plt

    img = Image.open(map_path).convert("RGB")
    w_px, h_px = img.size
    x0, y0 = origin_xy
    x1 = x0 + w_px * m_per_px
    y1 = y0 + h_px * m_per_px
    origin_kw = "lower" if origin_is_lower_left else "upper"

    fig, ax = plt.subplots(figsize=(10, 8))
    ax.imshow(img, extent=[x0, x1, y0, y1], origin=origin_kw)
    if occ_mask is not None:
        ax.imshow(occ_mask.astype(float), extent=[x0, x1, y0, y1],
                  origin=origin_kw, cmap="Reds", alpha=0.25, vmin=0, vmax=1)

    # Plot loaded targets
    if len(targets) > 0:
        t = np.array(targets) if not isinstance(targets, np.ndarray) else targets
        ax.scatter(t[:, 0], t[:, 1], s=80, c="red",
                   edgecolors="white", linewidths=0.8, zorder=5, marker="X")
        for i, (px, py) in enumerate(t):
            ax.annotate(f" T{i+1}", (px, py), fontsize=8,
                        color="red", fontweight="bold")

    ax.set_xlabel("X [m]")
    ax.set_ylabel("Y [m]")
    ax.set_title(
        f"Loaded {len(targets)} target position(s)  —  "
        "Close window or press Enter to continue"
    )

    finished = False

    def _on_key(event):
        nonlocal finished
        if event.key in ("enter", "return", "escape"):
            finished = True
            plt.close(fig)

    fig.canvas.mpl_connect("key_press_event", _on_key)
    print(f"\n   Previewing {len(targets)} loaded target(s).  Close window or press Enter to continue.")
    plt.show()


def _pick_target_positions_interactive(map_path, m_per_px, origin_xy,
                                       origin_is_lower_left, occ_mask=None):
    """Let the user click on the map to place individual target positions.

    Each left-click adds a target.  Backspace undoes the last one.
    Enter confirms the selection.  Esc cancels (returns empty array).

    Parameters
    ----------
    map_path : str
        Path to the map image file.
    m_per_px : float
        Metres per pixel.
    origin_xy : tuple (x0, y0)
        World-coordinate origin of the map image.
    origin_is_lower_left : bool
        Whether the image origin is at the lower-left corner.
    occ_mask : np.ndarray or None
        Optional occupancy mask shown as a translucent red overlay so
        the user can see walls / obstacles when placing targets.

    Returns
    -------
    np.ndarray (N, 2) - selected positions in world coordinates.
    """
    from PIL import Image
    import matplotlib.pyplot as plt

    img = Image.open(map_path).convert("RGB")
    w_px, h_px = img.size
    x0, y0 = origin_xy
    x1 = x0 + w_px * m_per_px
    y1 = y0 + h_px * m_per_px
    origin_kw = "lower" if origin_is_lower_left else "upper"

    fig, ax = plt.subplots(figsize=(10, 8))
    ax.imshow(img, extent=[x0, x1, y0, y1], origin=origin_kw)
    if occ_mask is not None:
        ax.imshow(occ_mask.astype(float), extent=[x0, x1, y0, y1],
                  origin=origin_kw, cmap="Reds", alpha=0.25, vmin=0, vmax=1)
    ax.set_xlabel("X [m]")
    ax.set_ylabel("Y [m]")
    ax.set_title(
        "Click = place TARGET  |  Backspace = undo  |  "
        "Enter = finish  |  Esc = cancel"
    )

    pts = []
    scatter = ax.scatter([], [], s=80, c="red", edgecolors="white",
                         linewidths=0.8, zorder=5, marker="X")
    annotations: list = []
    finished = False

    def _update():
        if pts:
            scatter.set_offsets(np.array(pts))
        else:
            scatter.set_offsets(np.empty((0, 2)))
        for ann in annotations:
            ann.remove()
        annotations.clear()
        for i, (px, py) in enumerate(pts):
            ann = ax.annotate(
                f" T{i+1}", (px, py), fontsize=8,
                color="red", fontweight="bold",
            )
            annotations.append(ann)
        fig.canvas.draw()

    def _on_click(event):
        nonlocal finished
        if finished or event.inaxes != ax or event.button != 1:
            return
        pts.append((event.xdata, event.ydata))
        _update()
        print(f"   T{len(pts)}: ({event.xdata:.3f}, {event.ydata:.3f}) m")

    def _on_key(event):
        nonlocal finished
        if event.key in ("enter", "return"):
            finished = True
            plt.close(fig)
        elif event.key in ("backspace", "u") and pts:
            removed = pts.pop()
            print(f"Removed ({removed[0]:.3f}, {removed[1]:.3f})")
            _update()
        elif event.key == "escape":
            pts.clear()
            finished = True
            plt.close(fig)

    fig.canvas.mpl_connect("button_press_event", _on_click)
    fig.canvas.mpl_connect("key_press_event", _on_key)

    print("\n[Manual target selection] Click on the map to place targets.")
    print("   Left-click = add  |  Backspace = undo  |  Enter = done  |  Esc = cancel")
    logging.info("--- Target-picker window opened ---")
    plt.show()

    if not pts:
        print("No targets selected.")
        return np.empty((0, 2))

    result = np.array(pts)
    print(f"{len(result)} target position(s) selected.")
    return result


def _point_in_polygon(x, y, poly):
    """Ray-casting point-in-polygon test.  *poly* is (N,2) np.ndarray."""
    n = len(poly)
    inside = False
    px, py = x, y
    j = n - 1
    for i in range(n):
        xi, yi = poly[i]
        xj, yj = poly[j]
        if ((yi > py) != (yj > py)) and (px < (xj - xi) * (py - yi) / (yj - yi + 1e-12) + xi):
            inside = not inside
        j = i
    return inside


def _point_in_any_polygon(x, y, polys):
    """Return True if (x, y) is inside ANY polygon in the list."""
    for poly in polys:
        if _point_in_polygon(x, y, poly):
            return True
    return False


def _world_to_obstacle_dist(x, y, dist_map, meta):
    """Convert world (x,y) pixel and look up distance-to-obstacle (m)."""
    m_per_px = meta["m_per_px"]
    x0, y0 = meta["origin_xy"]
    h, w = meta["size"]
    col = int((x - x0) / m_per_px)
    if meta["origin_is_lower_left"]:
        row = int((y - y0) / m_per_px)
        row = h - 1 - row
    else:
        row = int((y - y0) / m_per_px)
    if 0 <= row < h and 0 <= col < w:
        return float(dist_map[row, col])
    return None


def _filter_target_positions(
    target_positions,
    forbidden_zones,
    near_obstacle_zone=None,
    obstacle_proximity_m=None,
    occ_result=None,
):
    """Pre-filter the full list of target positions by spatial constraints.

    This is applied **before** sending positions to the agent side so that
    the agent will never randomly pick a forbidden or invalid target.

    Parameters
    ----------
    target_positions : list of [x, y]
        All candidate target positions from the mask.
    forbidden_zones : list of np.ndarray polygons
        Zones where the target must NOT be placed.
    near_obstacle_zone : np.ndarray or None
        If set (RL_3), only positions inside this polygon are kept.
    obstacle_proximity_m : float or None
        For RL_3: max distance from target to nearest obstacle.
    occ_result : dict or None
        Contains 'dist_to_obstacle_m' and 'meta' for obstacle distance lookups.

    Returns
    -------
    list of [x, y]
        Filtered target positions that satisfy all constraints.
    """
    if not target_positions:
        return []

    occ_dist_map = occ_result.get("dist_to_obstacle_m", None) if occ_result else None
    occ_meta = occ_result.get("meta", None) if occ_result else None

    filtered = []
    for pos in target_positions:
        tx, ty = pos[0], pos[1]

        # Forbidden zone check
        if forbidden_zones and _point_in_any_polygon(tx, ty, forbidden_zones):
            continue

        # Near-obstacle mode (RL_3): only keep positions inside the zone
        if near_obstacle_zone is not None:
            if not _point_in_polygon(tx, ty, near_obstacle_zone):
                continue
            # Additionally check proximity to closest obstacle
            if occ_dist_map is not None and obstacle_proximity_m is not None and occ_meta is not None:
                dist_to_obs = _world_to_obstacle_dist(tx, ty, occ_dist_map, occ_meta)
                if dist_to_obs is None or dist_to_obs > obstacle_proximity_m:
                    continue

        filtered.append(pos)

    logging.info(
        f"[Filter] {len(target_positions)} → {len(filtered)} target positions "
        f"after applying constraints "
        f"(forbidden={len(forbidden_zones) if forbidden_zones else 0}, "
        f"near_obs={'yes' if near_obstacle_zone is not None else 'no'})."
    )
    return filtered


def _is_position_in_list(
    position: list,
    allowed: list,
    tol: float = 0.05,
) -> bool:
    """Return True if 'position' [x, y] matches any entry in 'allowed'
    within an Euclidean tolerance 'tol' (metres).
    """
    px, py = position[0], position[1]
    for a in allowed:
        if math.sqrt((px - a[0]) ** 2 + (py - a[1]) ** 2) < tol:
            return True
    return False


# ------------------------
# ------ CORE LOOP -------
# ------------------------

def _run_experiment(
    rl_copp,
    model,
    exp_name,
    exp_cfg,
    map_name,
    n_episodes,
    target_positions
):
    """Run 'n_episodes' of an experiment and return a summary dict + csv path.

    This is conceptually similar to the loop inside 'test.py' but adds the
    experiment-specific target sampling constraints defined above.
    """
    # ---- paths ----
    testing_metrics_path = rl_copp.paths["testing_metrics"]
    model_basename = os.path.splitext(os.path.basename(rl_copp.args.model_name))[0]
    exp_folder = os.path.join(
        testing_metrics_path,
        f"{model_basename}_custom_exp",
        f"{exp_name}_{os.path.splitext(map_name)[0]}",
    )
    os.makedirs(exp_folder, exist_ok=True)
    trajs_folder = os.path.join(exp_folder, "trajs")
    os.makedirs(trajs_folder, exist_ok=True)

    experiment_csv_name, experiment_csv_path = utils.get_output_csv(model_basename, exp_folder, f"{exp_name}_test")
    _, otherdata_csv_path = utils.get_output_csv(model_basename, exp_folder, f"{exp_name}_otherdata")

    # ---- extract experiment parameters ----
    min_d = exp_cfg["min_target_dist"]
    max_d = exp_cfg["max_target_dist"]

    # ---- accumulators ----
    rewards_list, time_list, timesteps_list = [], [], []
    terminated_list, collision_list, max_achieved_list = [], [], []
    target_zone_list, episode_dist_list = [], []
    skipped = 0

    start_time = time.time()

    # CSV headers (same schema as test.py for compatibility)
    metrics_headers = [
        "Experiment", "Map",
        "Initial distance (m)", "Reached distance (m)",
        "Time (s)", "Reward", "Target zone",
        "TimeSteps count", "Terminated", "Truncated",
        "Crashes", "Max limits achieved", "Distance traveled (m)",
    ]

    with open(experiment_csv_path, mode="w", newline="") as mf:
        writer = csv.writer(mf)
        writer.writerow(metrics_headers)

        # --- otherdata headers (written once) ---
        otherdata_headers = ["Episode number", "LAT-Sim (s)", "LAT-Wall (s)"]
        action_names = rl_copp.env.envs[0].unwrapped.params_env.get("action_names", [])
        observation_names = rl_copp.env.envs[0].unwrapped.params_env.get("observation_names", [])
        otherdata_headers += action_names + observation_names
        with open(otherdata_csv_path, mode="w", newline="") as of:
            csv.writer(of).writerow(otherdata_headers)

        valid_episodes = 0
        max_total_resets = n_episodes * 20
        total_resets = 0
        pbar = tqdm(total=n_episodes, desc=f"{exp_name} | {map_name}", unit="ep")

        while valid_episodes < n_episodes and total_resets < max_total_resets:
            total_resets += 1

            # --- Reset environment to get a fresh episode ---
            # The agent side handles robot/target placement using the
            # robot_pos_samples and target_pos_samples that were set on
            # rl_copp before starting CoppeliaSim.  After reset() the
            # observation already reflects the new positions chosen by
            # the agent.
            observation, info_obs = rl_copp.env.envs[0].reset()
            env_unwrapped = rl_copp.env.envs[0].unwrapped

            # --- Validate target position against experiment targets ---
            # The agent side picks randomly from target_pos_samples
            # (the union of all experiments). If this experiment has a
            # restricted target list (e.g. RL_3 manual targets), we
            # must verify that the placed target belongs to that list.
            if info_obs and "target_position" in info_obs:
                placed_target = info_obs["target_position"]
                if not _is_position_in_list(placed_target, target_positions, tol=0.05):
                    skipped += 1
                    logging.debug(
                        f"[{exp_name}] Reset {total_resets}: target "
                        f"({placed_target[0]:.2f}, {placed_target[1]:.2f}) "
                        f"not in experiment target list. Re-rolling."
                    )
                    continue

            # --- Validate distance constraint from observation ---
            # observation[0] is the distance robot→target (first element
            # in the observation vector, as defined by observation_names).
            initial_distance = float(observation[0])

            if initial_distance < min_d or initial_distance > max_d:
                # The agent placed the target outside the desired distance
                # band.  Reset again without counting this as an episode.
                skipped += 1
                logging.debug(
                    f"[{exp_name}] Reset {total_resets}: initial distance "
                    f"{initial_distance:.2f} m outside [{min_d}, {max_d}] m. "
                    f"Re-rolling."
                )
                continue

            valid_episodes += 1
            n_ep = valid_episodes

            # --- Initialise episode metrics ---
            utils.init_metrics_test(env_unwrapped)

            terminated = False
            truncated = False

            while not (terminated or truncated):
                action, _states = model.predict(observation, deterministic=True)
                observation, _, terminated, truncated, info = rl_copp.env.envs[0].step(action)

                # Write step-level data
                with open(otherdata_csv_path, mode="a", newline="") as of:
                    ow = csv.writer(of)
                    action_values = [round(v, 4) for v in info["actions"].values()]
                    obs_values = [round(float(v), 4) for v in observation.tolist()]
                    lat_values = [info["lat_sim"], info["lat_wall"]]
                    ow.writerow([n_ep] + lat_values + action_values + obs_values)

            # --- Collect episode metrics ---
            (init_d, final_d, t_reach, reward,
             ts_count, collision, max_ach, tz) = utils.get_metrics_test(env_unwrapped)

            if ts_count == 1:
                logging.info(f"Episode {n_ep} discarded (single timestep).")
                skipped += 1
                valid_episodes -= 1  # don't count this as a valid episode
                continue

            rewards_list.append(reward)
            time_list.append(t_reach)
            timesteps_list.append(ts_count)
            terminated_list.append(terminated)
            collision_list.append(collision)
            max_achieved_list.append(max_ach)
            target_zone_list.append(tz)

            ep_dist = 0.0
            if getattr(rl_copp.args, "save_traj", False):
                traj_file = f"trajectory_{n_ep}.csv"
                ep_dist = utils.calculate_episode_distance(trajs_folder, traj_file)
            episode_dist_list.append(ep_dist)

            writer.writerow([
                exp_name, map_name,
                init_d, final_d, t_reach, reward, tz,
                ts_count, terminated, truncated,
                collision, max_ach, ep_dist,
            ])
            pbar.update(1)

        pbar.close()

        if total_resets >= max_total_resets and valid_episodes < n_episodes:
            logging.warning(
                f"[{exp_name}] Safety cap reached: {total_resets} resets but "
                f"only {valid_episodes}/{n_episodes} valid episodes.  "
                f"The distance band [{min_d}, {max_d}] m may be too narrow "
                f"for the current map/positions."
            )

    end_time = time.time()

    # ---- Summary ----
    n_valid = len(rewards_list)
    summary = {
        "experiment": exp_name,
        "map": map_name,
        "episodes_requested": n_episodes,
        "episodes_valid": n_valid,
        "episodes_skipped": skipped,
        "avg_reward": (sum(rewards_list) / n_valid) if n_valid else 0,
        "avg_time": (sum(time_list) / n_valid) if n_valid else 0,
        "avg_timesteps": (sum(timesteps_list) / n_valid) if n_valid else 0,
        "success_rate_%": (
            (1 - (sum(max_achieved_list) + sum(collision_list)) / n_valid) * 100
        ) if n_valid else 0,
        "collision_rate_%": (sum(collision_list) / n_valid * 100) if n_valid else 0,
        "target_zone_1_%": (target_zone_list.count(1) / n_valid * 100) if n_valid else 0,
        "target_zone_2_%": (target_zone_list.count(2) / n_valid * 100) if n_valid else 0,
        "target_zone_3_%": (target_zone_list.count(3) / n_valid * 100) if n_valid else 0,
        "avg_distance_per_episode_m": (sum(episode_dist_list) / n_valid) if n_valid else 0,
        "duration_s": end_time - start_time,
        "csv_path": experiment_csv_path,
    }

    # ---- Persist summary as JSON alongside the CSV ----
    summary_path = os.path.join(exp_folder, f"{exp_name}_summary.json")
    with open(summary_path, "w") as fp:
        json.dump(summary, fp, indent=2)
    logging.info(f"[{exp_name}] Summary saved → {summary_path}")

    # ---- Update global test_records (same format as test.py) ----
    record_csv = os.path.join(rl_copp.paths["testing_metrics"], "test_records.csv")

    n_collisions = sum(collision_list)
    pct_not_finished = (
        ((sum(max_achieved_list) + n_collisions) / n_valid) * 100
    ) if n_valid else 0

    data_to_store = {
        "Algorithm": rl_copp.params_test.get("sb3_algorithm", ""),
        "Experiment": exp_name,
        "Map": map_name,
        "Avg reward": summary["avg_reward"],
        "Avg time reach target": summary["avg_time"],
        "Avg timesteps": summary["avg_timesteps"],
        "Percentage terminated": 100 - pct_not_finished,
        "Percentage truncated": pct_not_finished,
        "Number of collisions": n_collisions,
        "Target zone 1 (%)": summary["target_zone_1_%"],
        "Target zone 2 (%)": summary["target_zone_2_%"],
        "Target zone 3 (%)": summary["target_zone_3_%"],
        "Average distance per episode (m)": summary["avg_distance_per_episode_m"],
    }
    utils.update_records_file(record_csv, experiment_csv_name, start_time, end_time, data_to_store)

    return summary


# ------------------------
# ----- main entry -------
# ------------------------


def main(args):
    """Run custom experiments to evaluate trained RL policies.

    The workflow:
      1. Initialise the RLCoppeliaManager (loads params, sets paths).
      2. Ask the user to select maps (or use --maps).
      3. For each experiment requested (--experiments), ask for parameter
         overrides and let the user draw allowed/forbidden zones on each map.
      4. Load the trained model.
      5. For every (experiment x map) pair run 'N' episodes, collecting
         the same metrics as ``test.py``.
      6. Save per-experiment CSVs and a global summary JSON.
    """
    rl_copp = RLCoppeliaManager(args)

    # --- 1. Select maps 
    maps_dir = os.path.join(rl_copp.base_path, "custom_maps")
    map_names = _select_maps_interactive(maps_dir, getattr(args, "maps", None) or [])
    logging.info(f"Maps selected: {map_names}")

    # --- 2. Select & configure experiments 
    exp_names = getattr(args, "experiments", None) or list(EXPERIMENT_CATALOG.keys())
    experiments = {}
    for name in exp_names:
        if name not in EXPERIMENT_CATALOG:
            logging.warning(f"Unknown experiment '{name}' — skipping.")
            continue
        cfg = _configure_experiment_interactive(name, EXPERIMENT_CATALOG[name]["defaults"])
        experiments[name] = cfg
    if not experiments:
        raise ValueError("No valid experiments selected.")

    n_episodes = getattr(args, "episodes", None) or 500
    logging.info(f"Episodes per (experiment × map): {n_episodes}")

    # --- 3. Map metadata defaults (can be overridden with --set) 
    origin_is_lower_left = False

    # --- 4. Pre-compute valid positions & zones for every (exp, map)
    #    Structure:  map_data[map_name] = { occ_result, robot_pos, target_pos }
    #                zone_data[(exp_name, map_name)] = { forbidden, near_obs }
    map_data = {}
    zone_data = {}

    for map_name in map_names:
        map_path = os.path.join(maps_dir, map_name)
        print(f"\n{'─'*60}")
        print(f"  Processing map: {map_name}")
        print(f"{'─'*60}")

        m_per_px, origin_xy = utils.extract_map_parameters(map_path)

        # --- Robot positions: build occupancy + valid grid positions
        # build_valid_positions_from_map already handles saving/loading
        # the polygon mask with masc_tag="robot" in the filename,
        # so robot and target masks are stored separately.
        occ_result = utils.build_valid_positions_from_map(
            map_path,
            m_per_px=m_per_px,
            origin_xy=origin_xy,
            origin_is_lower_left=origin_is_lower_left,
            obstacle_threshold=50,
            clearance_m=0.25,
            grid_step_m=0.25,
            interactive_polygon=True,
            masc_tag="robot",
        )
        robot_pos = occ_result["positions_xy"].tolist()
        utils.preview_mask_and_positions(map_path, occ_result)

        # Target positions — only needed when at least one experiment
        # uses the standard mask-based target placement (not manual).
        needs_target_mask = any(
            not cfg.get("manual_target_selection", False)
            for cfg in experiments.values()
        )

        if needs_target_mask:
            print("\n[target positions] Now define the allowed area for target placement.")
            occ_result_target = utils.build_valid_positions_from_map(
                map_path,
                m_per_px=m_per_px,
                origin_xy=origin_xy,
                origin_is_lower_left=origin_is_lower_left,
                obstacle_threshold=50,
                clearance_m=0.25,
                grid_step_m=0.20,
                interactive_polygon=True,
                masc_tag="target",
            )
            target_pos = occ_result_target["positions_xy"].tolist()
            utils.preview_mask_and_positions(map_path, occ_result_target)
        else:
            # No mask-based experiments: reuse robot occupancy for reference
            occ_result_target = occ_result
            target_pos = []

        map_data[map_name] = {
            "occ_result": occ_result,
            "occ_result_target": occ_result_target,
            "robot_pos": robot_pos,
            "target_pos": target_pos,
        }

        # Per-experiment zones for this map
        for exp_name, exp_cfg in experiments.items():
            forbidden = []
            near_obs = None
            manual_targets = None

            if exp_cfg.get("manual_target_selection", False):
                # RL_3-style: user clicks individual target positions
                print(f"\n── Manual target selection for {exp_name} on {map_name} ──")

                # Check if a previously saved target list exists for this map
                saved_targets_path = _get_manual_targets_path(maps_dir, map_name, exp_name)
                picked = None

                if os.path.isfile(saved_targets_path):
                    print(f"   A saved target list was found for this map:")
                    print(f"     '{os.path.basename(saved_targets_path)}'")
                    load_choice = input(">> Load saved targets? [Y/n]: ").strip().lower()
                    if load_choice not in ("n", "no"):
                        picked = _load_manual_targets(saved_targets_path)
                        # Visualize loaded targets on the map so the user can review them
                        _visualize_targets_on_map(
                            map_path, m_per_px, origin_xy, origin_is_lower_left,
                            picked, occ_mask=occ_result.get("occ_mask", None),
                        )

                if picked is None:
                    # No saved file or user chose not to load — interactive selection
                    picked = _pick_target_positions_interactive(
                        map_path, m_per_px, origin_xy, origin_is_lower_left,
                        occ_mask=occ_result.get("occ_mask", None),
                    )
                    # Offer to save the newly selected targets
                    if isinstance(picked, np.ndarray) and picked.size > 0:
                        save_choice = input(">> Save these target positions for future runs? [Y/n]: ").strip().lower()
                        if save_choice not in ("n", "no"):
                            _save_manual_targets(saved_targets_path, picked)

                # Convert to list for downstream usage
                if isinstance(picked, np.ndarray) and picked.size > 0:
                    manual_targets = picked.tolist()
                elif isinstance(picked, list) and len(picked) > 0:
                    manual_targets = picked
                else:
                    logging.warning(
                        f"No targets placed for {exp_name} on {map_name} — "
                        f"experiment will be skipped on this map."
                    )
            else:
                if exp_cfg.get("use_forbidden_zones", False):
                    print(f"\n-- Forbidden zones for {exp_name} on {map_name} ──")
                    forbidden = _draw_forbidden_zones(map_path, m_per_px, origin_xy, origin_is_lower_left)

                if exp_cfg.get("near_obstacle_mode", False):
                    near_obs = _draw_near_obstacle_zone(map_path, m_per_px, origin_xy, origin_is_lower_left)

            zone_data[(exp_name, map_name)] = {
                "forbidden": forbidden,
                "near_obs": near_obs,
                "manual_targets": manual_targets,
            }

    # --- 5. Resolve scene ↔ map assignments 
    #    Each map requires a different CoppeliaSim scene.  We precompute
    #    the mapping here so we can detect errors before launching sims.
    map_scene = {}
    for map_name in map_names:
        scene_file = _resolve_scene_for_map(map_name)
        map_scene[map_name] = scene_file
        logging.info(f"Map '{map_name}' → scene '{scene_file}'")

    # --- 6. Resolve model path
    models_path = rl_copp.paths["models"]
    training_metrics_path = rl_copp.paths["training_metrics"]

    if rl_copp.args.model_name is None:
        _, rl_copp.args.model_name = utils.get_last_model(models_path)
    else:
        rl_copp.args.model_name = os.path.join(models_path, rl_copp.args.model_name)

    logging.info(f"Model: {rl_copp.args.model_name}")

    model_basename = os.path.splitext(os.path.basename(rl_copp.args.model_name))[0]
    train_records_csv = os.path.join(training_metrics_path, "train_records.csv")
    try:
        rl_copp.params_test["sb3_algorithm"] = utils.get_data_from_training_csv(
            model_basename, train_records_csv, "Algorithm"
        )
    except Exception:
        rl_copp.params_test["sb3_algorithm"] = rl_copp.params_train["sb3_algorithm"]

    ModelClass = getattr(stable_baselines3, rl_copp.params_test["sb3_algorithm"])

    # custom_objects to handle lr_schedule deserialization issues that arise
    # when the model was saved with a different Python / SB3 version.  A constant
    # learning-rate function is safe because we are only doing inference (no
    # gradient updates).
    custom_objects = {
        "lr_schedule": lambda _: 3e-4,       # constant lr (unused at test time)
        "learning_rate": 3e-4,
    }

    # --- 7. Loop over maps  (outer) × experiments (inner)
    #    Each map needs its own CoppeliaSim scene, so we restart the
    #    simulation for every map change.
    all_summaries = []
    first_map = True

    for map_name in map_names:
        scene_file = map_scene[map_name]
        md = map_data[map_name]

        print(f"\n{'━'*60}")
        print(f"  Launching scene '{scene_file}' for map '{map_name}'")
        print(f"{'━'*60}")

        # Stop previous simulation (if not the first map)
        if not first_map:
            try:
                rl_copp.env.envs[0].unwrapped._commstoagent.stepExpFinished()
            except Exception:
                pass
            try:
                rl_copp.stop_coppelia_sim()
            except Exception as e:
                logging.warning(f"Could not stop previous simulation cleanly: {e}")

        # Switch scene via params_train.scene_name 
        rl_copp.params_train["scene_name"] = scene_file
        # Reset scene_path so start_coppelia_sim re-resolves it from params
        rl_copp.args.scene_path = None

        # Disable obstacle generation: the real map already has its own
        # walls/obstacles, so we do not want the agent to create random
        # columns on every reset.
        if scene_file == "burgerBot_real_scene1.ttt":
            rl_copp.params_scene["scene_x_dim"] = 11
            rl_copp.params_scene["scene_y_dim"] = 11
        elif scene_file == "burgerBot_real_scene2.ttt":
            rl_copp.params_scene["scene_x_dim"] = 15
            rl_copp.params_scene["scene_y_dim"] = 15
        elif scene_file == "burgerBot_real_scene3.ttt":
            rl_copp.params_scene["scene_x_dim"] = 16.3
            rl_copp.params_scene["scene_y_dim"] = 16.3
        rl_copp.params_scene["n_obstacles"] = 0
        rl_copp.params_scene["fixed_obs"] = False

        # Send valid positions to the agent side 
        rl_copp.robot_pos_samples = md["robot_pos"]
        logging.info(
            f"[{map_name}] {len(rl_copp.robot_pos_samples)} robot positions "
            f"will be sent to Agent side."
        )

        all_valid_targets = set()
        for exp_name, exp_cfg in experiments.items():
            zd = zone_data[(exp_name, map_name)]

            if zd.get("manual_targets"):
                # Manual target selection (RL_3): use picked positions directly
                for pos in zd["manual_targets"]:
                    all_valid_targets.add((pos[0], pos[1]))
            else:
                # Mask-based experiments (RL_1, RL_2): filter from target mask
                filtered = _filter_target_positions(
                    target_positions=md["target_pos"],
                    forbidden_zones=zd["forbidden"],
                    near_obstacle_zone=zd["near_obs"],
                    obstacle_proximity_m=exp_cfg.get("obstacle_proximity_m", None),
                    occ_result=md["occ_result_target"],
                )
                for pos in filtered:
                    all_valid_targets.add((pos[0], pos[1]))

        rl_copp.target_pos_samples = [list(p) for p in all_valid_targets]
        logging.info(
            f"[{map_name}] {len(rl_copp.target_pos_samples)} target positions "
            f"(union across experiments) will be sent to Agent side."
        )

        # Create env, start sim, establish comms 
        if not args.agent_side:
            rl_copp.create_env()
        if not args.rl_side:
            rl_copp.start_coppelia_sim("TestCustomExp")
        if not args.agent_side:
            rl_copp.start_communication()

        # Load model into the new environment 
        model = ModelClass.load(
            rl_copp.args.model_name,
            env=rl_copp.env,
            custom_objects=custom_objects,
        )
        logging.info(f"Model loaded (algo={rl_copp.params_test['sb3_algorithm']})")

        # Run every experiment on this map
        for exp_name, exp_cfg in experiments.items():
            zd = zone_data[(exp_name, map_name)]

            # Build the per-experiment target list.
            # For manual-target experiments the user already picked the
            # exact positions; for mask-based ones we filter spatially.
            if zd.get("manual_targets"):
                exp_targets = zd["manual_targets"]
            else:
                exp_targets = _filter_target_positions(
                    target_positions=md["target_pos"],
                    forbidden_zones=zd["forbidden"],
                    near_obstacle_zone=zd["near_obs"],
                    obstacle_proximity_m=exp_cfg.get("obstacle_proximity_m", None),
                    occ_result=md["occ_result_target"],
                )

            if not exp_targets:
                logging.warning(
                    f"[{exp_name}] No valid target positions for {map_name} — skipping."
                )
                continue

            logging.info(
                f"\n▶ Running {exp_name} on {map_name}  ({n_episodes} episodes, "
                f"{len(exp_targets)} valid target positions)"
            )
            summary = _run_experiment(
                rl_copp=rl_copp,
                model=model,
                exp_name=exp_name,
                exp_cfg=exp_cfg,
                map_name=map_name,
                n_episodes=n_episodes,
                robot_positions=md["robot_pos"],
                target_positions=exp_targets,
                forbidden_zones=zd["forbidden"],
                near_obstacle_zone=zd["near_obs"],
                occ_result=md["occ_result_target"],
            )
            all_summaries.append(summary)

        first_map = False

    # --- 8. Global summary 
    global_summary_path = os.path.join(
        rl_copp.paths["testing_metrics"],
        f"{model_basename}_custom_exp",
        "global_summary.json",
    )
    os.makedirs(os.path.dirname(global_summary_path), exist_ok=True)
    with open(global_summary_path, "w") as fp:
        json.dump(all_summaries, fp, indent=2)

    print(f"\n{'═'*60}")
    print("  CUSTOM EXPERIMENT EVALUATION — RESULTS")
    print(f"{'═'*60}")
    for s in all_summaries:
        print(
            f"  {s['experiment']:6s} | {s['map']:25s} | "
            f"SR={s['success_rate_%']:5.1f}%  "
            f"R̄={s['avg_reward']:+.2f}  "
            f"T̄={s['avg_time']:.1f}s  "
            f"Col={s['collision_rate_%']:.1f}%  "
            f"({s['episodes_valid']}/{s['episodes_requested']} eps)"
        )
    print(f"\n  Global summary → {global_summary_path}")
    print(f"{'═'*60}\n")

    # --- 9. Cleanup 
    rl_copp.env.envs[0].unwrapped._commstoagent.stepExpFinished()
    logging.info("Custom experiment evaluation finished.")
    rl_copp.remove_tmp_data()


if __name__ == "__main__":
    main()
