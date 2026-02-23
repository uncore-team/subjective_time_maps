import os

from common import utils
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3" # Suppress TensorFlow warnings
os.environ["GRPC_VERBOSITY"] = "ERROR"
import argparse

def main(argv=None):
    """
    Entry point for the uncore_rl CLI. Handles argument parsing and dispatches to the correct subcommand.

    Args:
        argv (list[str], optional): List of CLI arguments (for programmatic use). Defaults to None.
    """
    parser = argparse.ArgumentParser(prog="uncore_rl", description="Training and testing CLI")
    subparsers = parser.add_subparsers(dest="command")

    # --------------------------
    # ----- ROBOT CREATION -----
    # --------------------------

    create_robot_parser = subparsers.add_parser("create_robot", help="Create an environment and its corresponding agent for CoppeliaSim simulation.")


    # -------------------------------------------------------------------------------
    # ----- STANDARD FEATURES: TRAINING, TESTING, PLOTTING, SAVING, TENSORBOARD -----
    # -------------------------------------------------------------------------------

    train_parser = subparsers.add_parser("train", help="Train a RL algorithm for robot movement in CoppeliaSim.")
    train_parser.add_argument("--robot_name", type=str, help="Name for the robot.", required=True)
    train_parser.add_argument("--scene_path", type=str, help="Path to the CoppeliaSim scene file.", required=False)
    train_parser.add_argument("--dis_parallel_mode", action="store_true", help="Disables the parallel training or testing.", required=False)
    train_parser.add_argument("--no_gui", action="store_true", help="Disables Coppelia GUI, it will just show the terminal", required=False)
    train_parser.add_argument("--params_file", type=str, help="Path to the configuration file.",required=False)
    train_parser.add_argument("--obstacles_csv_folder", type=str, help="Path to scene configuration folder in case that we want to train with fixed obstacles. Please just indicate the folder not the whole path (e.g. 'Scene014')",required=False)
    train_parser.add_argument("--dis_save_notes", action="store_true", help="Flag to save some notes for the experiment.", default = False, required=False)
    train_parser.add_argument("--rl_side", action="store_true", help="Flag to just execute rl side code.", default = False, required=False)
    train_parser.add_argument("--agent_side", action="store_true", help="Flag to just execute agent side code.", default = False, required=False)
    train_parser.add_argument("--comms_port", type=int, help="Number of the port to start the communications with the RL Side.", required=False)
    train_parser.add_argument("--ip_address", type=str, help="IP address of the RL Side.", required=False)
    train_parser.add_argument("--timestamp", type=str, help="Timestamp provided externally (e.g., from GUI).", required=False)
    train_parser.add_argument("--set", dest="overrides", action="append", default=[], metavar="KEY=VALUE", help="Override of params.json parameters (use dot notation).")
    train_parser.add_argument("--verbose", type=int, help="Enable debugging through info logs using the terminal. 0: no logs at all. \
                             1: just a progress bar and save warnings. 2: just a progress bar and save everything. 3: all logs shown and saved for debugging. Other: just terminal, logs are not saved", default=0, required=False)

    test_parser = subparsers.add_parser("test", help="Test a trained RL algorithm for robot movement in CoppeliaSim.")
    test_parser.add_argument("--model_name", type=str, help="Name of the trained model is required (it must be located under 'models' folder)", required=True)
    test_parser.add_argument("--robot_name", type=str, help="Name for the robot. Default will be burgerBot.", required=False)
    test_parser.add_argument("--scene_path", type=str, help="Path to the CoppeliaSim scene file.", required=False)
    test_parser.add_argument("--save_scene", action="store_true", help="Enables saving scene mode.", required=False, default=False)
    test_parser.add_argument("--save_traj", action="store_true", help="Enables saving trajectory mode.", required=False, default=False)
    test_parser.add_argument("--grid_target_points", action="store_true", help="Activates grid target points mode.", required=False, default=False)
    test_parser.add_argument("--grid_robot_points", action="store_true", help="Activates grid robot points mode.", required=False, default=False)
    test_parser.add_argument("--map_name", type=str, help="Name of the map file (e.g., 'turtlemap.pgm'). Located in base_path/custom_maps/. Used for precomputing possible target/robot positions.", required=False)
    test_parser.add_argument("--obstacles_csv_folder", type=str, help="Path to scene configuration folder in case that we want to test with fixed obstacles. Please just indicate the folder not the whole path (e.g. /Scene014)",required=False)
    test_parser.add_argument("--dis_parallel_mode", action="store_true", help="Disables the parallel training or testing.", required=False)
    test_parser.add_argument("--no_gui", action="store_true", help="Disables Coppelia GUI, it will just show the terminal", required=False)
    test_parser.add_argument("--params_file", type=str, help="Path to the configuration file.",required=False)
    test_parser.add_argument("--iterations", type=int, help="Number of iterations for the test. If set, it will override the parameter from the parameters' json file.",required=False)
    test_parser.add_argument("--dis_save_notes", action="store_true", help="Flag to save some notes for the experiment.", default = False, required=False)
    test_parser.add_argument("--rl_side", action="store_true", help="Flag to just execute rl side code.", default = False, required=False)
    test_parser.add_argument("--agent_side", action="store_true", help="Flag to just execute agent side code.", default = False, required=False)
    test_parser.add_argument("--comms_port", type=int, help="Number of the port to start the communications with the RL Side.", required=False)
    test_parser.add_argument("--ip_address", type=str, help="IP address of the RL Side.", required=False)
    test_parser.add_argument("--timestamp", type=str, help="Timestamp provided externally (e.g., from GUI).", required=False)
    test_parser.add_argument("--set", dest="overrides", action="append", default=[], metavar="KEY=VALUE", help="Override of params.json parameters (use dot notation).")
    test_parser.add_argument("--verbose", type=int, help="Enable debugging through info logs using the terminal. 0: no logs at all. \
                             1: just a progress bar and save warnings. 2: just a progress bar and save everything. 3: all logs shown and saved for debugging. Other: just terminal, logs are not saved", default=0, required=False)

    plot_parser = subparsers.add_parser("plot", help="Creates a set of plots for getting the results of a trained model or for comparing some models.")
    plot_parser.add_argument("--robot_name", type=str, help="Name for the robot.", required=True)
    plot_parser.add_argument("--plot_types", type=str, nargs='+', help="List of types of plots that the user wants to create.", required=True)
    plot_parser.add_argument("--model_ids", type=int, nargs='+', help="List with numerical IDs of the different models to be plotted.", required=False)
    plot_parser.add_argument("--scene_to_load_folder", type=str, help="Folder name, located inside 'scene_configs', that contains the scene and trajectories to be loaded", required=False)
    plot_parser.add_argument("--save_plots", action="store_true", help="Saves the plots inside current folder instead of showing them.", required=False, default=False)
    plot_parser.add_argument("--lat_fixed_timestep", type=float, help="Fixed timestep for LAT plots (optional).", default=0, required=False)
    plot_parser.add_argument("--timestep_unit", type=str, help="Unit for timestep for LAT plots (optional).", default="s", required=False)
    plot_parser.add_argument("--csv_file_name", type=str, nargs='+', help="Path to CSV file(s). For timestep_map: pass multiple files to merge data from different experiments (e.g., facing target + facing away).", required=False)
    plot_parser.add_argument("--map_name", type=str, help="Name of the map file (e.g., 'turtlemap.pgm'). Located in base_path/custom_maps/. Used for building timestep map.", required=False)
    plot_parser.add_argument("--params_file", type=str, help="Path to the configuration file.",required=False)
    plot_parser.add_argument("--verbose", type=int, help="Enable debugging through info logs using the terminal. 0: no logs at all. \
                             1: just a progress bar and save warnings. 2: just a progress bar and save everything. 3: all logs shown and saved for debugging. Other: just terminal, logs are not saved", default=-1, required=False)
    plot_parser.add_argument("--plot_set", action="append", default=[], metavar="KEY=VALUE",help=("Override plot_timesteps_map keyword arguments. Example: --plot_set grid_cell=0.25 --plot_set origin_xy=-10.5,-6.0"),)

    
    save_parser = subparsers.add_parser("save", help="Save a trained model, along with all the date generated during its training/testing processes.")
    save_parser.add_argument("--model_name", type=str, help="Name of the model to be saved (it must be located under 'models' folder)", required=True)
    save_parser.add_argument("--new_name", type=str, help="New name for saving the model", required=True)
    save_parser.add_argument("--verbose", type=int, help="Enable debugging through info logs using the terminal. 0: no logs at all. \
                             1: just a progress bar and save warnings. 2: just a progress bar and save everything. 3: all logs shown and saved for debugging. Other: just terminal, logs are not saved", default=-1, required=False)

    tf_start_parser = subparsers.add_parser("tf_start", help="Starts the tensorboard to check the metrics generated during the training of a model.")
    tf_start_parser.add_argument("--model_name", type=str, help="Name of the model to be checked (it must be located under 'models' folder)", required=True)
    tf_start_parser.add_argument("--verbose", type=int, help="Enable debugging through info logs using the terminal. 0: no logs at all. \
                             1: just a progress bar and save warnings. 2: just a progress bar and save everything. 3: all logs shown and saved for debugging. Other: just terminal, logs are not saved", default=-1, required=False)

    # -------------------------------------
    # ----- SPECIALIZED TESTING MODES -----
    # -------------------------------------

    # Testing a trained RL algorithm for robot movement in CoppeliaSim following a path defined by a Dummy with ctrlPt* children.
    test_path_parser = subparsers.add_parser("test_path", help="Test a trained RL algorithm for robot movement in CoppeliaSim.")
    test_path_parser.add_argument("--model_name", type=str, required=True,help="Model file name under models/ (or full path).")
    test_path_parser.add_argument("--scene_path", type=str, help="Path to the CoppeliaSim scene file.", required=False)
    test_path_parser.add_argument("--path_alias", type=str, default="/RecordedPath",help="Alias/path of the Dummy that contains ctrlPt* children.")
    test_path_parser.add_argument("--trials_per_sample", type=int, default=10,help="Random target placements per sampled pose.")
    test_path_parser.add_argument("--n_samples", type=int, default=50,help="POints number to sample the path.")
    test_path_parser.add_argument("--n_extra_poses", type=int, default=2,help="Extra robot poses for testing the robot at each scenario in each direction, changing the orientation of the robot in 'delta_deg' degrees each time.")
    test_path_parser.add_argument("--delta_deg", type=float, default=5,help="Number of degrees to change the orientation of the robot at each iteration.")
    test_path_parser.add_argument("--robot_world_ori", type=float, default=0,help="Orientation in degrees between the robot and the world.")
    test_path_parser.add_argument("--robot_target_ori", type=float, default=0,help="Orientation in degrees between the robot and the target.")
    test_path_parser.add_argument("--map_name", type=str, help="Name of the map file (e.g., 'turtlemap.pgm'). Located in base_path/custom_maps/. If used, activates grid mode for testing robot at each cell.", required=False)
    test_path_parser.add_argument("--place_obstacles_flag", action="store_true", help="Flag for placing obstacles in the scene or not.", default=False, required=False)
    test_path_parser.add_argument("--random_target_flag", action="store_true", help="Flag for placing target randomly or not.", default=False, required=False)
    test_path_parser.add_argument("--robot_name", type=str, help="Name of the robot to be tested.")
    test_path_parser.add_argument("--no_gui", action="store_true", help="Run Coppelia without GUI.")
    test_path_parser.add_argument("--agent_side", action="store_true", help="Only start agent side (advanced).")
    test_path_parser.add_argument("--rl_side", action="store_true", help="Only start RL side (advanced).")
    test_path_parser.add_argument("--verbose", type=int, default=1, choices=[0,1,2,3], help="Verbosity (0..3).")
    test_path_parser.add_argument("--timestamp", type=str, help="Timestamp provided externally (e.g., from GUI).", required=False)
    test_path_parser.add_argument("--dis_parallel_mode", action="store_true", help="Disables the parallel training or testing.", required=False)
    test_path_parser.add_argument("--comms_port", type=int, help="Number of the port to start the communications with the RL Side.", required=False)
    test_path_parser.add_argument("--ip_address", type=str, help="IP address of the RL Side.", required=False)
    test_path_parser.add_argument("--params_file", type=str, help="Path to the configuration file.",required=False)
    test_path_parser.add_argument("--save_scene", action="store_true", help="Enables saving scene mode.", required=False, default=False)
    test_path_parser.add_argument("--save_traj", action="store_true", help="Enables saving trajectory mode.", required=False, default=False)
    test_path_parser.add_argument("--obstacles_csv_folder", type=str, help="Path to scene configuration folder in case that we want to test with fixed obstacles. Please just indicate the folder not the whole path (e.g. /Scene014)",required=False)
    test_path_parser.add_argument("--set", dest="overrides", action="append", default=[], metavar="KEY=VALUE", help="Override of params.json parameters (use dot notation).")


    # test_map: Grid-based evaluation with fixed target 
    test_map_parser = subparsers.add_parser("test_map", help="Grid-based evaluation with fixed target. Robot teleports to grid positions facing the target, with orientation augmentation and target noise.")
    test_map_parser.add_argument("--model_name", type=str, required=True, help="Model file name under models/ (or full path).")
    test_map_parser.add_argument("--map_name", type=str, required=False, help="Name of the map file (e.g., 'turtlemap.pgm'). Located in base_path/custom_maps/. If not provided, user will be prompted to select one.")
    test_map_parser.add_argument("--n_extra_poses", type=int, default=2, help="Extra robot orientations on each side of the base orientation (facing target). Total orientations = 1 + 2*n_extra_poses.")
    test_map_parser.add_argument("--delta_deg", type=float, default=5.0, help="Angular increment in degrees for orientation augmentation.")
    test_map_parser.add_argument("--target_noise_std", type=float, default=0.1, help="Standard deviation (meters) of Gaussian noise applied to target position.")
    test_map_parser.add_argument("--target_trials", type=int, default=10, help="Number of noisy target samples per augmented robot orientation.")
    test_map_parser.add_argument("--trials_per_sample", type=int, default=1, help="Number of repeated measurements per test case (to reduce observation noise).")
    test_map_parser.add_argument("--face_away", action="store_true", default=False, help="If set, robot base orientation faces AWAY from target (opposite direction) instead of towards it.")
    test_map_parser.add_argument("--scene_path", type=str, help="Path to the CoppeliaSim scene file.", required=False)
    test_map_parser.add_argument("--robot_name", type=str, help="Name of the robot to be tested.")
    test_map_parser.add_argument("--no_gui", action="store_true", help="Run Coppelia without GUI.")
    test_map_parser.add_argument("--agent_side", action="store_true", help="Only start agent side (advanced).")
    test_map_parser.add_argument("--rl_side", action="store_true", help="Only start RL side (advanced).")
    test_map_parser.add_argument("--verbose", type=int, default=1, choices=[0, 1, 2, 3], help="Verbosity (0..3).")
    test_map_parser.add_argument("--timestamp", type=str, help="Timestamp provided externally (e.g., from GUI).", required=False)
    test_map_parser.add_argument("--dis_parallel_mode", action="store_true", help="Disables parallel mode.", required=False)
    test_map_parser.add_argument("--comms_port", type=int, help="Port for communications with the RL Side.", required=False)
    test_map_parser.add_argument("--ip_address", type=str, help="IP address of the RL Side.", required=False)
    test_map_parser.add_argument("--params_file", type=str, help="Path to the configuration file.", required=False)
    test_map_parser.add_argument("--set", dest="overrides", action="append", default=[], metavar="KEY=VALUE", help="Override of params.json parameters (use dot notation).")

    # Testing a trained RL algorithm for robot movement in CoppeliaSim for just one iteration, using a preconfigured scene.
    test_scene_parser = subparsers.add_parser("test_scene", help="Test a trained RL algorithm for robot movement in CoppeliaSim for just one iteration, using a preconfigured scene.")
    test_scene_parser.add_argument("--model_ids", type=int, nargs='+', help="List with numerical IDs of the different models to be plotted. They must be located inside 'models' folder. Program will take the '_last' one", required=True)
    test_scene_parser.add_argument("--scene_to_load_folder", type=str, help="Folder name, located inside 'scene_configs', that contains the scene be loaded", required=True)
    test_scene_parser.add_argument("--robot_name", type=str, help="Name for the robot.", required=True)
    test_scene_parser.add_argument("--iters_per_model", type=int, help="Number of iterations for testing each model.",required=False, default=1)
    test_scene_parser.add_argument("--scene_path", type=str, help="Path to the CoppeliaSim scene file.", required=False)
    test_scene_parser.add_argument("--dis_parallel_mode", action="store_true", help="Disables the parallel training or testing.", required=False)
    test_scene_parser.add_argument("--no_gui", action="store_true", help="Disables Coppelia GUI, it will just show the terminal", required=False)
    test_scene_parser.add_argument("--params_file", type=str, help="Path to the configuration file.",required=False)
    test_scene_parser.add_argument("--timestamp", type=str, help="Timestamp provided externally (e.g., from GUI).", required=False)
    test_scene_parser.add_argument("--verbose", type=int, help="Enable debugging through info logs using the terminal. 0: no logs at all. \
                             1: just a progress bar and save warnings. 2: just a progress bar and save everything. 3: all logs shown and saved for debugging. Other: just terminal, logs are not saved", default=0, required=False)


    auto_training_parser = subparsers.add_parser("auto_training", help="Auto training of several models using different parameters pre-configured by using different configuration files.")
    auto_training_parser.add_argument("--session_name", type=str, help="Name for the session's folder.", required=True)
    auto_training_parser.add_argument("--robot_name", type=str, help="Name for the robot.", required=True)
    auto_training_parser.add_argument("--dis_parallel_mode", action="store_true", help="True if the user wants to disable the parallel execution and run the different trainings sequentially.", default=False)
    auto_training_parser.add_argument("--max_workers", type=int, help="Number of parallel processes if '--parallel_mode' flag is activated", default=3)
    auto_training_parser.add_argument("--timestamp", type=str, help="Timestamp provided externally (e.g., from GUI).", required=False)
    auto_training_parser.add_argument("--verbose", type=int, help="Enable debugging through info logs using the terminal. 0: no logs at all. \
                             1: just a progress bar and save warnings. 2: just a progress bar and save everything. 3: all logs shown and saved for debugging. Other: just terminal, logs are not saved", default=0, required=False)

    auto_testing_parser = subparsers.add_parser("auto_testing", help="Auto testing of several models, saving the results of the comparision.")
    auto_testing_parser.add_argument("--session_name", type=str, help="Name for the testing session.", required=True)
    auto_testing_parser.add_argument("--robot_name", type=str, help="Name for the robot.", required=True)
    auto_testing_parser.add_argument("--model_ids", type=int, nargs='+', help="List with numerical IDs of the different models to be tested.", required=True)
    auto_testing_parser.add_argument("--iterations", type=int, help="Number of iterations for the test.", default=200, required=True)
    auto_testing_parser.add_argument("--dis_parallel_mode", action="store_true", help="True if the user wants to disable the parallel execution and run the different trainings sequentially.", default=False)
    auto_testing_parser.add_argument("--max_workers", type=int, help="Number of parallel processes if '--parallel_mode' flag is activated", default=3)
    auto_testing_parser.add_argument("--verbose", type=int, help="Enable debugging through info logs using the terminal. 0: no logs at all. \
                             1: just a progress bar and save warnings. 2: just a progress bar and save everything. 3: all logs shown and saved for debugging. Other: just terminal, logs are not saved", default=0, required=False)

    sampling_at_parser = subparsers.add_parser("sat_training", help="Auto training of several models modifying just the fixed action time from an unique configuration file.")
    sampling_at_parser.add_argument("--session_name", type=str, help="Name for the session's folder.", required=True)
    sampling_at_parser.add_argument("--robot_name", type=str, help="Name for the robot.", required=True)
    sampling_at_parser.add_argument("--base_params_file", type=str, help="Path to the base parameters file to modify.", required=True)
    sampling_at_parser.add_argument("--dis_parallel_mode", action="store_true", help="True if the user wants to disable the parallel execution and run the different trainings sequentially.", default=False)
    sampling_at_parser.add_argument("--max_workers", type=int, help="Number of parallel processes if '--parallel_mode' flag is activated", default=3)
    sampling_at_parser.add_argument("--start_value", type=float, help="Starting value for fixed_actime.", default=0.06)
    sampling_at_parser.add_argument("--end_value", type=float, help="Ending value for fixed_actime.", default=2.1)
    sampling_at_parser.add_argument("--increment", type=float, help="Increment value for fixed_actime.", default=0.01)
    sampling_at_parser.add_argument("--verbose", type=int, help="Enable debugging through info logs using the terminal. 0: no logs at all. \
                             1: just a progress bar and save warnings. 2: just a progress bar and save everything. 3: all logs shown and saved for debugging. Other: just terminal, logs are not saved", default=0, required=False)



    retrain_parser = subparsers.add_parser("retrain", help="Retrain a pretrained RL algorithm for robot movement in CoppeliaSim.")
    retrain_parser.add_argument("--model_name", type=str, help="Name of the trained model is required (it must be located under 'models' folder)", required=True)
    retrain_parser.add_argument("--retrain_steps", type=int, help="Number of steps for the retraining. Default = 50.000",required=False, default=50000)
    retrain_parser.add_argument("--scene_path", type=str, help="Path to the CoppeliaSim scene file.", required=False)
    retrain_parser.add_argument("--dis_parallel_mode", action="store_true", help="Disables the parallel training or testing.", required=False)
    retrain_parser.add_argument("--no_gui", action="store_true", help="Disables Coppelia GUI, it will just show the terminal", required=False)
    retrain_parser.add_argument("--params_file", type=str, help="Path to the configuration file. It's not recommended to use a different one from the one used for the previous training",required=False)
    retrain_parser.add_argument("--dis_save_notes", action="store_true", help="Flag to save some notes for the experiment.", default = False, required=False)
    retrain_parser.add_argument("--rl_side", action="store_true", help="Flag to just execute rl side code.", default = False, required=False)
    retrain_parser.add_argument("--agent_side", action="store_true", help="Flag to just execute agent side code.", default = False, required=False)
    retrain_parser.add_argument("--comms_port", type=int, help="Number of the port to start the communications with the RL Side.", required=False)
    retrain_parser.add_argument("--ip_address", type=str, help="IP address of the RL Side.", required=False)
    retrain_parser.add_argument("--timestamp", type=str, help="Timestamp provided externally (e.g., from GUI).", required=False)
    retrain_parser.add_argument("--set", dest="overrides", action="append", default=[], metavar="KEY=VALUE", help="Override of params.json parameters (use dot notation).")
    retrain_parser.add_argument("--verbose", type=int, help="Enable debugging through info logs using the terminal. 0: no logs at all. \
                             1: just a progress bar and save warnings. 2: just a progress bar and save everything. 3: all logs shown and saved for debugging. Other: just terminal, logs are not saved", default=0, required=False)

    # Custom RL experiment evaluation with multiple maps and preconfigured experiments (RL_1, RL_2, RL_3).
    test_custom_exp_parser = subparsers.add_parser("test_custom_exp", help="Evaluate trained RL policies with preconfigured experiments (RL_1, RL_2, RL_3) on multiple maps.")
    test_custom_exp_parser.add_argument("--model_name", type=str, required=True, help="Name of the trained model (must be inside 'models' folder).")
    test_custom_exp_parser.add_argument("--robot_name", type=str, required=False, help="Name for the robot. Default will be resolved from params.")
    test_custom_exp_parser.add_argument("--experiments", type=str, nargs="+", default=None, help="List of experiments to run (e.g. RL_1 RL_2 RL_3). If omitted all three are run.")
    test_custom_exp_parser.add_argument("--maps", type=str, nargs="+", default=None, help="List of map file names inside 'custom_maps/' (e.g. map1.pgm map2.pgm). Interactive selection if omitted.")
    test_custom_exp_parser.add_argument("--episodes", type=int, default=500, help="Number of episodes per (experiment × map) combination. Default: 500.")
    test_custom_exp_parser.add_argument("--scene_path", type=str, required=False, help="Path to the CoppeliaSim scene file.")
    test_custom_exp_parser.add_argument("--save_traj", action="store_true", default=False, help="Save trajectory CSV for every episode.")
    test_custom_exp_parser.add_argument("--no_gui", action="store_true", help="Run CoppeliaSim without GUI.")
    test_custom_exp_parser.add_argument("--dis_parallel_mode", action="store_true", default=False, help="Disable parallel mode.")
    test_custom_exp_parser.add_argument("--agent_side", action="store_true", default=False, help="Only start agent side (advanced).")
    test_custom_exp_parser.add_argument("--rl_side", action="store_true", default=False, help="Only start RL side (advanced).")
    test_custom_exp_parser.add_argument("--params_file", type=str, required=False, help="Path to the configuration JSON file.")
    test_custom_exp_parser.add_argument("--comms_port", type=int, required=False, help="Port for RL ↔ Agent communication.")
    test_custom_exp_parser.add_argument("--ip_address", type=str, required=False, help="IP address of the RL side.")
    test_custom_exp_parser.add_argument("--timestamp", type=str, required=False, help="Timestamp provided externally (e.g., from GUI).")
    test_custom_exp_parser.add_argument("--set", dest="overrides", action="append", default=[], metavar="KEY=VALUE", help="Override params.json parameters (dot notation).")
    test_custom_exp_parser.add_argument("--verbose", type=int, default=0, choices=[0,1,2,3], help="Verbosity level. 0=quiet, 1=progress+warnings, 2=progress+all, 3=debug.")

    run_session_parser = subparsers.add_parser("run_session", help="RL Coppelia Session Manager: run multiple training and testing experiments in parallel.")
    run_session_parser.add_argument("--session_name", type=str, default=None, 
                        help="Name of the session folder. Defaults to SessionXX")
    run_session_parser.add_argument("--verbose", type=int, help="Enable debugging through info logs using the terminal. 0: no logs at all. \
                             1: just a progress bar and save warnings. 2: just a progress bar and save everything. 3: all logs shown and saved for debugging. Other: just terminal, logs are not saved", default=-1, required=False)
    run_session_parser.add_argument("--max_workers", type=int, help="Number of parallel processes allowed", default=3)


    args = parser.parse_args(argv)  # Parse CLI arguments (from sys.argv or passed manually)

    # Print UnCORE logo
    utils.print_uncore_logo()

    if args.command == "train":
        from rl_coppelia import train
        train.main(args)
    elif args.command == "test":
        from rl_coppelia import test
        test.main(args)
    elif args.command == "auto_training":
        from rl_coppelia import auto_training
        auto_training.main(args)
    elif args.command == "sat_training":
        from rl_coppelia import sat_training
        sat_training.main(args)
    elif args.command == "save":
        from rl_coppelia import save
        save.main(args)
    elif args.command == "tf_start":
        from rl_coppelia import tf_start
        tf_start.main(args)
    elif args.command == "auto_testing":
        from rl_coppelia import auto_testing
        auto_testing.main(args)
    elif args.command == "plot":
        from rl_coppelia import plot
        plot.main(args)
    elif args.command == "retrain":
        from rl_coppelia import retrain
        retrain.main(args)
    elif args.command == "test_scene":
        from rl_coppelia import test_scene
        test_scene.main(args)
    elif args.command == "create_robot":
        from rl_coppelia import create_robot
        create_robot.main()
    elif args.command == "test_path":
        from rl_coppelia import test_path
        test_path.main(args)
    elif args.command == "test_map":
        from rl_coppelia import test_map
        test_map.main(args)
    elif args.command == "run_session":
        from rl_coppelia import run_session
        run_session.main(args)
    elif args.command == "test_custom_exp":
        from rl_coppelia import test_custom_exp
        test_custom_exp.main(args)
        
    else:
        parser.print_help() # Show help if no command provided

if __name__ == "__main__":
    main()