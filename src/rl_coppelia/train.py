"""
Project: Robot Training and Testing RL Algorithms in CoppeliaSim
Author: Adrián Bañuls Arias
Version: 2.0
Date: 2025-03-25
License: GNU General Public License v3.0

Description:
    This script trains a reinforcement learning agent in a simulated CoppeliaSim environment using Stable-Baselines3.
    It supports configuration via parameter files, headless mode, logging, and can run in parallel with other training or 
    testing processes.
    Models and logs are saved automatically for later analysis and visualization.

Usage:
    uncore_rl train --robot_name <robot_name>
                      [--scene_path <path_to_scene_file>]
                      [--dis_parallel_mode]
                      [--no_gui]
                      [--params_file <path_to_config_file>]
                      [--timestamp <timestamp>]
                      [--verbose <0|1|2|3>]

Features:
    - Automatically starts a CoppeliaSim instance and prepares the training environment.
    - Supports parallel and sequential training modes.
    - Allows GUI-less execution for headless servers.
    - Loads training configuration from a user-defined parameters file.
    - Saves trained models, logs, and metrics for evaluation and visualization.
    - Detailed logging system with multiple verbosity levels and file output.
    - Automatically creates output directories for models, logs, and metrics.
    - Can be launched directly from the GUI or terminal with CLI arguments.
"""

import inspect
import logging
import os
import time
import stable_baselines3
from common import utils
from common.rl_coppelia_manager import RLCoppeliaManager
from stable_baselines3.common.callbacks import CheckpointCallback
import traceback
from stable_baselines3.common.off_policy_algorithm import OffPolicyAlgorithm


def save_final_model_and_artifacts(model, save_dir: str, base_name: str) -> None:
    """Save model, replay buffer (if any) and VecNormalize statistics.

    Args:
        model: Trained SB3 model instance.
        save_dir (str): Directory where artifacts will be stored.
        base_name (str): Base filename (without extension).
    """
    os.makedirs(save_dir, exist_ok=True)
    base_path = os.path.join(save_dir, base_name)

    # 1) Model
    model.save(base_path)
    logging.info(f"Model saved at: {base_path}.zip")

    # 2) Replay buffer
    try:
        if isinstance(model, OffPolicyAlgorithm):
            rb_path = base_path + "_replay_buffer.pkl"
            model.save_replay_buffer(rb_path)
            logging.info(f"Replay buffer saved at: {rb_path}")
        else:
            logging.info("On-policy algorithm, no replay buffer to save.")
    except Exception as exc:
        logging.warning(f"Could not save replay buffer: {exc}")

    # 3) VecNormalize stats
    try:
        vec_env = model.get_vec_normalize_env()
        if vec_env is not None:
            vn_path = base_path + "_vecnormalize.pkl"
            vec_env.save(vn_path)
            logging.info(f"VecNormalize stats saved at: {vn_path}")
        else:
            logging.info("No VecNormalize wrapper detected, nothing to save.")
    except Exception as exc:
        logging.warning(f"Could not save VecNormalize stats: {exc}")


def main(args):
    """
    Train the model using a custom environment.

    This function creates an environment, starts a CoppeliaSim instance and trains an agent 
    using that environment. Finally, it closes the opened simulation.
    """
    rl_copp = RLCoppeliaManager(args)

    ### Create the environment
    if not args.agent_side:
        rl_copp.create_env()

    if not args.rl_side:
        ### Start CoppeliaSim instance
        rl_copp.start_coppelia_sim("Train")

    if not args.agent_side:
        ### Start communication RL - CoppeliaSim
        rl_copp.start_communication()

        
        ### Train the model
        # Extract the needed paths for training
        models_path = rl_copp.paths["models"]
        callbacks_path = rl_copp.paths["callbacks"]
        train_log_path = rl_copp.paths["tf_logs"]
        training_metrics_path = rl_copp.paths["training_metrics"]
        parameters_used_path = rl_copp.paths["parameters_used"]
        train_log_file_path = os.path.join(train_log_path,f"{rl_copp.args.robot_name}_tflogs_{rl_copp.file_id}")

        logging.info(f"Training mode. Final trained model will be saved in {models_path}")
        logging.info(f"EXPERIMENT ID: {rl_copp.file_id}")

        # Get final model name
        to_save_model_path, model_name = utils.get_next_model_name(models_path, rl_copp.args.robot_name, rl_copp.file_id)

        # Set path for saving LATs during training
        rl_copp.env.envs[0].unwrapped.csv_lats_path = os.path.join(training_metrics_path, f"{model_name}_lats.csv")
        logging.info(f"LATs during training will be saved in: {rl_copp.env.envs[0].unwrapped.csv_lats_path}")

        # Callback function to save the model every x timesteps
        to_save_callbacks_path, _ = utils.get_next_model_name(callbacks_path, rl_copp.args.robot_name, rl_copp.file_id, callback_mode=True)
        checkpoint_callback = CheckpointCallback(save_freq=rl_copp.params_train['callback_frequency'], save_path=to_save_callbacks_path, name_prefix=rl_copp.args.robot_name)

        # Callback function for stopping the learning process if a specific key is pressed
        stop_callback = utils.StopTrainingOnKeypress(key="F") 

        # Callback for saving the best model based on the training reward every x timesteps
        eval_train_callback = utils.SaveOnBestTrainingRewardCallback(
            check_freq=1000,    # default: 1000
            save_path=to_save_model_path, 
            log_dir=rl_copp.log_monitor, 
            verbose=1)

        # Callback for logging custom metrics in Tensorboard
        metrics_callback = utils.CustomMetricsCallback(rl_copp, total_timesteps=rl_copp.params_train["total_timesteps"])

        # Get the training algorithm from the parameters file
        try:
            ModelClass = getattr(stable_baselines3, rl_copp.params_train["sb3_algorithm"])
        except:
            logging.error(f"Algorithm indicated in parameters file (json) is not valid. Parameter read: {rl_copp.params_train['sb3_algorithm']} ")
            raise

        # Configure the model
        init_params = inspect.signature(ModelClass.__init__).parameters
        if "n_steps" in init_params:
            model = ModelClass(
                policy = rl_copp.params_train["policy"], 
                env = rl_copp.env,
                n_steps = rl_copp.params_train["n_training_steps"], 
                verbose=True, 
                tensorboard_log=train_log_path
                )   

        else:
            model = ModelClass(
                policy = rl_copp.params_train["policy"], 
                env = rl_copp.env, 
                verbose=True, 
                tensorboard_log=train_log_path
                )   

      
        # Save the configuration that will actually be used for this training.
        # Always save from in-memory params so that any --set overrides are reflected
        # in the stored file rather than the unmodified source JSON.
        merged_params = {
            "params_scene": rl_copp.params_scene,
            "params_env":   rl_copp.params_env,
            "params_train": rl_copp.params_train,
            "params_test":  rl_copp.params_test,
        }
        params_file_save_path = utils.save_params_with_id(
            merged_params, parameters_used_path, rl_copp.args.params_file, rl_copp.file_id
        )

        logging.warning("Training will start in few seconds. If you want to end it at any time, press 'F' + Enter key, and then 'Y' + Enter key."
                        " It's not recommended to pause and then resume the training, as it will affect the current episode. That said, grab a cup of coffee and enjoy the process ;)")
        
        time.sleep(8)

        # Save a timestamp of the beggining of the training
        start_time = time.time()

        # Start the training
        
        try:
            if rl_copp.args.verbose ==0:
                model.learn(
                    total_timesteps=rl_copp.params_train['total_timesteps'],
                    callback=[checkpoint_callback, stop_callback, eval_train_callback, metrics_callback], 
                    tb_log_name=f"{rl_copp.args.robot_name}_tflogs"
                    )
            else:
                model.learn(
                    total_timesteps=rl_copp.params_train['total_timesteps'],
                    callback=[checkpoint_callback, stop_callback, eval_train_callback, metrics_callback], 
                    tb_log_name=f"{rl_copp.args.robot_name}_tflogs",
                    progress_bar = True
                    )

            
        except Exception as e:
            traceback.print_exc()
            logging.critical(f"There was an error during the learning process. Exception: {e}")

        # Save a timestamp of the ending of the training
        end_time = time.time()

        # Close csv lats file
        try:
            rl_copp.env.envs[0].unwrapped.csv_lats_file.close()
        except:
            pass

        # Save the final trained model
        logging.info(f"PATH TO SAVE MODEL: {to_save_model_path}")
        save_final_model_and_artifacts(
            model=model,
            save_dir=to_save_model_path,
            base_name=f"{model_name}_last",
        )

        # Parse metrics from tensorboard log and save them in a csv file. Also, we get the metrics of the last row of that csv file
        _, experiment_csv_path = utils.get_output_csv(model_name, training_metrics_path, "train")
        logging.info(f"PATH TO SAVE CSV TRAINIG: {experiment_csv_path}")
        try:
            _, last_metric_row = utils.parse_tensorboard_logs(train_log_file_path, output_csv=experiment_csv_path)
        except:
            last_metric_row = {}
            logging.error("There was an exception while trying to get data from tensorboard log.")

        # Name of the records csv to store the final values of the training experiment.
        records_csv_name = os.path.join(training_metrics_path,"train_records.csv")

        # Get time to converge using the data from the training csv
        try:
            convergence_time, _, _, _, _ = utils.get_convergence_point (experiment_csv_path, "Steps", convergence_threshold=0.02)
        except Exception as e:
            logging.error(f"No convergence time was found. Exception: {e}")
            convergence_time = 0.0

        # Construct the dictionary with some data to store in the records file
        if last_metric_row != {}:
            data_to_store ={
                "Algorithm" : rl_copp.params_train["sb3_algorithm"],
                "Policy" : rl_copp.params_train["policy"],
                "Action time (s)" : rl_copp.params_env["fixed_actime"],
                "Time to converge (h)" : convergence_time,
                **last_metric_row,
                "Params file" : os.path.basename(params_file_save_path)
            }

            # Update the train record.
            utils.update_records_file (records_csv_name, model_name, start_time, end_time, data_to_store)
        
        else:
            logging.error("No data was stored in the training records file because no metrics were obtained from tensorboard logs.")
        
        # Send a FINISH command to the agent
        rl_copp.env.envs[0].unwrapped._commstoagent.stepExpFinished()   # Unwrapped is needed so we can access the attributes of our wrapped env 

        logging.info("Training completed")

        rl_copp.remove_tmp_data()
        
    # if not args.rl_side:
    #     ### Close the CoppeliaSim instance
    #     rl_copp.stop_coppelia_sim()


if __name__ == "__main__":
    main()