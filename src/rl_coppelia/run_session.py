import os
import json
import re
import subprocess
import glob
import time
import logging
import pandas as pd
from concurrent.futures import ThreadPoolExecutor
from typing import List, Dict, Optional, Tuple, Tuple
from common import utils
import threading


class SessionManager:
    """
    Manages the lifecycle of an RL session: configuration, execution, and data aggregation.
    """

    def __init__(self, args):
        """
        Initializes the SessionManager.
        Args:
            args: Command-line arguments parsed by argparse.
        """
        self.args = args
        self.base_path = self._get_base_path()
        self.sessions_dir = os.path.join(self.base_path, "sessions")
        self.session_name = self._resolve_session_name(self.args.session_name)
        self.session_path = os.path.join(self.sessions_dir, self.session_name)
        self.config_path = os.path.join(self.session_path, "session_config.json")
        self.log_path = os.path.join(self.session_path, "session.log")
        self.launch_lock = threading.Lock() # semaphore for avoiding concurrent launches

        # Create session directory immediately
        os.makedirs(self.session_path, exist_ok=True)

        # Add file handler to logger to save logs inside session folder
        file_handler = logging.FileHandler(self.log_path)
        file_handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
        logging.getLogger().addHandler(file_handler)

    def _get_base_path(self) -> str:
        """Return project base path (2 levels up from this file)."""
        return os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    

    def _resolve_session_name(self, name: str) -> str:
        """
        Determines the session name. If None, auto-increments SessionXX.

        Args:
            name: The user-provided session name or None.

        Returns:
            The resolved unique session name.
        """
        if name:
            return name
        
        # Auto-increment logic
        if not os.path.exists(self.sessions_dir):
            return "Session01"
        
        existing = glob.glob(os.path.join(self.sessions_dir, "Session*"))
        max_id = 0
        for path in existing:
            folder_name = os.path.basename(path)
            if folder_name.startswith("Session") and folder_name[7:].isdigit():
                try:
                    num = int(folder_name[7:])
                    if num > max_id:
                        max_id = num
                except ValueError:
                    continue
        
        return f"Session{max_id + 1:02d}"

    def get_available_robots(self) -> List[str]:
        """
        Scans the robots directory to find available robot names.

        Returns:
            List of robot names found in the directory.
        """
        robots_dir = os.path.join(self.base_path, "robots")
        if not os.path.exists(robots_dir):
            return []
        return [d for d in os.listdir(robots_dir) if os.path.isdir(os.path.join(robots_dir, d))]

    def wizard(self):
        """
        Interactively guides the user to configure the session experiments.
        """
        print(f"\n--- CONFIGURING SESSION: {self.session_name} ---")
        print(f"Session path: {self.session_path}")
        
        experiments = []
        

        while True:
            print("\n" + "="*40)
            if not experiments:
                ans = input("Do you want to set a new training? (y/n): ").lower().strip()
            else:
                ans = input("Do you want to set another training? (y/n): ").lower().strip()
            
            if ans != 'y':
                break
            
            # --- Training Config ---
            print(f"\nRobot selection:")
            robots_dir = os.path.join(self.base_path, "robots")
            _, robot_name = utils.terminal_list_items(robots_dir, valid_extensions=None, selection_is_mandatory=True)

            train_args = input("Enter extra training arguments (e.g., --verbose 1 --no_gui): ").strip()

            current_exp = {
                "robot_name": robot_name,
                "train_args": train_args,
                "tests": []
            }
            print(f"\nCurrent training config: Robot='{robot_name}', Extra Train Args='{train_args}'")

            # --- Test Config Loop ---
            while True:
                test_ans = input(f"Do you want to set a test for '{robot_name}'? (y/n): ").lower().strip()
                if test_ans != 'y':
                    break
                
                scene_path = input("Enter scene path (e.g., burgerBot_scene1.ttt): ").strip()
                test_args = input("Enter test arguments (e.g., --iterations 400): ").strip()
                
                current_exp["tests"].append({
                    "scene_path": scene_path,
                    "args": test_args
                })
            
            experiments.append(current_exp)

            print(f"\nAdded experiment: {current_exp}")

        # Save Configuration
        with open(self.config_path, 'w') as f:
            json.dump(experiments, f, indent=4)
        
        logging.info(f"Configuration saved to {self.config_path}")
        return experiments

    def run(self):
        """
        Main execution flow: Load config, Execute Pipeline, Aggregate Results.
        """
        # 1. Load or Create Configuration
        if os.path.exists(self.config_path):
            logging.info("Loading existing configuration...")
            with open(self.config_path, 'r') as f:
                experiments = json.load(f)
        else:
            experiments = self.wizard()
        
        if not experiments:
            logging.warning("No experiments configured. Exiting.")
            return

        # 2. Execute Scheduler
        logging.info(f"Starting execution of {len(experiments)} experiments with max 3 workers.")
        
        # We need to track generated models to know where to look for CSVs later
        # Since threads are independent, we'll scan the filesystem after execution
        # based on timestamps or we can return data from threads.
        # For simplicity/robustness, we'll scan based on logic in the aggregation step.
        
        with ThreadPoolExecutor(max_workers=3) as executor:
            # We map the run_pipeline function to each experiment dict
            executor.map(self._run_single_pipeline, experiments)
        
        logging.info("All experiments finished.")

        # 3. Aggregate CSVs
        self.aggregate_results(experiments)


    def _run_single_pipeline(self, config: Dict):
        """
        Internal worker function to run one full train->test cycle.

        Args:
            config: The dictionary containing 'robot_name', 'train_args', and 'tests'.
        """
        # Get data from config dict
        robot_name = config['robot_name']
        train_args = config.get('train_args', "")
        tests = config.get('tests', [])
        prefix = f"[{robot_name}]"

        # Init a new key for saving model folder
        config['generated_model_folder'] = None
        
        # --- TRAIN ---
        cmd_train = f"uncore_rl train --robot_name {robot_name} {train_args}"
        success, train_log = self._exec_cmd(cmd_train, f"{prefix}[TRAIN]")

        if not success:
            return

        # --- IDENTIFY MODEL ---
        target_pattern = rf"({re.escape(robot_name)}_model_\d+)"
        model_folder = self._extract_info_from_log(train_log, target_pattern)

        if not model_folder:
            logging.error(f"{prefix} CRITICAL: Could not identify model name from training log. Output:\n{train_log[:200]}...")
            return
    
        config['generated_model_folder'] = model_folder
        
        # Determine the model path argument (usually folder/folder_last)
        # Assuming folder name is "robot_model_X", the saved model is usually inside.
        model_arg = f"{model_folder}/{model_folder}_last"

        # --- TESTS ---
        for i, test in enumerate(tests):
            # Get data from test dict
            scene = test['scene_path']
            t_args = test.get('args', "")

            # Add key for identifying the test experiment
            test['exp_id'] = None

            # Build and run test command
            cmd_test = f"uncore_rl test --model_name {model_arg} --scene_path {scene} {t_args}"
            success_test, test_log = self._exec_cmd(cmd_test, f"{prefix}[TEST-{i+1}]")

            if not success_test:
                return

            # --- IDENTIFY MODEL ---
            exp_id = self._extract_info_from_log(test_log, target_pattern)
            if not exp_id:
                logging.error(f"{prefix} CRITICAL: Could not identify experiment id from testing log. Output:\n{test_log[:200]}...")
                return

            test['exp_id'] = exp_id


    def _exec_cmd(self, command: str, log_tag: str) -> Tuple[bool, str]:
        """
        Helper to run shell command. Returns (Success, Stdout_Text).
        """

        with self.launch_lock:
            logging.info(f"{log_tag} Waiting 10s to ensure port reservation safety...")
            time.sleep(10) 
            logging.info(f"{log_tag} Launching process now...")


        logging.info(f"{log_tag} START: {command}")
        try:
            res = subprocess.run(command, shell=True, check=False, 
                                 stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
            
            # Return both success status and output text
            output_text = res.stdout
            
            if res.returncode == 0:
                logging.info(f"{log_tag} DONE.")
                return True, output_text
            else:
                logging.error(f"{log_tag} FAILED ({res.returncode}):\n{res.stderr}")
                return False, output_text
        except Exception as e:
            logging.error(f"{log_tag} ERROR: {e}")
            return False, ""
        
        
    def _find_latest_model_folder(self, robot_name: str) -> Optional[str]:
        """Finds the newest folder starting with robot_name_model_."""
        # This assumes models are saved in the current working directory (base_path)
        pattern = os.path.join(self.base_path, f"{robot_name}_model_*")
        candidates = [f for f in glob.glob(pattern) if os.path.isdir(f)]
        if not candidates: return None
        return os.path.basename(max(candidates, key=os.path.getctime))
    

    def _extract_info_from_log(self, log_text: str, pattern_text: str) -> Optional[str]:
        """
        Searches a pattern in the log text and returns the first match.
        """
        matches = re.findall(pattern_text, log_text)
        
        if matches:
                # Return the last match found
                unique_matches = list(set(matches))
            
                if len(unique_matches) == 1:
                    return unique_matches[0]
                else:
                    return matches[-1]
        
        return None


    def aggregate_results(self, experiments: List[Dict]):
        """
        Scans generated model folders for CSV summaries and merges them.

        This function assumes that 'uncore_rl test' generates a summary CSV 
        inside the model folder or in a known location. 
        We will look for ALL .csv files in the latest model folders corresponding 
        to the experiments run.
        """
        logging.info("Aggregating results into global CSV...")
        
        global_csv_path = os.path.join(self.session_path, "global_session_summary.csv")
        all_data = []

        for exp in experiments:
            robot_name = exp['robot_name']
            
            # Find the folder that was just created/used
            model_folder_name = exp.get('generated_model_folder')
            if not model_folder_name:
                logging.warning(f"Skipping aggregation for {robot_name} (No model generated or training failed).")
                continue
                
            test_folder = os.path.join(self.base_path, robot_name, "testing_metrics")
            
            # Search the summary test csv for the current experiment
            test_file = os.path.join(test_folder, "test_records.csv")

            # Get the test experiment ID for getting the right rows from the CSV    
            tests = exp.get('tests', [])

            for i, test in enumerate(tests):
                test_id = test.get('exp_id')
                logging.info(f"Looking for data for {robot_name} test #{i+1} with experiment ID: {test_id}")
            
                if not test_id:
                    logging.warning(f"Skipping test aggregation for {robot_name} test #{i+1} (No experiment ID found).")
                    continue
                
                # Read the CSV and filter rows for this test_id. First columns of summary csv is 'Exp_id'
                df = pd.read_csv(test_file)
                df_filtered = df[df['Exp_id'] == test_id]

                # df_filtered will have just one row, and we need to add one last column to identify the source file (test_file)
                df_filtered['Source_File'] = os.path.basename(test_file)

                # Append to all_data
                all_data.append(df_filtered)
                logging.info(f"Added data for {robot_name} test #{i+1} from {test_file}.")
        
        # Generate the session csv with all_data
        if not all_data:
            logging.warning("No data found to aggregate into global CSV.")
            return
        final_df = pd.concat(all_data, ignore_index=True)
        final_df.to_csv(global_csv_path, index=False)
        logging.info(f"SUCCESS: Global summary saved at {global_csv_path}")
             

def main(args):
    session_manager = SessionManager(args)
    session_manager.run()
    
if __name__ == "__main__":
    main()