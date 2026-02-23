# Subjective Time Maps

This repository accompanies the paper “Time-Aware Mobile Robot Navigation: Learning Subjective Time Maps (STM)”, published in *Computer Modeling in Engineering & Sciences* (Special Issue: *Environment Modeling for Applications of Mobile Robots*).

It provides a complete framework to train and test reinforcement learning (RL) policies for mobile robots in CoppeliaSim, and to visualize those policies as Subjective Time Maps (STM).

## Abstract

The basic operation of a mobile robot is navigating to some target, avoiding collisions and possibly minimizing other criteria. A diversity of methods have been developed since the past century, and the research is still active, but there is one aspect that is often neglected: the duration of the steps in which computational devices divide the navigation process. Usually it is set heuristically to a small, constant value for sampling observations frequently enough to ensure safety; however, each robot and environment have particularities that can make such a fixed timestep sub-optimal.

In this paper we explore the possibility of learning the best time for each step for a particular robot-environment interaction, and show that those subjective, variable timesteps constitute an important aspect of both motion safety and efficiency. The approach has an initial learning stage in an environment designed for providing the main situations the robot will face; through deep RL, a time-aware reactive navigation policy is found that optimizes collision avoidance, total navigation time and accurateness in reaching the target, yielding (motion, timestep-duration) commands. Due to the generalization capabilities of DRL, that policy induces Subjective Time Maps (STMs) for time-aware navigation in other scenarios. Thus, STMs become time-aware subjective models of the robot-world interaction that improve that particular robot navigation and also allow it to leverage time effort, e.g., longer timesteps can be used for other operations.

We also cope with uncertainties in both localization and decision-making; for that, we first learn the time-aware, motion policy ignoring localization uncertainty, consequently improving learning convergence and reducing computational cost, and then we integrate it with the resulting STMs through a particle filter (PF), by associating each particle (robot pose hypothesis) and its importance (weight) with the time-aware motion command provided by the STM for that hypothesis. The resulting time- and motion- aware particle cloud provides a full treatment of uncertainty with reduced complexity.

Diverse experiments with STMs show the advantages of this approach for both safe and efficient navigation in indoor structured environments. STMs integrated with PFs improve both aspects in comparison with STMs that do not use full uncertainty, and also have some advantages with respect to perfect localization.

## Video

https://github.com/user-attachments/assets/bb6ae7fb-586a-4219-a814-9a2805a8fbac




## 📁 Project Structure

The project is organized into several key directories:

- **`configs`**: JSON files with configuration parameters for the environment and the training/testing processes. By default, you will find a model file, which you can modify and make copies of to place in this same folder. Please do not delete the model file `params_default_file.json`. Also, every time you create a new robot using `create_robot`, a new default parameters file will be created for that new robot.
- **`dependencies`**: External libraries or modules not installable via pip. In particular, it includes the `rl_spin_decoupler` package (added as a submodule).
- **`src`**: Python source code. It contains:
    - **`common`**: 
        - **`coppelia_agents.py`**: Definition of CoppeliaAgent class for managing the interaction of the agent with CoppeliaSim simulation. If you want to add a new robot, just copy one of the current subclasses (e.g., BurgerBotAgent, TurtleBotAgent), and modify their key handles to adapt the robot to your scene. If you run `create_robot`, the child class will be created automatically.
        - **`coppelia_envs.py`**: Definition of CoppeliaEnv class for managing the interaction of the environments with CoppeliaSim simulator. Here you will find the step, reset, and calculate-reward functions. If you want to add a new robot, just copy one of the current subclasses (e.g., BurgerBotEnv, TurtleBotEnv). It's important to adapt their action and observation spaces to your needs. Again, this will be automatically done running the script for creating a new robot.
        - **`rl_coppelia_manager.py`**: Definition of the RLCoppeliaManager class, which is responsible for managing some of the initial processes of every core functionality of this project. It manages the creation of the environment, and the start/stop of the simulations.
        - **`utils.py`**: Utility scripts used across various processes. 
        - **`robot_generator.py`**: Helpers used by the interactive robot creator.
    - **`coppelia_scripts`**:
        - **`rl_script_copp.py`**: Not directly executable. Its variables are updated at the start of the experiment and its content is automatically copied into a script within the CoppeliaSim scene called 'Agent_Script'. This way you don’t need to edit the scene manually. This script is responsible for the communication between the Python RL agent and the simulation, as well as managing the reinforcement learning loop.
        - **`robot_script_copp.py`**: Similarly to the previous file, it is automatically copied into a CoppeliaSim scene script called `Robot_Script`. This script contains functions for controlling the robot and the scene.
    - **`plugins`**: `envs` and `agents` are lightweight plugin registries. Add `<robot>.py` modules that call `register_env(...)` / `register_agent(...)` to plug any robot in cleanly. These files are automatically managed by the `create_robot` functionality.
    - **`misc`**: Analysis scripts, isolated tests, and helper tools.
    - **`gui`**: A PyQt5 GUI with tabs for Train, Test, Auto-Train, SAT-Train (action-time sweep), Test Scene, Plot, and Manage (import/export, cleanup). __Not fully tested yet__
    - **`rl_coppelia`**: Core functionalities of the project. It contains the core entry points for command-line usage:
        - **`cli.py`**: Handles command-line interface (CLI) functionality for all the possible processes of this project.
        - **`create_robot.py`**: Interactive script to help users creating the python and scene structure for a new robot.
        - **`train.py`**: Manages the model training process.
        - **`test.py`**: Runs model evaluation and logs performance metrics.
        - **`test_path.py`**: Evaluates the model on a path following task using a temporal map approach.
        - **`test_path_v2.py`**: Grid-based evaluation with fixed target.
        - **`custom_exp.py`**: Runs preconfigured evaluation scenarios (experiments) on custom maps.
        - **`test_scene.py`**: Tests a model in a specific scene configuration for a single iteration.
        - **`run_session.py`**: Manages the execution of multiple experiments (sessions) in parallel.
        - **`save.py`**: Saves the trained model and its associated data to a zip file inside the 'results' folder.
        - **`tf_start.py`**: Starts TensorFlow training logs and visualizations, handling TensorBoard. It automatically opens your browser using the right web address and port.
        - **`plot.py`**: Generates plots for visualizing different comparisons of training and testing metrics.
        - **`auto_training.py`**: Automates the training process, managing hyperparameter tuning. It is only necessary to have the different configurations to be tested (each in a json file of the type 'params_file') inside the subfolder robots/<robot_name>/auto_trainings/<session_name>.
        - **`sat_training.py`**: Automates the training process just changing one specific parameter, the action time. This makes possible to easily find the optimal action time for a specific setup. __Not tested yet__
        - **`auto_testing.py`**: Automates the testing of models. __Not tested yet__
        - **`retrain.py`**: It allows the user to continue the training of a pretrained model. __Not tested yet__
- **`scenes`**: It contains CoppeliaSim scene files (.ttt format) for each robot/task.
- **`robots`**: Stores the data related to each robot, including model and callback files, tensorboard logs and generated metrics. All the subfolders inside 'robots' folder will be automatically created whenever you called the 'train' functionality for a new robot for the first time. This will be the structure that would be created for a robot called burgerBot:
```
    ├── robots/
      └── burgerBot/
          ├── auto_trainings/
          ├── callbacks/
          ├── models/
          ├── parameters_used/
          ├── sat_trainings/
          ├── scene_configs/
          ├── script_logs/
          ├── testing_metrics/
          ├── tf_logs/
          └── training_metrics/
```
- **`ROS2_nodes`**: Nodes created for coordinating the navigation policy with the localization estimator, and for testing the real robot.
- **`sessions`**: Stores configuration and results for multi-experiment sessions managed by `run_session`.
- **`custom_maps`**: Store user-provided maps (.pgm, .yaml) and auxiliary files (.npy) for custom experiments.
- **`results`**: Each zip file generated with the `save` functionality will be saved here.


## 🎯 Overview

The primary goal of this project is to facilitate robot training or testing models in the CoppeliaSim simulator by interacting with the robot through Python scripts. The training or testing process will automatically start CoppeliaSim, load a specific scene, and initiate training using a provided robot model.

Any new robot name used in the `main.py` script will generate a new folder within the `robots` directory. This folder contains all the data generated during training or testing for that robot, including logs, models, and any additional outputs.

## ⚙️ Notes Before Running anything

- **Note 1**: You need to clone this project with its submodule, `rl_spin_decoupler`, which is a repository located at https://github.com/uncore-team/rl_spin_decoupler.git. For doing that, please clone the repository using the next command:

```bash
git clone --recurse-submodules git@github.com:uncore-team/rl_coppelia.git
```

If you already cloned it before reading these instructions (don't worry, I never read them either), please use the next commands:

```bash
git submodule init
git submodule update
```

If `dependencies/rl_spin_decoupler` folder is empty, you will have to synchronize it manually with the remote repo:

```bash
git submodule sync --recursive
```

Moreover, to be sure that the submodule is updated to its last version, run the next command:

```bash
git submodule update --init --recursive --progress --jobs 4
```

At this point, the repository and its submodule should be correctly cloned.

- **Note 2**: When running train/test functionalities, the content inside `robot_script_copp.py` and `rl_script_copp.py` will be copied into the `Robot_Script` and `Agent_Script` CoppeliaSim scripts, respectively. Just keep it in mind in case you need to make a backup of your scene. Same happens with `generate_obstacles.py` and `laser.lua` files, that will replace `ObstaclesGenerator` and `Laser_Script` files in the scene.

## 🧩 Installation

Your OS needs to be Ubuntu. You can try WSL also, but it has not been tested (we are working on it).
Before using this project, ensure that the following dependencies are installed:

- **Python 3.x** (preferably 3.8 or later). It has been tested with python 3.8 and 3.10.
- **CoppeliaSim**: The simulator must be installed and configured correctly for the project to work. The project has been tested with CoppeliaSim Edu v4.9.0 (rev. 6) 64bit.

It is recommended to install all the dependencies within a virtual environment. For that purpose, you will need to install venv package:
```bash
sudo apt update
sudo apt install -y python3-venv
```

To install the required Python libraries, you can directly use the `install.sh` file included in the root directory of the project. This will add the `rl_spin_decoupler` package and the base directory of the project to the path. It will also create a virtual environment and install all the required dependencies within it, so you don't need to manually create the environment if you don't want to.

```bash
source install.sh
```

If any error happened during the installation and the terminal was closed, please try again running next commands:
```bash
export KEEP_OPEN=1
source install.sh
```
This will allow you to debug the error and install any problematic package manually.

After that, your new virtual environment should be already activated, but if it's not, please run:
```bash
source ~/.venvs/uncore_rl_venv/bin/activate
```

If you skipped adding permanent PATH/PYTHONPATH exports during installation, you can also use:
```bash
source ~/.venvs/uncore_rl_venv/bin/activate_rl
```

Everything should be already installed, including the `uncore_rl` package. In fact, it is installed in editable mode (-e), so any changes you make in the code will be automatically reflected without needing to reinstall the package.


## 🚀 Usage

To start training a model for a robot, execute the train option of the `uncore_rl` package. You do not need to have CoppeliaSim opened, as a new instance of the program will be opened if you do not set the `dis_parallel_mode` to True. 

```bash
uncore_rl train --robot_name turtleBot --verbose 3
```
**Key arguments**:
- **`--robot_name`** (required): The name of the robot you wish to train or test for. This will create a folder for the robot in the `robots` directory. 
- **`--params_file`**: Name of the custom JSON with training/env parameters (defaults to configs/params_default_file_\<robot\>.json).
- **`--scene_path`**: Name of the CoppeliaSim scene used for the experiment (defaults to scenes/\<robot\>_scene.ttt)
- **`--no_gui`**: Flag to disable CoppeliaSim GUI. It's recommended for trainings.
- **`--dis_parallel_mode`**: Force to use an already opened CoppeliaSim scene. 
- **`--timestamp`**: External argument used by the GUI to name the terminal of the experiment.  
- **`--verbose`**: Level of verbosity. For your first steps with this repository, it's recommended to set it to 3, so you can check all the logs generated during the process.

For the training, as well as for creating the environment and for testing any model, there are some parameters needed which are assigned within the `configs/params_default_file_\<robot\>.json` file. The user can replicate this file and change the parameters values, and then use the argument `--params_file` indicating the absolute or relative path of the new json file (it's recommended to keep them in the same `configs` folder).

After every training, two models will be saved: the last and the one with the best training reward obtained. Moreover, you will have callbacks available every 10K steps (with the default parameters). In case that you want to test the last model saved for an specific experiment (for example, a model called `burgerBot_model_15`), the user can test it using the next command:

```bash
uncore_rl test --model_name burgerBot_model_15/burgerBot_model_15_last
```

If you need to check the possible key arguments for any functionality (e.g., `test`), please refer to the help option for more information.

```bash
uncore_rl test -h
```

### Decouple experiments

With the default configuration, all the experiments will run on the same machine, with RL and agent loops decoupled but running locally. If you want to use two different computers, one for the RL process and another for the agent simulation, then you will need to use the flags `--agent_side` and `--rl_side`, respectively. To indicate the RL IP address and port, you will need to run that process first, check the IP address and comms port that it has enabled, and then run the agent side in your second PC using `--comms_port` and `--ip_address`.


```bash
uncore_rl train --robot_name turtleBot --verbose 3 --agent_side --comms_port 49058 --ip_address "150.216.111.152"
```

### Run sessions

This functionality allows you to schedule, execute, and monitor full RL pipelines (Training → Testing) in an unsupervised manner. It is designed to maximize hardware usage by running experiments in parallel while preventing port conflicts.

```bash
uncore_rl run_session --session_name SessionDecember_03 --max_workers 3
```

**Key Features:**

- **Interactive Wizard:** If the session is new, a new folder will be created inside `/sessions` with the name provided, and a terminal wizard will guide you step-by-step to define the robots, training parameters, and test scenes.
- **Configuration Persistence:** A `session_config.json` file is automatically generated. You can resume or re-run the exact same session configuration later by simply pointing to the existing folder.
- **Safe Parallelism:** Executes multiple experiments concurrently to maximize hardware usage. It implements a **global lock** with a safety delay to prevent port collisions during the initialization of simulation instances.
- **Automated Aggregation:** Upon completion, the manager scans all generated model folders and merges test metrics into a single `global_session_summary.csv` for easy analysis.


### Create a new robot

To create a new robot structure, simply run:

```bash
uncore_rl create_robot
```
This interactive wizard will guide you through setting up the Python classes (Agent and Env) and the CoppeliaSim scene scripts ensuring all necessary files and folders are created correctly.



### Advanced Testing & Custom Maps

#### Temporal Map Evaluation (test_path)
Evaluates the model on a path following task using a temporal map approach.

```bash
uncore_rl test_path --model_name turtleBotFlex_model_6/turtleBotFlex_model_6_last --scene_path turtleBotPath_labv6_scene.ttt --verbose 3 --n_samples 100 --trials_per_sample 1 --n_extra_poses 4 --set params_scene.scene_x_dim=11 --set params_scene.scene_y_dim=11 --map_name ts2.pgm --robot_world_ori 0 --robot_target_ori 180
```

#### Grid-based Evaluation (test_map)
Tests the robot on a grid of positions with a fixed target, ideal for heatmap generation. It handles orientation augmentation and target noise. This method is the one used for the STMs visualization

```bash
uncore_rl test_map --model_name turtleBot_model_1/turtleBot_model_1_last --map_name ts2.pgm --n_extra_poses 2
```

#### Custom Experiments (custom_exp)
Runs canonical evaluation scenarios (RL_1: Short-range, RL_2: Long-range, RL_3: Near-obstacle) on custom maps.

```bash
uncore_rl custom_exp --model_name turtleBot_model_1/turtleBot_model_1_last --experiments RL_1 RL_3 --maps ts3.pgm
```

#### Plotting Results (Temporal Map)
After generating data with `test_path`, you can plot the obtained map:

```bash
uncore_rl plot --plot_types timestep_map --robot_name turtleBotFixedSpeed --model_ids 5 --csv_file_name turtleBotFixedSpeed_model_5_last_path_data_RW0.0_RT270.0_2025-11-21_13-37-24.csv --map_name ts1.png  --verbose 3 --plot_set stat="all" --plot_set complete_ranges=False
```


### Retrain a model

To continue training a previously saved model:

```bash
uncore_rl retrain --model_name myRobot_model_10/myRobot_model_10_last --retrain_steps 50000
```
This allows you to load the weights and optimizer state of a previous run and continue training for additional steps.


### 🖥️ Graphical Interface (GUI)

The project also includes a **PyQt5-based GUI** that allows users to interact with the same core functionalities through a graphical interface.  
It includes tabs for **Training**, **Testing**, **Auto-Training**, **SAT-Training**, **Test Scene**, **Plotting**, and **Manage** (import/export, cleanup).

To launch the GUI, run:

```bash
uncore_rl_gui
```

This interface allows you to configure experiments, run and monitor training/testing processes, and visualize logs in real time. ⚠️ **Note:** The GUI is still under active development, and some features may not be fully tested yet.
