# CuttleBot Delayed Gratification Experiment

This repository contains simulation and TurtleBot4 code for a delayed gratification experiment inspired by cuttlefish prey-choice behavior. The project models a robot choosing between two chambers that contain different reward options: an immediately available lower-value option and a delayed higher-value option.

The codebase is organized around the same experiment at three levels:

1. Computational simulations of the reinforcement learning task.
2. A Gazebo/TurtleBot4 simulation with ROS 2 nodes for decision making, navigation, vision, and result logging.
3. Physical TurtleBot4 scripts for running versions of the task on a real robot.

## Experimental Design

The task asks whether an agent will choose a better reward when access to that reward is delayed. In this project, the agent is a Q-learning policy controlling either a simulated decision maker or a TurtleBot4.

Each trial presents one of four possible states:

| State     | Left chamber        | Right chamber       | Trial type   | Better choice |
| --------- | ------------------- | ------------------- | ------------ | ------------- |
| `EXPM_LR` | Live shrimp         | Dead shrimp         | Experimental | Left          |
| `EXPM_RL` | Dead shrimp         | Live shrimp         | Experimental | Right         |
| `CTRL_LR` | Unobtainable shrimp | Dead shrimp         | Control      | Right         |
| `CTRL_RL` | Dead shrimp         | Unobtainable shrimp | Control      | Left          |

The reward values are:

| Option              | Reward |
| ------------------- | ------ |
| Live shrimp         | `5.0`  |
| Dead shrimp         | `1.0`  |
| Unobtainable shrimp | `0.5`  |

### Learning Rule

The agent stores a Q-table with four states and two actions:

```text
Q[state][action]
```

Actions are:

| Action  | Meaning                  |
| ------- | ------------------------ |
| `LEFT`  | Choose the left chamber  |
| `RIGHT` | Choose the right chamber |

The Q-table is updated after each trial:

```text
Q(s, a) = Q(s, a) + alpha * (reward - Q(s, a))
```

The default learning parameters are:

| Parameter | Value  | Meaning                                                                       |
| --------- | ------ | ----------------------------------------------------------------------------- |
| `ALPHA`   | `0.10` | Learning rate                                                                 |
| `BETA`    | `1.0`  | Softmax action-selection inverse temperature                                  |
| `GAMMA`   | `0.99` | Included in some scripts, but the current update is immediate-reward learning |

Actions are chosen using softmax over the current state's Q-values.

### Delay Manipulation

Delay affects only the live-shrimp option. Before action selection, the Q-value for the live option is multiplied by a delay-dependent probability of waiting:

```python
prob_wait(delay_seconds)
```

This function is based on a cumulative normal curve with:

| Parameter          | Value        |
| ------------------ | ------------ |
| Mean delay         | `70` seconds |
| Standard deviation | `20` seconds |
| Weight             | `2`          |

As delay increases, the effective value of the live option decreases, making the agent more likely to choose the immediately available option.

### Phases

Most scripts divide the experiment into two phases:

1. **Training phase**
   - Run trials with `p_wait = 1.0`.
   - The agent learns which side is better for each state without delay discounting.
   - The standard training length is `100` trials.

2. **Delay testing phase**
   - Test trained behavior across delays:

     ```text
     10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 130 seconds
     ```

   - Experimental trials measure whether the agent still chooses the live shrimp.
   - Control trials measure whether the agent chooses the obtainable dead shrimp instead of the unobtainable option.

## Repository Structure

```text
.
├── computational simulation/
├── cuttlebot_nodes/
├── cuttlefish_sim/
├── physical robot embodiment/
├── scripts/
├── maps/
├── docs/
├── install.sh
├── build.sh
├── kill_sim.sh
└── setup.bash
```

### `computational simulation/`

Standalone Python simulations of the delayed gratification model.

| File                                         | Purpose                                                                                                 |
| -------------------------------------------- | ------------------------------------------------------------------------------------------------------- |
| `delayed_gratification_simulation.py`        | Single-run Q-learning simulation. Plots training learning curves and delay performance.                 |
| `delayed_gratification_30runs_simulation.py` | Runs the same simulation 30 times and plots mean performance with standard-error bars.                  |
| `compare_qtable_turtlebot.py`                | Compares simulated Q-tables against TurtleBot4 Q-values and saves `qtable_simulation_vs_turtlebot.png`. |

Run a simulation with:

```bash
python3 "computational simulation/delayed_gratification_30runs_simulation.py"
```

### `cuttlefish_sim/sim_gazebo/`

Gazebo-specific simulation files and the reusable delayed gratification library.

| Path                            | Purpose                                                                                                                     |
| ------------------------------- | --------------------------------------------------------------------------------------------------------------------------- |
| `delayed_gratification.py`      | Shared Q-learning library used by ROS nodes. Defines `QTable`, `prob_wait`, `run_trial`, and correctness checks.            |
| `worlds/cuttlebot_world.sdf`    | Custom Gazebo world for the two-chamber task.                                                                               |
| `launch/cuttlebot_gz.launch.py` | Custom TurtleBot4 Gazebo launch file that forwards localization, Nav2, map, world, and simulation-time arguments correctly. |
| `config/nav2.yaml`              | Custom Nav2 parameters for the task environment.                                                                            |
| `config/localization.yaml`      | Custom localization/AMCL parameters.                                                                                        |

### `cuttlebot_nodes/`

ROS 2 Python package containing the main robot-in-simulation experiment nodes.

| Node                 | File                                    | Purpose                                                                                                                              |
| -------------------- | --------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------ |
| `brain_node`         | `cuttlebot_nodes/brain_node.py`         | Q-learning decision maker. Receives trial commands, selects an action, returns state/action/reward, and updates the Q-table.         |
| `state_manager`      | `cuttlebot_nodes/state_manager.py`      | Orchestrates the experiment. Runs training/testing, sends trial commands, navigates the robot, confirms rewards, and writes results. |
| `vision_node`        | `cuttlebot_nodes/vision_node.py`        | Detects red/yellow targets using HSV color segmentation and can scan or seek a target color.                                         |
| `location_awareness` | `cuttlebot_nodes/location_awareness.py` | Tracks robot pose from AMCL/odometry and publishes coarse zones such as `CENTER`, `GAP`, `LEFT_CHAMBER`, and `RIGHT_CHAMBER`.        |
| `plot_results`       | `cuttlebot_nodes/plot_results.py`       | Reads saved experiment results and creates training and delay-performance plots.                                                     |
| `nav_to_pose`        | `cuttlebot_nodes/nav_to_pose.py`        | Small navigation test script.                                                                                                        |

The main ROS topic flow is:

```text
state_manager  -> /carl/trial_cmd       -> brain_node
brain_node     -> /carl/trial_result    -> state_manager
state_manager  -> /carl/reward_confirm  -> brain_node
vision_node    -> /carl/color_detection -> state_manager
location_node  -> /carl/zone            -> state_manager
state_manager  -> /carl/vision_cmd      -> vision_node
vision_node    -> /carl/vision_status   -> state_manager
```

### `scripts/`

Convenience scripts for launching and testing.

| File                              | Purpose                                                                                                                |
| --------------------------------- | ---------------------------------------------------------------------------------------------------------------------- |
| `sim.py`                          | Main simulation launcher. Starts Gazebo, Nav2, RViz, and the experiment nodes. Supports test, train, and teleop modes. |
| `delayed_turtlebot.py`            | Earlier integrated TurtleBot experiment script with navigation and Q-learning in one node. (older version, do not use)                            |
| `automation.py`, `automation2.py` | Older helpers for launching physical TurtleBot navigation commands in separate terminals.                              |

### `physical robot embodiment/`

Physical TurtleBot4 experiment prototypes.

| File                      | Purpose                                                                                                                                                       |
| ------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `delayed_turtlebot_v1.py` | Early physical robot version using fixed chamber coordinates and BasicNavigator.                                                                              |
| `delayed_working.py`      | Physical robot version using a pretrained Q-table and TurtleBot4 navigation.                                                                                  |
| `delayed_with_camera.py`  | More complete physical robot version with camera-based red/yellow/purple detection, operator delay timing, door-removal detection, and bumper-based stopping. |

The physical scripts are useful for understanding the robot embodiment, but the current modular ROS/Gazebo workflow is centered on `cuttlebot_nodes/` and `scripts/sim.py`.

### `maps/`

Map files used for localization in the custom Gazebo world:

| File                   | Purpose                           |
| ---------------------- | --------------------------------- |
| `cuttlebot_world.yaml` | Map metadata for Nav2/AMCL.       |
| `cuttlebot_world.pgm`  | Occupancy grid image for the map. |

### `docs/`

Notes and helper documentation, including TurtleBot4 setup notes.

## Setup

The install script targets **Ubuntu 24.04 LTS** with **ROS 2 Jazzy**.

```bash
chmod +x install.sh
./install.sh
```

The script installs or configures:

- ROS 2 Jazzy
- TurtleBot4 packages
- Nav2 and SLAM tools
- Gazebo Harmonic / TurtleBot4 simulation packages
- Webots
- Python dependencies: `numpy`, `scipy`, `matplotlib`, `opencv-python`
- A colcon workspace at `~/turtlebot4_ws`
- A symlink from this repository's `cuttlebot_nodes/` package into the workspace

After installation, source the environment:

```bash
source setup.bash
```

or open a new terminal if `.bashrc` was updated by the installer.

## Build

Rebuild the ROS workspace after changing node code:

```bash
./build.sh
```

For a faster rebuild of only this package:

```bash
./build.sh --this
```

## Running the Gazebo Experiment

Run the full Gazebo simulation and experiment nodes:

```bash
python3 scripts/sim.py
```

By default, this starts in test mode. In test mode, `state_manager` skips training and uses the pretrained Q-table inside `cuttlefish_sim/sim_gazebo/delayed_gratification.py`.

To run the training phase before testing:

```bash
python3 scripts/sim.py --train
```

To launch Gazebo for manual driving:

```bash
python3 scripts/sim.py --teleop
```

The launcher starts:

- Custom Gazebo world.
- TurtleBot4 simulation.
- Localization with the map in `maps/cuttlebot_world.yaml`.
- Nav2 using custom parameters.
- RViz.
- Experiment nodes:
  - `location_awareness`
  - `vision_node`
  - `brain_node`
  - `state_manager`

## Results and Plots

During a ROS/Gazebo run, `state_manager` saves results to:

```text
cuttlefish_sim/sim_gazebo/results.json
```

Generate plots from the saved results:

```bash
ros2 run cuttlebot_nodes plot_results
```

This writes plots to:

```text
cuttlefish_sim/sim_gazebo/plots/
```

Expected plots:

| Plot                    | Meaning                                                   |
| ----------------------- | --------------------------------------------------------- |
| `training_curves.png`   | Percent-correct learning curves by state during training. |
| `delay_performance.png` | Experimental vs. control performance across delay values. |

## Running Individual ROS Nodes

After building and sourcing the workspace, nodes can also be run manually:

```bash
ros2 run cuttlebot_nodes brain_node
ros2 run cuttlebot_nodes state_manager
ros2 run cuttlebot_nodes vision_node
ros2 run cuttlebot_nodes location_awareness
```

For simulation time, add:

```bash
--ros-args -p use_sim_time:=true
```

Example:

```bash
ros2 run cuttlebot_nodes brain_node --ros-args -p use_sim_time:=true
```

## Current Implementation Notes

- `brain_node` starts with a pretrained Q-table by default through `QTable(pretrained=True)`.
- In `state_manager`, `mode=test` skips the training phase and runs only delay testing.
- In `mode=train`, the system first runs `100` no-delay training trials.
- Testing uses `20` trials per delay in `state_manager.py`.
- The robot observes target colors while approaching the gap, infers left/right arrangement, asks the brain node for a decision, navigates to the selected chamber, and confirms reward based on whether navigation succeeded.
- The physical camera script `delayed_with_camera.py` uses red/yellow/purple color logic, where purple marks a removable door blocking the yellow/live chamber.

## Quick Command Reference

```bash
# Install system and workspace dependencies
./install.sh

# Source ROS/workspace environment
source setup.bash

# Rebuild
./build.sh --this

# Run Gazebo experiment in test mode
python3 scripts/sim.py

# Run Gazebo experiment with training first
python3 scripts/sim.py --train

# Run manual teleop mode
python3 scripts/sim.py --teleop

# Plot saved ROS experiment results
ros2 run cuttlebot_nodes plot_results

# Run 30 independent computational simulations
python3 "computational simulation/delayed_gratification_30runs_simulation.py"

# Compare simulation Q-values with TurtleBot Q-values
python3 "computational simulation/compare_qtable_turtlebot.py"
```

## Development Status

This repository contains both active code and earlier experimental prototypes. The cleanest path for new development is:

1. Use `computational simulation/` to validate learning behavior quickly.
2. Use `cuttlefish_sim/sim_gazebo/delayed_gratification.py` for shared task logic.
3. Use `cuttlebot_nodes/` for modular ROS 2 robot experiments.
4. Use `scripts/sim.py` as the main Gazebo launcher.
5. Refer to `physical robot embodiment/` when transferring the behavior to a real TurtleBot4.
