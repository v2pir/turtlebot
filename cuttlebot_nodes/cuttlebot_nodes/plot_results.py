"""
plots results from delayed gratification

reads results.json which is saved by state_manager node
displays
    1. training learning curves by state
    2. experimental vs control performance across delays

to run node: ros2 run cuttlebot_nodes plot_results
"""

import json
import os
import sys

import numpy as np
import matplotlib.pyplot as plt

# Add sim_gazebo to path for constants
# use realpath to resolve symlinks from colcon build
sys.path.insert(0, os.path.join(
    os.path.dirname(os.path.realpath(__file__)), '..', '..', '..', 'cuttlefish_sim', 'sim_gazebo'))
from delayed_gratification import (
    STATE_NAMES, STATES, DEAD_RWD, UNOBTAINABLE_RWD,
)

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
RESULTS_FILE = os.path.join(REPO_ROOT, 'cuttlefish_sim', 'sim_gazebo', 'results.json')
OUTPUT_DIR = os.path.join(REPO_ROOT, 'cuttlefish_sim', 'sim_gazebo', 'plots')


def load_results():
    if not os.path.exists(RESULTS_FILE):
        print(f'Results file not found: {RESULTS_FILE}')
        print('Run the experiment first (state_manager + brain_node).')
        sys.exit(1)

    with open(RESULTS_FILE) as f:
        return json.load(f)


def plot_training(results):
    """Plot per-state learning curves during training."""
    training = results['training']
    if not training:
        print('No training data to plot.')
        return

    state_trials = [[] for _ in range(STATES)]
    state_perf = [[] for _ in range(STATES)]
    state_total = np.zeros(STATES)
    state_hits = np.zeros(STATES)

    for trial in training:
        s = trial['state']
        state_total[s] += 1
        state_hits[s] += int(trial['correct'])
        state_trials[s].append(state_total[s])
        state_perf[s].append(100 * state_hits[s] / state_total[s])

    labels = [
        "EXPM_LR (Live Left)",
        "EXPM_RL (Live Right)",
        "CTRL_LR (Dead Right)",
        "CTRL_RL (Dead Left)",
    ]

    plt.figure(figsize=(10, 6))
    for s in range(STATES):
        if state_trials[s]:
            plt.plot(state_trials[s], state_perf[s], linewidth=2, label=labels[s])

    plt.xlabel("Training Trials (within each state)")
    plt.ylabel("Percent Correct (%)")
    plt.title("Training Learning Curves by State")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend()

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    path = os.path.join(OUTPUT_DIR, 'training_curves.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    print(f'Saved: {path}')
    plt.show()


def plot_testing(results):
    """Plot experimental vs control performance across delays."""
    testing = results['testing']
    delays = results['delays']

    exp_percent = []
    ctrl_percent = []

    for d in delays:
        trials = testing.get(str(d), [])
        exp_total = 0
        exp_correct = 0
        ctrl_total = 0
        ctrl_correct = 0

        for trial in trials:
            if trial['is_experimental']:
                exp_total += 1
                exp_correct += int(trial['reward'] > DEAD_RWD)
            else:
                ctrl_total += 1
                ctrl_correct += int(trial['reward'] > UNOBTAINABLE_RWD)

        exp_percent.append((100 * exp_correct / exp_total) if exp_total > 0 else 0)
        ctrl_percent.append((100 * ctrl_correct / ctrl_total) if ctrl_total > 0 else 0)

    plt.figure(figsize=(10, 6))
    plt.plot(delays, exp_percent, marker='o', linewidth=2, label="Experimental")
    plt.plot(delays, ctrl_percent, marker='s', linewidth=2, label="Control")
    plt.xlabel("Delay (seconds)")
    plt.ylabel("Percent Choosing Better Option (%)")
    plt.title("Delayed Gratification Performance vs. Delay")
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.legend()

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    path = os.path.join(OUTPUT_DIR, 'delay_performance.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    print(f'Saved: {path}')
    plt.show()


def main():
    results = load_results()
    plot_training(results)
    plot_testing(results)
    print('All plots generated.')


if __name__ == '__main__':
    main()
