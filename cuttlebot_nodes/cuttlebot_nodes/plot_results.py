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

# compute repo root from source file location (works with --symlink-install)
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.realpath(__file__))))
sys.path.insert(0, os.path.join(REPO_ROOT, 'cuttlefish_sim', 'sim_gazebo'))
from delayed_gratification import (
    STATES, DEAD_RWD,
)
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


def plot_testing(results):
    """Plot percent choosing live (patient) option across delays."""
    testing = results['testing']
    delays = results['delays']

    percent_correct = []

    for d in delays:
        trials = testing.get(str(d), [])
        total = len(trials)
        correct = sum(1 for trial in trials if trial['correct'])
        percent_correct.append((100 * correct / total) if total > 0 else 0)

    plt.figure(figsize=(10, 6))
    plt.plot(delays, percent_correct, marker='o', linewidth=2, label="Chose Live (Patient)")
    plt.xlabel("Delay (seconds)")
    plt.ylabel("Percent Choosing Live Shrimp (%)")
    plt.title("Delayed Gratification: Patience vs. Delay")
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.legend()

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    path = os.path.join(OUTPUT_DIR, 'delay_performance.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    print(f'Saved: {path}')


def main():
    results = load_results()
    plot_training(results)
    plot_testing(results)
    plt.show()
    print('All plots generated.')


if __name__ == '__main__':
    main()
