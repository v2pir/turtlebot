import random
from math import exp

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import norm

"""
Kary Zheng
Based on v2_delayed_gratification.py

Runs the same delayed gratification simulation 30 times so the plots can show
mean performance with standard-error error bars.
"""

ACTIONS = 2
LEFT = 0
RIGHT = 1

STATES = 4
EXPM_LR = 0  # Live Shrimp (left); Dead Shrimp (right)
EXPM_RL = 1  # Live Shrimp (right); Dead Shrimp (left)
CTRL_LR = 2  # Unobtainable Shrimp (left); Dead Shrimp (right)
CTRL_RL = 3  # Unobtainable Shrimp (right); Dead Shrimp (left)

LIVE_RWD = 5.0
DEAD_RWD = 1.0
UNOBTAINABLE_RWD = 0.5

# Learning parameters
ALPHA = 0.10
BETA = 1.0
GAMMA = 0.99

# Repeated-run settings
RUNS = 30
TRAINING_TRIALS = 100
DELAY_TRIALS = 100
DELAYS = np.array([10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 130])


def prob_wait(tim):
    """
    Calculate the probability to wait.
    """
    mean = 70
    std_dev = 20
    beta_weight = 2
    pw = 1 / np.exp(beta_weight * norm.cdf(tim, loc=mean, scale=std_dev))
    return pw


def action_select(q, beta):
    """
    Calculate the Softmax function to choose an action.
    """
    sum_softmax = 0
    sum_p = 0

    for i in range(len(q)):
        sum_softmax += exp(beta * q[i])

    r = random.random()

    for i in range(len(q)):
        p = exp(beta * q[i]) / sum_softmax
        sum_p += p
        if sum_p >= r:
            return i

    return len(q) - 1


def print_qtbl(q_tbl):
    """
    Prints out the Q table.
    """
    print("")
    print("State Q(LEFT) Q(RIGHT)")
    for i in range(len(q_tbl)):
        print("%d      %3.2f     %3.2f" % (i, q_tbl[i][0], q_tbl[i][1]))


def new_qtbl():
    """
    Create a fresh Q table for one independent run.
    """
    return [[0.0 for _ in range(ACTIONS)] for _ in range(STATES)]


def run_trial(q_tbl, p_wait):
    """
    Runs a single delayed gratification trial.
    """
    r = random.random()

    if r < 0.25:
        current_state = EXPM_LR
    elif r < 0.50:
        current_state = EXPM_RL
    elif r < 0.75:
        current_state = CTRL_LR
    else:
        current_state = CTRL_RL

    q = q_tbl[current_state].copy()

    # Apply patience ONLY to live shrimp choices.
    if current_state == EXPM_LR:
        q[LEFT] *= p_wait
    elif current_state == EXPM_RL:
        q[RIGHT] *= p_wait

    act = action_select(q, BETA)

    if current_state == EXPM_LR:
        if act == LEFT:
            rwd = LIVE_RWD
        else:
            rwd = DEAD_RWD
    elif current_state == EXPM_RL:
        if act == LEFT:
            rwd = DEAD_RWD
        else:
            rwd = LIVE_RWD
    elif current_state == CTRL_LR:
        if act == LEFT:
            rwd = UNOBTAINABLE_RWD
        else:
            rwd = DEAD_RWD
    else:
        if act == LEFT:
            rwd = DEAD_RWD
        else:
            rwd = UNOBTAINABLE_RWD

    q_tbl[current_state][act] = q_tbl[current_state][act] + ALPHA * (
        rwd - q_tbl[current_state][act]
    )

    experimental_trial = current_state < CTRL_LR

    return rwd, experimental_trial, current_state, act


def is_correct_action(state, action):
    """
    Return True when the selected action is the better option for the state.
    """
    if state == EXPM_LR:
        return action == LEFT
    if state == EXPM_RL:
        return action == RIGHT
    if state == CTRL_LR:
        return action == RIGHT
    return action == LEFT


def standard_error(values, axis=0):
    """
    Return standard error across independent runs.
    """
    return np.std(values, axis=axis, ddof=1) / np.sqrt(values.shape[axis])


def run_training_phase(q_tbl):
    """
    Run training once and return percent-correct curves for each state.
    """
    state_total = np.zeros(STATES)
    state_hits = np.zeros(STATES)
    state_perf = [[] for _ in range(STATES)]

    for _ in range(TRAINING_TRIALS):
        _, _, state, action = run_trial(q_tbl, 1.0)
        correct = is_correct_action(state, action)

        state_total[state] += 1
        state_hits[state] += int(correct)
        state_perf[state].append(100 * state_hits[state] / state_total[state])

    return state_perf


def run_delay_phase(q_tbl):
    """
    Test one trained Q table across all delays.
    """
    exp_percent = np.zeros(DELAYS.shape[0])
    ctrl_percent = np.zeros(DELAYS.shape[0])

    for delay_idx, delay in enumerate(DELAYS):
        exp_hits = 0
        ctrl_hits = 0
        exp_count = 0
        ctrl_count = 0

        for _ in range(DELAY_TRIALS):
            reward, experimental_trial, _, _ = run_trial(q_tbl, prob_wait(delay))

            if experimental_trial:
                exp_count += 1
                exp_hits += int(reward > DEAD_RWD)
            else:
                ctrl_count += 1
                ctrl_hits += int(reward > UNOBTAINABLE_RWD)

        exp_percent[delay_idx] = 100 * exp_hits / exp_count
        ctrl_percent[delay_idx] = 100 * ctrl_hits / ctrl_count

    return exp_percent, ctrl_percent


def run_one_simulation():
    """
    Run one complete copy of v2: train first, then test delays.
    """
    q_tbl = new_qtbl()
    training_perf = run_training_phase(q_tbl)
    exp_delay_perf, ctrl_delay_perf = run_delay_phase(q_tbl)
    return training_perf, exp_delay_perf, ctrl_delay_perf, q_tbl


def summarize_training(all_training_perf):
    """
    Convert ragged per-state trial histories into mean and SEM arrays.
    """
    all_perf = np.full((RUNS, STATES, TRAINING_TRIALS), np.nan)

    for run_idx, training_perf in enumerate(all_training_perf):
        for state in range(STATES):
            all_perf[run_idx, state, : len(training_perf[state])] = training_perf[state]

    mean_perf = np.full((STATES, TRAINING_TRIALS), np.nan)
    sem_perf = np.full((STATES, TRAINING_TRIALS), np.nan)

    for state in range(STATES):
        for trial_idx in range(TRAINING_TRIALS):
            values = all_perf[:, state, trial_idx]
            values = values[~np.isnan(values)]
            if values.size > 0:
                mean_perf[state, trial_idx] = np.mean(values)
            if values.size > 1:
                sem_perf[state, trial_idx] = np.std(values, ddof=1) / np.sqrt(values.size)

    return mean_perf, sem_perf


def plot_training_curves(mean_perf, sem_perf):
    """
    Plot the 30-run average training curves with standard-error bars.
    """
    labels = [
        "EXPM_LR (Live Left)",
        "EXPM_RL (Live Right)",
        "CTRL_LR (Dead Right)",
        "CTRL_RL (Dead Left)",
    ]

    plt.figure(figsize=(10, 6))

    for state in range(STATES):
        x_vals = np.arange(1, TRAINING_TRIALS + 1)
        valid = ~np.isnan(mean_perf[state])
        plt.errorbar(
            x_vals[valid],
            mean_perf[state][valid],
            yerr=sem_perf[state][valid],
            marker="o",
            capsize=3,
            linewidth=2,
            label=labels[state],
        )

    plt.xlabel("Training Trials (within each state)")
    plt.ylabel("Percent Correct (%)")
    plt.title("Training Learning Curves by State, 30 Runs")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend()
    plt.tight_layout()
    plt.show()


def plot_delay_curves(exp_runs, ctrl_runs):
    """
    Plot the 30-run average delay curves with standard-error bars.
    """
    exp_mean = np.mean(exp_runs, axis=0)
    ctrl_mean = np.mean(ctrl_runs, axis=0)
    exp_sem = standard_error(exp_runs, axis=0)
    ctrl_sem = standard_error(ctrl_runs, axis=0)

    plt.figure(figsize=(10, 6))
    plt.errorbar(
        DELAYS,
        exp_mean,
        yerr=exp_sem,
        marker="o",
        capsize=4,
        linewidth=2,
        label="Experimental",
    )
    plt.errorbar(
        DELAYS,
        ctrl_mean,
        yerr=ctrl_sem,
        marker="s",
        capsize=4,
        linewidth=2,
        label="Control",
    )
    plt.xlabel("Delay (seconds)")
    plt.ylabel("Percent Choosing Better Option (%)")
    plt.title("Delayed Gratification Performance vs. Delay, 30 Runs")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend()
    plt.tight_layout()
    plt.show()

    print("delay\texpm_mean\texpm_sem\tctrl_mean\tctrl_sem")
    for delay, exp_m, exp_s, ctrl_m, ctrl_s in zip(
        DELAYS, exp_mean, exp_sem, ctrl_mean, ctrl_sem
    ):
        print("%d\t%3.2f\t\t%3.2f\t\t%3.2f\t\t%3.2f" % (delay, exp_m, exp_s, ctrl_m, ctrl_s))


def main():
    all_training_perf = []
    exp_delay_runs = np.zeros((RUNS, DELAYS.shape[0]))
    ctrl_delay_runs = np.zeros((RUNS, DELAYS.shape[0]))
    final_q_tables = []

    for run_idx in range(RUNS):
        training_perf, exp_delay_perf, ctrl_delay_perf, q_tbl = run_one_simulation()

        all_training_perf.append(training_perf)
        exp_delay_runs[run_idx] = exp_delay_perf
        ctrl_delay_runs[run_idx] = ctrl_delay_perf
        final_q_tables.append(q_tbl)

    mean_training_perf, sem_training_perf = summarize_training(all_training_perf)

    plot_training_curves(mean_training_perf, sem_training_perf)
    plot_delay_curves(exp_delay_runs, ctrl_delay_runs)

    print("\nFinal Q table from the last run:")
    print_qtbl(final_q_tables[-1])


if __name__ == "__main__":
    main()
