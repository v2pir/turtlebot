import random
from math import exp

import matplotlib.pyplot as plt
import numpy as np

"""
Compare simulated Q-tables against TurtleBot4 Q-values.

The simulation runs 30 independent 100-trial training sessions. Bars show the
simulation mean +/- standard deviation, and black diamonds show the TurtleBot4
Q-values from the robot run.
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

ALPHA = 0.10
BETA = 1.0

RUNS = 30
TRAINING_TRIALS = 100
OUTPUT_FIGURE = "qtable_simulation_vs_turtlebot.png"

TURTLEBOT_Q = np.array(
    [
        [3.588, 0.469],
        [0.410, 4.074],
        [0.433, 0.771],
        [0.718, 0.235],
    ]
)


def action_select(q_values, beta):
    """
    Choose an action with softmax action selection.
    """
    sum_softmax = 0
    sum_p = 0

    for q_value in q_values:
        sum_softmax += exp(beta * q_value)

    r = random.random()

    for action, q_value in enumerate(q_values):
        p = exp(beta * q_value) / sum_softmax
        sum_p += p
        if sum_p >= r:
            return action

    return len(q_values) - 1


def new_qtbl():
    """
    Create a fresh Q table for one independent run.
    """
    return [[0.0 for _ in range(ACTIONS)] for _ in range(STATES)]


def run_trial(q_tbl):
    """
    Run one training trial with no delay discount.
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

    action = action_select(q_tbl[current_state], BETA)

    if current_state == EXPM_LR:
        if action == LEFT:
            reward = LIVE_RWD
        else:
            reward = DEAD_RWD
    elif current_state == EXPM_RL:
        if action == LEFT:
            reward = DEAD_RWD
        else:
            reward = LIVE_RWD
    elif current_state == CTRL_LR:
        if action == LEFT:
            reward = UNOBTAINABLE_RWD
        else:
            reward = DEAD_RWD
    else:
        if action == LEFT:
            reward = DEAD_RWD
        else:
            reward = UNOBTAINABLE_RWD

    q_tbl[current_state][action] = q_tbl[current_state][action] + ALPHA * (
        reward - q_tbl[current_state][action]
    )


def run_training(seed):
    """
    Run one 100-trial training session and return its final Q table.
    """
    random.seed(seed)
    q_tbl = new_qtbl()

    for _ in range(TRAINING_TRIALS):
        run_trial(q_tbl)

    return np.array(q_tbl)


def print_summary(sim_mean, sim_std):
    """
    Print the simulation and TurtleBot Q-values for each state/action.
    """
    print("state\taction\tsim_mean\tsim_std\tturtlebot")
    for state in range(STATES):
        for action, action_name in [(LEFT, "LEFT"), (RIGHT, "RIGHT")]:
            print(
                "%d\t%s\t%3.3f\t\t%3.3f\t\t%3.3f"
                % (
                    state,
                    action_name,
                    sim_mean[state, action],
                    sim_std[state, action],
                    TURTLEBOT_Q[state, action],
                )
            )


def plot_qtable_comparison(sim_mean, sim_std):
    """
    Draw grouped bars for simulation Q-values and black diamonds for TurtleBot4.
    """
    states = np.arange(STATES)
    width = 0.38
    left_x = states - width / 2
    right_x = states + width / 2

    fig, ax = plt.subplots(figsize=(14, 8))

    ax.bar(
        left_x,
        sim_mean[:, LEFT],
        width,
        yerr=sim_std[:, LEFT],
        capsize=8,
        color="#4c72b0",
        edgecolor="black",
        linewidth=1,
        label="Simulation Q(LEFT)",
    )
    ax.bar(
        right_x,
        sim_mean[:, RIGHT],
        width,
        yerr=sim_std[:, RIGHT],
        capsize=8,
        color="#dd8452",
        edgecolor="black",
        linewidth=1,
        label="Simulation Q(RIGHT)",
    )

    ax.scatter(
        left_x,
        TURTLEBOT_Q[:, LEFT],
        marker="D",
        s=130,
        color="black",
        zorder=5,
        label="TurtleBot4 Q-value",
    )
    ax.scatter(
        right_x,
        TURTLEBOT_Q[:, RIGHT],
        marker="D",
        s=130,
        color="black",
        zorder=5,
    )

    ax.set_title("Q-Table: Simulation (mean +/- std, 30 seeds) vs. TurtleBot4", fontsize=24)
    ax.set_ylabel("Q-value", fontsize=18)
    ax.set_xticks(states)
    ax.set_xticklabels([f"State {state}" for state in range(STATES)], fontsize=18)
    ax.tick_params(axis="y", labelsize=16)
    ax.set_ylim(0, 5.05)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(fontsize=16, frameon=False)

    plt.tight_layout()
    plt.savefig(OUTPUT_FIGURE, dpi=300)
    plt.show()


def main():
    q_tables = np.array([run_training(seed) for seed in range(RUNS)])
    sim_mean = np.mean(q_tables, axis=0)
    sim_std = np.std(q_tables, axis=0, ddof=1)

    print_summary(sim_mean, sim_std)
    plot_qtable_comparison(sim_mean, sim_std)


if __name__ == "__main__":
    main()
