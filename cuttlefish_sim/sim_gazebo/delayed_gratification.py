"""
Delayed Gratification Q-Learning Library

Refactored from CARL-Delayed-Gratification/v2_delayed_gratification.py
for use as a reusable library in the Gazebo simulation.

Original author: Kary Zheng (14 January 2026)
Based on: Jeff Krichmar (30 November 2025)
"""

import random
from math import exp

import numpy as np
from scipy.stats import norm


# Actions
ACTIONS = 2
LEFT = 0
RIGHT = 1

# States
STATES = 4
EXPM_LR = 0  # Live Shrimp (left); Dead Shrimp (right)
EXPM_RL = 1  # Live Shrimp (right); Dead Shrimp (left)
CTRL_LR = 2  # Unobtainable Shrimp (left); Dead Shrimp (right)
CTRL_RL = 3  # Unobtainable Shrimp (right); Dead Shrimp (left)

# Rewards
LIVE_RWD = 5.0
DEAD_RWD = 1.0
UNOBTAINABLE_RWD = 0.0

# Learning parameters
ALPHA = 0.10
BETA = 1.0
GAMMA = 0.99

# State labels for logging
STATE_NAMES = ["EXPM_LR", "EXPM_RL", "CTRL_LR", "CTRL_RL"]
ACTION_NAMES = ["LEFT", "RIGHT"]


class QTable:
    """wrap the Q-table in a class"""

    def __init__(self):
        self.table = [[0.0 for _ in range(ACTIONS)] for _ in range(STATES)]

    def get(self, state, action):
        return self.table[state][action]

    def update(self, state, action, reward):
        old = self.table[state][action]
        self.table[state][action] = old + ALPHA * (reward - old)

    def get_row(self, state):
        return list(self.table[state])

    def print_table(self):
        print("\nState      Q(LEFT)  Q(RIGHT)")
        for i in range(STATES):
            print(f"{STATE_NAMES[i]:10s}  {self.table[i][LEFT]:6.3f}   {self.table[i][RIGHT]:6.3f}")

    def to_dict(self):
        return {STATE_NAMES[s]: {ACTION_NAMES[a]: self.table[s][a]
                for a in range(ACTIONS)} for s in range(STATES)}


def prob_wait(tim):
    """
    Calculate the probability to wait using a cumulative normal distribution.

    :param tim: elapsed time in seconds
    :return: probability (0 = don't wait, 1 = definitely wait)
    """
    mean = 70
    std_dev = 20
    beta_weight = 2
    pw = 1 / np.exp(beta_weight * norm.cdf(tim, loc=mean, scale=std_dev))
    return pw

def action_select(q, beta):
    """
    Softmax action selection. Converts Q-values into a probability
    distribution and samples an action.

    q: list of Q-values for each action
    beta: temperature parameter
    return: selected action index
    """
    # get the summed softmax
    sum_softmax = sum(exp(beta * q[i]) for i in range(len(q)))

    # random number between 0 and 1
    r = random.random()
    cumulative = 0.0
    
    # loop through q values and calculate cumulative probability
    for i in range(len(q)):
        cumulative += exp(beta * q[i]) / sum_softmax
        # if sum is more than random number, return that action
        if cumulative >= r:
            return i

    return len(q) - 1


def run_trial(qtable, p_wait, forced_state=None):
    """
    Run a single delayed gratification trial.

    qtable: QTable instance
    p_wait: probability of waiting (1.0 = no delay effect)
    forced_state: if provided, use this state instead of random
    return: (reward, is_experimental, state, action)
    """
    # pick a state randomly or use forced_state (25% chance for each state)
    if forced_state is not None:
        current_state = forced_state
    else:
        r = random.random()
        if r < 0.25:
            current_state = EXPM_LR
        elif r < 0.50:
            current_state = EXPM_RL
        elif r < 0.75:
            current_state = CTRL_LR
        else:
            current_state = CTRL_RL

    # copy q-values so patience discount doesn't change the table
    q = list(qtable.get_row(current_state))

    # apply patience discount (live shrimp states only)
    if current_state == EXPM_LR:
        q[LEFT] *= p_wait
    elif current_state == EXPM_RL:
        q[RIGHT] *= p_wait

    # select action via softmax
    act = action_select(q, BETA)

    # determine reward based on state and action
    if current_state == EXPM_LR:
        rwd = LIVE_RWD if act == LEFT else DEAD_RWD
    elif current_state == EXPM_RL:
        rwd = DEAD_RWD if act == LEFT else LIVE_RWD
    elif current_state == CTRL_LR:
        rwd = UNOBTAINABLE_RWD if act == LEFT else DEAD_RWD
    else:
        rwd = DEAD_RWD if act == LEFT else UNOBTAINABLE_RWD

    # update q-table with actual reward
    qtable.update(current_state, act, rwd)

    is_experimental = current_state < CTRL_LR

    return rwd, is_experimental, current_state, act


def is_correct(state, action):
    """check if the action is the 'correct' (higher reward) choice for a state."""
    if state == EXPM_LR:
        return action == LEFT
    elif state == EXPM_RL:
        return action == RIGHT
    elif state == CTRL_LR:
        return action == RIGHT
    else:
        return action == LEFT


def action_to_chamber(state, action):
    """
    Map a (state, action) pair to a physical chamber direction.

    return: 'LEFT' or 'RIGHT' (the physical chamber the robot should go to)
    """
    return ACTION_NAMES[action]
