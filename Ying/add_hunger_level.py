###########################################################################
# Two-Chamber Aquarium Episodic-Like Memory (ELM)
# with Neuromodulated "Hunger/Patience" System
# Inspired by: Xing, Zou & Krichmar (2021) - Neuromodulated Patience
###########################################################################
import numpy as np
import matplotlib.pyplot as plt
import random
from dataclasses import dataclass
from typing import List
import math
from scipy.stats import norm

# -----------------------------
# Parameters
# -----------------------------
LR = 0.25
REWARD_DEAD = 1.0
REWARD_LIVE = 4.0
DELAY_LEVELS = [10, 20, 30, 40, 50, 60]
SESSIONS_PER_DELAY = 3

# Hunger modulation parameters
HUNGER_INIT = 0.5       # baseline hunger (0 = full/patient, 1 = starving/impatient)
HUNGER_INCREASE = 0.05  # hunger rises slightly each trial without food
HUNGER_DECAY = 0.2      # hunger drops when rewarded
BETA = 8.0              # sensitivity parameter controlling sigmoid steepness


# -----------------------------
# Neuromodulated patience model
# -----------------------------
def patience_probability(t, delay_s, hunger):
    """Return p(wait|t) as a function of elapsed time, delay, and hunger."""
    # Likelihood of reward as a function of elapsed time
    L_t = norm.cdf(t, loc=delay_s * 0.8, scale=delay_s * 0.4)
    # Map hunger (0-1) to serotonin-like modulation (-1 to +1)
    mod = 1.0 - 2.0 * hunger   # high hunger → impatient (negative mod)
    # Logistic decision curve
    return 1.0 / (1.0 + math.exp(-BETA * mod * (L_t - 0.5)))


# -----------------------------
# Trial log
# -----------------------------
@dataclass
class TrialLog:
    delay_s: int
    is_experimental: bool
    choice: str          # "immediate" or "delayed"
    waited: bool
    abandoned: bool
    wait_duration_s: float
    success: bool
    reward: float
    hunger_before: float
    hunger_after: float


# -----------------------------
# Memory update (delta rule)
# -----------------------------
def assocMemoryLR(epiMem, lr, reward, what, when, where):
    epiMem[what, when, where] += lr * (reward - epiMem[what, when, where])
    return epiMem


# -----------------------------
# Core simulation
# -----------------------------
def ceph_epimem(lr=LR):
    DEAD, LIVE = 0, 1
    LEFT, RIGHT = 0, 1
    epiMem = np.zeros((2, len(DELAY_LEVELS), 2))
    logs: List[TrialLog] = []

    hunger = HUNGER_INIT  # initialize hunger state

    for w_i, delay_s in enumerate(DELAY_LEVELS):
        for _ in range(SESSIONS_PER_DELAY):
            exp_successes = []

            # ---------- Experimental condition ----------
            # Left = immediate dead shrimp, Right = delayed live shrimp
            for _ in range(3):
                v_imm = epiMem[DEAD, w_i, LEFT]
                v_del = epiMem[LIVE, w_i, RIGHT]

                # --- softmax exploration (Boltzmann policy)
                probs = np.exp([v_imm, v_del])
                probs /= probs.sum()
                choice = np.random.choice(["immediate", "delayed"], p=probs)
                waited = (choice == "delayed")
                abandoned = False
                success = False
                wait_duration_s = 0
                reward = 0
                hunger_before = hunger

                if waited:
                    # patience model governs probability of staying patient
                    p_wait = patience_probability(
                        t=random.uniform(0, delay_s),
                        delay_s=delay_s,
                        hunger=hunger
                    )
                    if random.random() > p_wait:
                        # abandoned before success → zero update for delayed/live
                        abandoned = True
                        wait_duration_s = random.uniform(1, delay_s - 1)
                        epiMem = assocMemoryLR(epiMem, lr, 0.0, LIVE, w_i, RIGHT)
                        hunger = min(1.0, hunger + HUNGER_INCREASE)  # got nothing → more hungry
                    else:
                        # successful wait → live shrimp reward
                        success = True
                        wait_duration_s = delay_s
                        reward = REWARD_LIVE
                        epiMem = assocMemoryLR(epiMem, lr, reward, LIVE, w_i, RIGHT)
                        hunger = max(0.0, hunger - HUNGER_DECAY)  # eating → less hungry
                else:
                    # immediate capture of dead shrimp
                    reward = REWARD_DEAD
                    epiMem = assocMemoryLR(epiMem, lr, reward, DEAD, w_i, LEFT)
                    hunger = max(0.0, hunger - HUNGER_DECAY / 2)  # small meal

                logs.append(
                    TrialLog(delay_s, True, choice, waited, abandoned,
                             wait_duration_s, success, reward,
                             hunger_before, hunger)
                )
                exp_successes.append(success)

            # ---------- Control condition ----------
            # Both chambers dead shrimp, side randomized
            for _ in range(3):
                hunger_before = hunger
                if random.random() < 0.5:
                    imm_side, del_side = LEFT, RIGHT
                else:
                    imm_side, del_side = RIGHT, LEFT

                # immediate side normal; delayed side strongly biased lower
                v_imm = epiMem[DEAD, w_i, imm_side]
                v_del = epiMem[DEAD, w_i, del_side] - 2.0  # discourages delay
                probs = np.exp([v_imm, v_del])
                probs /= probs.sum()
                choice = np.random.choice(["immediate", "delayed"], p=probs)
                waited = (choice == "delayed")

                if waited:
                    reward = 0.0
                    epiMem = assocMemoryLR(epiMem, lr, reward, DEAD, w_i, del_side)
                    hunger = min(1.0, hunger + HUNGER_INCREASE)  # wasted time
                else:
                    reward = REWARD_DEAD
                    epiMem = assocMemoryLR(epiMem, lr, reward, DEAD, w_i, imm_side)
                    hunger = max(0.0, hunger - HUNGER_DECAY / 2)

                logs.append(
                    TrialLog(delay_s, False, choice, waited, False,
                             delay_s if waited else 0, True, reward,
                             hunger_before, hunger)
                )

            # move to next delay only if all exp trials succeeded
            if all(exp_successes):
                break
    return epiMem, logs


# -----------------------------
# Plot 1 - mean % choosing delayed
# -----------------------------
def plot_choice_stats(all_runs: List[List[TrialLog]]):
    delays = DELAY_LEVELS
    n_runs = len(all_runs)
    exp = np.zeros((n_runs, len(delays)))
    ctrl = np.zeros_like(exp)

    for i, logs in enumerate(all_runs):
        counts = {d: {"exp": [0, 0], "ctrl": [0, 0]} for d in delays}
        for L in logs:
            k = "exp" if L.is_experimental else "ctrl"
            counts[L.delay_s][k][0] += 1
            if L.choice == "delayed":
                counts[L.delay_s][k][1] += 1
        exp[i] = [100 * (counts[d]["exp"][1] / counts[d]["exp"][0] if counts[d]["exp"][0] else 0) for d in delays]
        ctrl[i] = [100 * (counts[d]["ctrl"][1] / counts[d]["ctrl"][0] if counts[d]["ctrl"][0] else 0) for d in delays]

    mean_exp, mean_ctrl = exp.mean(0), ctrl.mean(0)

    plt.figure(figsize=(6, 4))
    plt.plot(delays, mean_exp, 'o-', color='tab:blue', label='Experimental (live shrimp)')
    plt.plot(delays, mean_ctrl, 's-', color='tab:orange', label='Control (dead shrimp)')
    plt.xlabel('Delay (s)')
    plt.ylabel('Chose delayed (%)')
    plt.title('Choice of delayed door across delay levels')
    plt.ylim(0, 105)
    plt.xticks(delays)
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.legend()
    plt.tight_layout()
    plt.show()


# -----------------------------
# Plot 2 - hunger dynamics (optional)
# -----------------------------
def plot_hunger_dynamics(logs: List[TrialLog]):
    hunger_vals = [L.hunger_after for L in logs if L.is_experimental]
    plt.figure(figsize=(6, 3))
    plt.plot(hunger_vals, color='tab:red')
    plt.xlabel("Trial # (Experimental)")
    plt.ylabel("Hunger level (0–1)")
    plt.title("Hunger dynamics across experimental trials")
    plt.grid(alpha=0.4)
    plt.tight_layout()
    plt.show()


# -----------------------------
# Main
# -----------------------------
if __name__ == "__main__":
    runs = 10
    all_logs = []
    for r in range(runs):
        print(f"Run {r+1}/{runs}")
        _, logs = ceph_epimem()
        all_logs.append(logs)

    plot_choice_stats(all_logs)

    # visualize hunger in the last run
    plot_hunger_dynamics(all_logs[-1])

    print("Simulation complete.")
