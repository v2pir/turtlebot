from controller import Robot, GPS
from better import assocMemoryLR, LR, REWARD_LIVE, REWARD_DEAD
import numpy as np
import random
import time

# setup and params
robot = Robot()
timestep = int(robot.getBasicTimeStep())

left_motor = robot.getDevice("left wheel motor")
right_motor = robot.getDevice("right wheel motor")
camera = robot.getDevice("camera")
gps = robot.getDevice("gps")

camera.enable(timestep)
gps.enable(timestep)
left_motor.setPosition(float("inf"))
right_motor.setPosition(float("inf"))
left_motor.setVelocity(0)
right_motor.setVelocity(0)

DELAY_LEVELS = [50, 100, 150, 200, 250, 300]
epiMem = np.random.uniform(low=-1.0, high=3.0, size=(2, len(DELAY_LEVELS), 2))  # pre-filled episodic memory
delay_index = 0  # test
LEFT, RIGHT = 0, 1
DEAD, LIVE = 0, 1

# Threshold coordinates (TODO: fix threshold length?)
BLUE_THRESHOLD_X, BLUE_THRESHOLD_Z = -0.62, 1.45
RED_THRESHOLD_X,  RED_THRESHOLD_Z  =  0.61, 1.45

decision_state = "idle"
current_choice = None
pre_trial_timer = 0.0
wait_timer = 0.0

random.seed(time.time())

print(f"Randomized epiMem slice(delay 50s):") # tentative
print(epiMem[:, delay_index, :])

# helpers
def move_forward(speed=6.0):
    left_motor.setVelocity(speed)
    right_motor.setVelocity(speed)

def turn_left(speed=6.0):
    left_motor.setVelocity(-speed)
    right_motor.setVelocity(speed)

def turn_right(speed=6.0):
    left_motor.setVelocity(speed)
    right_motor.setVelocity(-speed)

def stop():
    left_motor.setVelocity(0)
    right_motor.setVelocity(0)

def detect_colors(image, width, height):
    """Return mean x positions of detected red and blue regions."""
    reds, blues = [], []
    for x in range(0, width, 2):
        for y in range(height // 3, 2 * height // 3, 2):
            r = camera.imageGetRed(image, width, x, y)
            g = camera.imageGetGreen(image, width, x, y)
            b = camera.imageGetBlue(image, width, x, y)
            if r > g * 1.5 and r > b * 1.5:
                reds.append(x)
            elif b > r * 1.5 and b > g * 1.5:
                blues.append(x)
    mean_red = np.mean(reds) if reds else None
    mean_blue = np.mean(blues) if blues else None
    return mean_red, mean_blue


# --- Main loop ---
while robot.step(timestep) != -1:
    image = camera.getImage()
    width = camera.getWidth()
    height = camera.getHeight()
    mean_red, mean_blue = detect_colors(image, width, height)
    x, _, z = gps.getValues()
    dt = timestep / 1000.0
    delay_s = DELAY_LEVELS[delay_index]

    # ------------------ PRE-TRIAL ------------------
    if decision_state == "idle":
        pre_trial_timer += dt
        stop()
        if pre_trial_timer >= 3.0:
            pre_trial_timer = 0.0
            decision_state = "choose"
            print("\n[State] Trial start — robot perceives both chambers.")

    # ------------------ DECISION ------------------
    elif decision_state == "choose":
        v_imm = epiMem[DEAD, delay_index, LEFT]
        v_del = epiMem[LIVE, delay_index, RIGHT]
        tau = 1.0
        probs = np.exp([v_imm / tau, v_del / tau])
        probs /= probs.sum()
        choice = np.random.choice(["blue", "red"], p=probs)
        current_choice = choice

        print(f"[Decision] memory_dead={v_imm:.2f}, memory_live={v_del:.2f}, p(red)={probs[1]:.2f}")
        print(f"[Decision] Robot chooses {choice.upper()} chamber.")
        decision_state = f"turn_{choice}"

    # ------------------ TURNING ------------------
    elif decision_state == "turn_blue":
        if mean_blue is not None and mean_blue < width / 2 - 10:
            turn_left(3.0)
        elif mean_blue is not None and mean_blue > width / 2 + 10:
            turn_right(3.0)
        else:
            stop()
            decision_state = "approach_blue"

    elif decision_state == "turn_red":
        if mean_red is not None and mean_red < width / 2 - 10:
            turn_left(3.0)
        elif mean_red is not None and mean_red > width / 2 + 10:
            turn_right(3.0)
        else:
            stop()
            decision_state = "approach_red"

    # ------------------ APPROACH ------------------
    elif decision_state == "approach_blue":
        move_forward(2.0)
        x, _, z = gps.getValues()
        if z >= BLUE_THRESHOLD_Z - 0.02 and abs(x - BLUE_THRESHOLD_X) < 0.8:
            stop()
            print("[Capture] Crossed BLUE threshold → captured dead shrimp (immediate reward).")
            decision_state = "done"

    elif decision_state == "approach_red":
        move_forward(2.0)
        x, _, z = gps.getValues()
        if z >= RED_THRESHOLD_Z - 0.02 and abs(x - RED_THRESHOLD_X) < 0.8:
            stop()
            print(f"[Capture] Crossed RED threshold → waiting for live shrimp door ({delay_s}s delay).")
            wait_timer = 0.0
            decision_state = "waiting_red"

    # ------------------ WAITING (RED) ------------------
    elif decision_state == "waiting_red":
        wait_timer += dt
        if wait_timer >= delay_s:
            print(f"[Outcome] Door opened after {delay_s}s → robot receives live shrimp reward.")
            decision_state = "done"

    # ------------------ END OF TRIAL ------------------
    elif decision_state == "done":
        stop()
        print("[Simulation] Trial complete — demonstration ends.")
        break