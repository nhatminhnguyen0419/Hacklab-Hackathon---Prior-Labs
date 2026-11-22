"""
Shared task definitions: the five objects, their shape features and wrist symmetry, the rule
classifier (the baseline TabPFN is compared with), reward weights, and the dataset CSV columns
(spec section 8) the web app's upload check uses.

(This was generate_dataset.py, the scripted data generator of our first, homemade simulator;
that simulator and its generator are in the git history.)
"""

import math

import mujoco
import numpy as np

OBJECTS = ["cup", "bottle", "box", "pen", "ball"]

# Half-thickness of each object in the direction of the palm normal when grasped (m)
GRASP_RADIUS = {"cup": 0.04, "bottle": 0.033, "box": 0.02, "pen": 0.006, "ball": 0.035}

# Wrist symmetry period (rad). None = no target (ball).
SYMMETRY = {"cup": math.pi, "bottle": math.pi, "pen": math.pi, "box": math.pi / 2, "ball": None}

# True shape features: (height along long axis, width, depth) in m
TRUE_DIMS = {
    "cup": (0.10, 0.11, 0.08),
    "bottle": (0.24, 0.066, 0.066),
    "box": (0.10, 0.06, 0.04),
    "pen": (0.152, 0.012, 0.012),
    "ball": (0.07, 0.07, 0.07),
}
ROUNDNESS = {"cup": 0.5, "bottle": 0.5, "box": 0.0, "pen": 0.5, "ball": 1.0}

DETECT_DIST = 0.50           # m, object counts as detected inside this distance
ALIGN_TOL = math.radians(15)
HOLD_TIME_FOR_SUCCESS = 1.0  # s
MAX_EPISODE_TIME = 8.0       # s

REWARD = {
    "align": 0.5,          # per step, scaled 0..1 by alignment
    "approach": 0.5,       # per step, minus distance hand-object (m)
    "gentle_touch": 1.0,   # once, aligned gentle contact
    "grasp": 2.0,          # once, fingers close after aligned contact
    "hold": 0.2,           # per step while object held
    "impact": -2.0,        # per step with contact force above sim/env.py FORCE_LIMIT
    "jerk": -0.1,          # times sum of |change in action|
    "early_grasp": -1.0,   # per step closing without contact
    "no_command": -0.2,    # per step moving while not commanded
}

# Dataset CSV columns (spec section 8)
STEP_COLUMNS = [
    "episode_id", "time_step", "timestamp", "voice_command", "phase",
    "object_type",
    "obj_x", "obj_y", "obj_z", "obj_qw", "obj_qx", "obj_qy", "obj_qz", "obj_speed",
    "obj_rel_x", "obj_rel_y", "obj_rel_z",
    "feat_height", "feat_width", "feat_depth", "feat_elongation", "feat_roundness",
    "object_detected", "predicted_object_type",
    "shoulder_angle", "elbow_angle", "wrist_angle",
    "shoulder_vel", "elbow_vel", "wrist_vel", "finger_closure",
    "imu_acc_x", "imu_acc_y", "imu_acc_z", "imu_gyro_x", "imu_gyro_y", "imu_gyro_z",
    "act_shoulder", "act_elbow", "act_wrist", "act_grip",
    "contact", "contact_force", "grip_force", "object_held",
    "ideal_wrist_angle", "wrist_error", "reward", "success",
]
EPISODE_COLUMNS = [
    "episode_id", "object_type", "predicted_object_type", "success",
    "time_to_grasp", "max_contact_force", "wrist_error_at_grasp",
    "had_stop", "episode_length_s",
]


def wrap_err(q, target, period):
    """Signed smallest difference q - target, modulo the symmetry period."""
    return (q - target + period / 2) % period - period / 2


def nearest_equivalent(q, target, period, lo, hi):
    """The equivalent of target (target + k*period) inside [lo, hi] nearest to q."""
    cands = [target + k * period for k in range(-3, 4)]
    cands = [c for c in cands if lo <= c <= hi] or [float(np.clip(target, lo, hi))]
    return min(cands, key=lambda c: abs(c - q))


def mat_to_quat(mat):
    q = np.zeros(4)
    mujoco.mju_mat2Quat(q, np.asarray(mat, dtype=float).reshape(9))
    return q


def classify(features):
    """Baseline rule classifier on noisy shape features (TabPFN's comparison in classifier.py)."""
    h, w, d, elong, rnd = features
    if elong > 5.0:
        return "pen"
    if rnd > 0.75:
        return "ball"
    if rnd < 0.25:
        return "box"
    if h / max(w, d) > 2.2:
        return "bottle"
    return "cup"
