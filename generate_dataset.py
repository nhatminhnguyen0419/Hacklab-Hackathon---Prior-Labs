"""
Dataset generator - HackLab #2 prosthetic arm.

Runs scripted episodes in scene.xml and writes two CSV files:
  <out>_steps.csv     one row per control step (default 25 Hz)
  <out>_episodes.csv  one row per episode (summary)

Episode flow (matches the spec doc):
  voice "ready" -> object detected -> object identified -> wrist aligns
  -> arm reaches -> fingers close -> object held -> voice "release"

Usage (from the folder with scene.xml):
  python generate_dataset.py --episodes 20 --view        # watch a few first
  python generate_dataset.py --episodes 3000 --out data/run1

The controller here is SCRIPTED (with randomness), not RL. It produces
varied, sensible episodes so the team has data before the agent exists.
Reward weights are in REWARD below so they can be tuned in one place.
"""

import argparse
import csv
import math
import os
import time

import numpy as np
import mujoco

# ----------------------------------------------------------------------------
# Settings
# ----------------------------------------------------------------------------
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

DETECT_DIST = 0.50          # m, object counts as detected inside this distance
GRASP_TRIGGER_DIST = 0.06   # m, start closing when object is this close to the grasp point
GRASP_LOCAL_X = 0.05        # m, grasp point along the palm (between thumb at the heel and the fingers)
SQUEEZE = 0.1               # rad, extra closing after the fingers first touch the object
LET_GO_DELAY = 0.15         # s, carrier lets go this long after the fingers touch
ALIGN_TOL = math.radians(15)
HOLD_TIME_FOR_SUCCESS = 1.0  # s
MAX_EPISODE_TIME = 8.0       # s
FORCE_LIMIT = 20.0           # N, above this counts as a hard impact

REWARD = {
    "align": 0.5,          # per step, scaled 0..1 by alignment
    "approach": 0.5,       # per step, minus distance hand-object (m)
    "gentle_touch": 1.0,   # per step with aligned, gentle contact
    "grasp": 2.0,          # once, fingers close after aligned contact
    "hold": 0.2,           # per step while object held
    "impact": -2.0,        # per step with contact force above FORCE_LIMIT
    "jerk": -0.1,          # times sum of |change in action|
    "joint_limit": -0.5,   # per step when a joint is within 2 deg of its limit
    "early_grasp": -1.0,   # per step closing without contact
    "no_command": -0.2,    # per step moving while not commanded
}

ARM_JOINTS = ["shoulder_inout", "elbow", "wrist_rot"]
ACTUATORS = ["shoulder_inout", "elbow", "wrist_rot", "grip"]
GRIP_CLOSED = 1.1


# ----------------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------------
def wrap_err(q, target, period):
    """Signed smallest difference q - target, modulo the symmetry period."""
    d = (q - target + period / 2) % period - period / 2
    return d


def nearest_equivalent(q, target, period, lo, hi):
    """The equivalent of target (target + k*period) inside [lo, hi] nearest to q."""
    cands = [target + k * period for k in range(-3, 4)]
    cands = [c for c in cands if lo <= c <= hi] or [float(np.clip(target, lo, hi))]
    return min(cands, key=lambda c: abs(c - q))


def mat_to_quat(mat):
    q = np.zeros(4)
    mujoco.mju_mat2Quat(q, np.asarray(mat, dtype=float).reshape(9))
    return q


def contact_geoms(c):
    if hasattr(c, "geom1"):
        return c.geom1, c.geom2
    return c.geom[0], c.geom[1]


def grasp_local(obj):
    """Grasp point in the hand frame: in front of the palm, between thumb and fingers."""
    return np.array([GRASP_LOCAL_X, 0.012 + GRASP_RADIUS[obj] + 0.02, 0.0])


def classify(features):
    """Baseline rule classifier on noisy shape features (placeholder for the trained one)."""
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


# ----------------------------------------------------------------------------
# Simulator wrapper
# ----------------------------------------------------------------------------
class ArmSim:
    def __init__(self, scene_path, rate_hz):
        self.m = mujoco.MjModel.from_xml_path(scene_path)
        self.d = mujoco.MjData(self.m)
        self.substeps = max(1, int(round(1.0 / (rate_hz * self.m.opt.timestep))))
        self.dt = self.substeps * self.m.opt.timestep

        m = self.m
        self.jq = {j: m.joint(j).qposadr[0] for j in ARM_JOINTS}
        self.jdof = {j: m.joint(j).dofadr[0] for j in ARM_JOINTS}
        self.jrange = {j: m.jnt_range[m.joint(j).id].copy() for j in ARM_JOINTS}
        self.grip_q = m.joint("index_j1").qposadr[0]
        self.act = {a: m.actuator(a).id for a in ACTUATORS}
        self.hand = m.body("hand").id
        self.carrier_mocap = m.body_mocapid[m.body("carrier").id]
        self.obj_body = {o: m.body(o).id for o in OBJECTS}
        self.obj_q = {o: m.joint(f"{o}_free").qposadr[0] for o in OBJECTS}
        self.obj_v = {o: m.joint(f"{o}_free").dofadr[0] for o in OBJECTS}
        self.weld = {o: m.equality(f"hold_{o}").id for o in OBJECTS}
        self.grip_weld = {o: m.equality(f"grip_{o}").id for o in OBJECTS}
        self.obj_geoms = {o: list(range(m.body_geomadr[self.obj_body[o]],
                                        m.body_geomadr[self.obj_body[o]] + m.body_geomnum[self.obj_body[o]]))
                          for o in OBJECTS}
        self.conaffinity0 = m.geom_conaffinity.copy()

        # Arm and finger bodies (for contact checks)
        root = m.body("shoulder").id
        self.arm_bodies = set()
        for b in range(m.nbody):
            p = b
            while p != 0:
                if p == root:
                    self.arm_bodies.add(b)
                    break
                p = m.body_parentid[p]
        self.finger_bodies = {
            m.body(f"{f}_{i}").id
            for f in ["index", "middle", "ring", "little"]  # not thumb: it touches the object before the grip closes
            for i in (1, 2, 3)
        }

    def sensor(self, name):
        return self.d.sensor(name).data.copy()

    def hand_pose(self):
        return self.d.xpos[self.hand].copy(), self.d.xmat[self.hand].reshape(3, 3).copy()

    def grasp_point(self, obj):
        pos, R = self.hand_pose()
        return pos + R @ grasp_local(obj)

    def forward_at(self, q_arm):
        """Hand pose for given (shoulder, elbow, wrist) without changing the state."""
        saved = self.d.qpos.copy()
        for j, q in zip(ARM_JOINTS, q_arm):
            self.d.qpos[self.jq[j]] = q
        mujoco.mj_kinematics(self.m, self.d)
        pos, R = self.hand_pose()
        self.d.qpos[:] = saved
        mujoco.mj_kinematics(self.m, self.d)
        return pos, R

    def obj_in_hand(self, obj):
        """Object pose in the hand frame: (position, quaternion)."""
        ob = self.obj_body[obj]
        Rh = self.d.xmat[self.hand].reshape(3, 3)
        qh_inv, q = np.zeros(4), np.zeros(4)
        mujoco.mju_negQuat(qh_inv, self.d.xquat[self.hand])
        mujoco.mju_mulQuat(q, qh_inv, self.d.xquat[ob])
        return Rh.T @ (self.d.xpos[ob] - self.d.xpos[self.hand]), q

    def attach(self, obj, on, pose=None):
        """Grasp assist: weld obj to the hand at pose (object in hand frame, default: where it is now),
        or let go. Call again with a new pose to move the held object."""
        e = self.grip_weld[obj]
        if on:
            p, q = pose if pose is not None else self.obj_in_hand(obj)
            # the weld's relpose is the other way round: hand (body2) in the object (body1) frame
            q_inv, R = np.zeros(4), np.zeros(9)
            mujoco.mju_negQuat(q_inv, q)
            mujoco.mju_quat2Mat(R, q)
            self.m.eq_data[e][3:6] = -R.reshape(3, 3).T @ p
            self.m.eq_data[e][6:10] = q_inv
        self.d.eq_active[e] = int(on)

    def hand_contact(self, obj, on):
        """Turn the object's collisions with the arm on/off (arm geoms have conaffinity 0, so the
        object's conaffinity alone decides; it still hits the floor through its contype)."""
        for g in self.obj_geoms[obj]:
            self.m.geom_conaffinity[g] = self.conaffinity0[g] if on else 0

    def contact_forces(self, obj):
        """Total normal force object-arm and object-fingers (N)."""
        total, fingers = 0.0, 0.0
        f6 = np.zeros(6)
        ob = self.obj_body[obj]
        for i in range(self.d.ncon):
            g1, g2 = contact_geoms(self.d.contact[i])
            b1, b2 = self.m.geom_bodyid[g1], self.m.geom_bodyid[g2]
            if ob in (b1, b2):
                other = b2 if b1 == ob else b1
                if other in self.arm_bodies:
                    mujoco.mj_contactForce(self.m, self.d, i, f6)
                    total += abs(f6[0])
                    if other in self.finger_bodies:
                        fingers += abs(f6[0])
        return total, fingers


# ----------------------------------------------------------------------------
# One episode
# ----------------------------------------------------------------------------
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


def run_episode(sim, ep, rng, viewer=None):
    m, d = sim.m, sim.d
    mujoco.mj_resetData(m, d)
    for o in OBJECTS:
        d.eq_active[sim.weld[o]] = 0

    obj = OBJECTS[rng.integers(len(OBJECTS))]
    period = SYMMETRY[obj]
    wlo, whi = sim.jrange["wrist_rot"]

    # --- start pose of the arm
    q0 = [rng.uniform(-0.2, 0.3), rng.uniform(-0.3, 0.3), rng.uniform(-0.5, 0.5)]
    for j, q in zip(ARM_JOINTS, q0):
        d.qpos[sim.jq[j]] = q
    ctrl = np.array(q0 + [0.0])
    d.ctrl[:] = ctrl

    # --- target arm config (reachable by construction) and the true wrist target
    q_star = rng.uniform(math.radians(-80), math.radians(80))
    q_target = [rng.uniform(-0.4, 0.9), rng.uniform(-0.5, 1.2), q_star]
    hpos, hR = sim.forward_at(q_target)
    G = hpos + hR @ grasp_local(obj)  # hand-over point

    # Object orientation: long axis along hand z at the true wrist target
    if obj == "box":
        align = np.array([[0, 1, 0], [0, 0, 1], [1, 0, 0]], dtype=float)  # cols: x->z, y->x, z->y
    else:
        align = np.eye(3)
    spin = rng.uniform(0, 2 * math.pi) if obj in ("cup", "bottle", "pen") else 0.0
    cs, sn = math.cos(spin), math.sin(spin)
    Rspin = np.array([[cs, -sn, 0], [sn, cs, 0], [0, 0, 1]])
    if obj == "ball":
        R_obj = hR @ Rspin
    else:
        R_obj = hR @ align @ Rspin
    quat = mat_to_quat(R_obj)

    # --- approach path of the carrier: mostly along the palm normal, with some variation
    direction = hR[:, 1] + rng.normal(0, 0.25, 3)
    direction /= np.linalg.norm(direction)
    S = G + direction * rng.uniform(0.4, 0.7)
    speed = rng.uniform(0.15, 0.35)
    travel_time = np.linalg.norm(G - S) / speed

    qa = sim.obj_q[obj]
    d.qpos[qa:qa + 3] = S
    d.qpos[qa + 3:qa + 7] = quat
    d.mocap_pos[sim.carrier_mocap] = S
    d.mocap_quat[sim.carrier_mocap] = quat
    d.eq_active[sim.weld[obj]] = 1
    mujoco.mj_forward(m, d)

    # --- noisy shape features and identification (done at detection)
    h, w, dd = TRUE_DIMS[obj]
    feats_true = np.array([h, w, dd]) * (1 + rng.normal(0, 0.05, 3))
    elong = feats_true.max() / feats_true.min()
    rnd = float(np.clip(ROUNDNESS[obj] + rng.normal(0, 0.12), 0, 1))
    features = (*feats_true, elong, rnd)

    # --- voice timeline
    t_ready = rng.uniform(0.2, 1.0)
    stop_window = None
    if rng.random() < 0.05:
        t0 = rng.uniform(t_ready + 0.3, t_ready + 2.0)
        stop_window = (t0, t0 + 0.8)

    # --- per-episode controller style
    reach_rate = rng.uniform(0.8, 1.6)    # rad/s
    wrist_rate = rng.uniform(1.5, 3.0)    # rad/s
    grip_rate = rng.uniform(2.0, 3.5)     # rad/s
    target_noise = rng.normal(0, 0.01, 2)
    wrist_noise = rng.normal(0, math.radians(4))

    detected = False
    predicted = ""
    grip_cmd_target = 0.0
    weld_on = True
    t_grasp = None
    t_weld_off = None
    held_time = 0.0
    released = False
    t_release = None
    grasp_bonus_given = False
    t_touch = None
    max_force = 0.0
    err_at_grasp = float("nan")
    success = 0
    end_time = MAX_EPISODE_TIME
    prev_action = ctrl.copy()
    rows = []

    n_steps = int(MAX_EPISODE_TIME / sim.dt)
    for k in range(n_steps):
        t = k * sim.dt

        # ---- carrier motion
        if weld_on:
            frac = min(1.0, t / travel_time)
            d.mocap_pos[sim.carrier_mocap] = S + (G - S) * frac

        # ---- voice command
        if released:
            voice = "release"
        elif stop_window and stop_window[0] <= t < stop_window[1]:
            voice = "stop"
        elif t >= t_ready:
            voice = "ready"
        else:
            voice = "idle"

        # ---- perception (option A: pose from the simulator)
        opos = d.qpos[qa:qa + 3].copy()
        oquat = d.qpos[qa + 3:qa + 7].copy()
        ovel = d.qvel[sim.obj_v[obj]:sim.obj_v[obj] + 3].copy()
        hand_pos, hand_R = sim.hand_pose()
        to_hand = hand_pos - opos
        dist = float(np.linalg.norm(to_hand))
        if not detected and voice == "ready" and dist < DETECT_DIST and float(ovel @ to_hand) > 0:
            detected = True
            predicted = classify(features)

        wrist = d.qpos[sim.jq["wrist_rot"]]
        pred_period = SYMMETRY.get(predicted) if predicted else None
        if period is None:
            ideal = float("nan")
            werr = 0.0
        else:
            ideal = nearest_equivalent(wrist, q_star, period, wlo, whi)
            werr = abs(wrap_err(wrist, q_star, period))  # modulo the period: ideal may sit at a joint limit

        gp = sim.grasp_point(obj)
        grasp_dist = float(np.linalg.norm(opos - gp))
        total_f, finger_f = sim.contact_forces(obj)
        max_force = max(max_force, total_f)
        contact = total_f > 0.05

        # ---- phase and scripted controller
        if voice == "stop":
            phase = "stopped"
            # freeze: keep the current command
        elif released:
            phase = "releasing"
            grip_cmd_target = 0.0
        elif voice != "ready" or not detected:
            phase = "waiting"
        else:
            # wrist target from the predicted object (ball or no symmetry: stay)
            if pred_period is None:
                w_target = ctrl[2]
            else:
                w_target = nearest_equivalent(ctrl[2], q_star + wrist_noise, pred_period, wlo, whi)
            reach_target = np.array(q_target[:2]) + target_noise

            if weld_on and grasp_dist > GRASP_TRIGGER_DIST and t_grasp is None:
                aligned_now = period is None or werr < ALIGN_TOL
                phase = "reaching" if aligned_now else "aligning"
            elif weld_on:
                phase = "grasping"
                if t_touch is None:
                    grip_cmd_target = GRIP_CLOSED
                if t_grasp is None:
                    t_grasp = t
                    err_at_grasp = werr
            else:
                phase = "holding"

            # stop closing once the fingers touch: freeze the grip with a small squeeze
            if t_grasp is not None and t_touch is None and finger_f > 0.2:
                t_touch = t
                grip_cmd_target = min(ctrl[3] + SQUEEZE, GRIP_CLOSED)

            step = lambda cur, tgt, rate: cur + np.clip(tgt - cur, -rate * sim.dt, rate * sim.dt)
            ctrl[0] = step(ctrl[0], reach_target[0], reach_rate)
            ctrl[1] = step(ctrl[1], reach_target[1], reach_rate)
            ctrl[2] = step(ctrl[2], w_target, wrist_rate)

        ctrl[3] = ctrl[3] + np.clip(grip_cmd_target - ctrl[3], -grip_rate * sim.dt, grip_rate * sim.dt)
        if voice != "stop":
            ctrl[:3] += rng.normal(0, 0.003, 3)
        for i, a in enumerate(ACTUATORS):
            lo, hi = m.actuator_ctrlrange[sim.act[a]]
            ctrl[i] = np.clip(ctrl[i], lo, hi)
        d.ctrl[:] = ctrl

        # ---- hand-over: carrier lets go once the fingers have closed on the object
        if weld_on and t_touch is not None and t - t_touch >= LET_GO_DELAY:
            weld_on = False
            d.eq_active[sim.weld[obj]] = 0
            t_weld_off = t

        held = (not weld_on) and finger_f > 0.2 and grasp_dist < 0.06 and not released
        if held:
            held_time += sim.dt

        # ---- reward
        r = 0.0
        moving = float(np.abs(d.qvel[[sim.jdof[j] for j in ARM_JOINTS]]).sum())
        if voice == "ready":
            if period is not None:
                r += REWARD["align"] * (1 - min(werr, math.pi / 2) / (math.pi / 2))
            r += -REWARD["approach"] * dist
        elif moving > 0.1 and not released:
            r += REWARD["no_command"]
        aligned = period is None or werr < ALIGN_TOL
        if contact and aligned and total_f < FORCE_LIMIT / 2:
            r += REWARD["gentle_touch"]
        if t_grasp is not None and not grasp_bonus_given and contact and aligned:
            r += REWARD["grasp"]
            grasp_bonus_given = True
        if held:
            r += REWARD["hold"]
        if total_f > FORCE_LIMIT:
            r += REWARD["impact"]
        r += REWARD["jerk"] * float(np.abs(ctrl - prev_action).sum())
        for j in ARM_JOINTS:
            lo, hi = sim.jrange[j]
            q = d.qpos[sim.jq[j]]
            if q - lo < math.radians(2) or hi - q < math.radians(2):
                r += REWARD["joint_limit"]
        if ctrl[3] > 0.3 and not contact and not released:
            r += REWARD["early_grasp"]
        prev_action = ctrl.copy()

        # ---- log
        rel = hand_R.T @ (opos - hand_pos)
        acc = sim.sensor("thumb_acc")
        gyr = sim.sensor("thumb_gyro")
        rows.append([
            ep, k, round(t, 4), voice, phase, obj,
            *np.round(opos, 5), *np.round(oquat, 5), round(float(np.linalg.norm(ovel)), 5),
            *np.round(rel, 5),
            *np.round(features, 5),
            int(detected), predicted,
            *[round(float(d.qpos[sim.jq[j]]), 5) for j in ARM_JOINTS],
            *[round(float(d.qvel[sim.jdof[j]]), 5) for j in ARM_JOINTS],
            round(float(d.qpos[sim.grip_q]), 5),
            *np.round(acc, 4), *np.round(gyr, 4),
            *np.round(ctrl, 5),
            int(contact), round(total_f, 4), round(finger_f, 4), int(held),
            round(ideal, 5) if not math.isnan(ideal) else "",
            round(werr, 5), round(r, 5), "",
        ])

        # ---- physics
        mujoco.mj_step(m, d, nstep=sim.substeps)
        if viewer is not None:
            viewer.sync()
            time.sleep(sim.dt)
            if not viewer.is_running():
                break

        # ---- episode end conditions
        if not released and held_time >= HOLD_TIME_FOR_SUCCESS:
            success = int(math.isnan(err_at_grasp) or err_at_grasp < ALIGN_TOL or period is None)
            released = True
            t_release = t
        if released and t - t_release > 0.6:
            end_time = t
            break
        if t_weld_off is not None and not released and held_time == 0 and t - t_weld_off > 0.5:
            end_time = t  # object dropped right after hand-over
            break

    for row in rows:
        row[-1] = success
    summary = [
        ep, obj, predicted, success,
        round(t_grasp, 3) if t_grasp is not None else "",
        round(max_force, 3),
        round(err_at_grasp, 4) if not math.isnan(err_at_grasp) else "",
        int(stop_window is not None), round(end_time, 3),
    ]
    return rows, summary


# ----------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene", default="scene.xml")
    ap.add_argument("--episodes", type=int, default=20)
    ap.add_argument("--rate", type=float, default=25.0, help="samples per second")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="dataset")
    ap.add_argument("--view", action="store_true", help="watch episodes in the viewer (slow)")
    args = ap.parse_args()

    sim = ArmSim(args.scene, args.rate)
    rng = np.random.default_rng(args.seed)
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)

    viewer = None
    if args.view:
        import mujoco.viewer
        viewer = mujoco.viewer.launch_passive(sim.m, sim.d)

    n_success = 0
    start = time.time()
    with open(f"{args.out}_steps.csv", "w", newline="") as fs, \
         open(f"{args.out}_episodes.csv", "w", newline="") as fe:
        ws, we = csv.writer(fs), csv.writer(fe)
        ws.writerow(STEP_COLUMNS)
        we.writerow(EPISODE_COLUMNS)
        for ep in range(args.episodes):
            rows, summary = run_episode(sim, ep, rng, viewer)
            ws.writerows(rows)
            we.writerow(summary)
            n_success += summary[3]
            if (ep + 1) % max(1, args.episodes // 20) == 0 or ep == args.episodes - 1:
                print(f"episode {ep + 1}/{args.episodes}  success so far {n_success / (ep + 1):.0%}"
                      f"  ({time.time() - start:.0f}s)")
            if viewer is not None and not viewer.is_running():
                break

    print(f"Done. Wrote {args.out}_steps.csv and {args.out}_episodes.csv")


if __name__ == "__main__":
    main()
