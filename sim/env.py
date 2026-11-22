"""
Gymnasium environment on the new simulator (sim/build.py: Kinova Gen3 arm + Aero Hand Open).
Same task as the old arm_env.py (spec sections 3-6): an object is handed over, the arm must
detect it, turn the wrist to suit it, reach, grasp and hold it for 1 s, only after "ready".
Grasps are physical: no grasp assist.

Action (5, each -1..1): hand velocity x, y, z (world frame), wrist turn rate, grip rate.
Inverse kinematics (sim/ik.py) turns the hand target into the 7 joint targets each step.

  python -m sim.env      # self-check: Gymnasium API, one random episode, expert success, speed
"""

import math

import gymnasium as gym
import mujoco
import numpy as np

from common import (
    ALIGN_TOL, DETECT_DIST, GRASP_RADIUS, HOLD_TIME_FOR_SUCCESS, MAX_EPISODE_TIME, OBJECTS, REWARD,
    ROUNDNESS, SYMMETRY, TRUE_DIMS, classify, mat_to_quat, nearest_equivalent, wrap_err,
)
from sim.build import ARM_JOINTS, grip_ctrl, load_model
from sim.camera import DepthCamera, Tracker
from sim.ik import SITE, hand_frame, solve

RATE_HZ = 25.0
SHOULDER = np.array([0.0, -0.22, 1.40])  # sim/scene.xml "shoulder" site
FORCE_LIMIT = 150.0              # N, summed contact force above this = hard impact. A normal hold of this hand
                                 # sums 54-124 N over its many finger contacts (old hand: 20 N)
MAX_VEL = 0.5                    # m/s hand speed at action = +-1
MAX_WRIST = 3.0                  # rad/s
MAX_GRIP = 3.0                   # grip units (0 open .. 1 closed) per s (2.0 / 1.5 tested: fewer successes, more force)
WRIST_RANGE = (-1.6, 1.6)        # rad, wrist turn the targets may ask for
WORKSPACE = (np.array([0.05, -0.55, 0.75]), np.array([0.62, 0.10, 1.45]))  # hand target box (m)
REGRASP_BELOW = 0.5              # TabPFN hold chance under this at first touch: open and grasp again...
MAX_REGRASPS = 1                 # ...at most this often...
REGRASP_TIME = 0.4               # s ...the hand staying open this long while the person keeps holding
LOOK_EVERY = 0.2                 # s, TabPFN re-predicts the stop point this often while the object moves
LET_GO_GRIP = 0.6               # the person lets go once the hand is this closed on the object...
LET_GO_DELAY = 0.15              # ...for this long after the fingers first touch it. Tested and dropped (100 fresh
                                 # episodes): waiting for the fingers to settle (82 vs 81 %, more force); the person
                                 # stopping pushing at first touch (66 %: pressing it into the palm helps the grasp)
HELD_DIST = 0.10                 # m, object within this of the grasp point counts as held
                                 # (0.06 scored 6 of 100 real holds as drops: this hand holds things off-centre)
GRIP_HOLD = 0.3                  # grip above this counts as "closing" (grasp bonus, early-grasp penalty)
EARLY_GRASP_DIST = 0.10          # m, closing farther than this from the object is penalised
SUCCESS_BONUS = 200.0
STOP_ACTION = np.zeros(5, np.float32)                          # "stop": freeze everything
RELEASE_ACTION = np.array([0, 0, 0, 0, -1.0], np.float32)      # "release": open the hand, keep the arm still
VOICES = ["idle", "ready", "stop"]
# Where the object sits in the hand frame when held (measured in a grasp test; see docs/images/grasps.png): along the palm
# normal (x) and the fingers (z), from the hand's grasp site
HOLD_OFFSET = {o: np.array([-0.015 if o == "pen" else 0.0, 0.0, -0.03]) for o in OBJECTS}
# Object orientation in the hand frame when held: long axis across the palm (hand y)
A_LONG = np.column_stack([[0, 0, 1], [1, 0, 0], [0, 1, 0]]).astype(float)   # cup, bottle, pen: z along hand y
A_BOX = np.column_stack([[0, 1, 0], [0, 0, 1], [1, 0, 0]]).astype(float)    # box: long x along y, thin z to palm
OBS_SIZE = 3 + 3 + 2 + 2 + 6 + 3 + 4 + 3 + 3 + 5 + 2 + 1 + 3 + 2


def wrist_angle(R):
    """The hand's wrist turn from its grasp-site frame R (0 = handshake; inverse of ik.hand_frame)."""
    v = hand_frame(0.0).T @ R[:, 1]
    return math.atan2(v[0], v[1])


class ArmEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, goal_predictor=None, goal_noise=None, hold_predictor=None, camera=False):
        m = self.m = load_model()
        self.d = mujoco.MjData(m)
        self.ik_d = mujoco.MjData(m)  # kinematics-only copy for inverse kinematics
        self.substeps = int(round(1.0 / (RATE_HZ * m.opt.timestep)))
        self.dt = self.substeps * m.opt.timestep
        self.n_steps = int(MAX_EPISODE_TIME / self.dt)
        self.qadr = [m.joint(j).qposadr[0] for j in ARM_JOINTS]
        self.dofs = [m.joint(j).dofadr[0] for j in ARM_JOINTS]
        self.arm_act = [m.actuator(j).id for j in ARM_JOINTS]
        self.site = m.site(SITE).id
        self.grip_q = m.joint("hand_right_index_mcp_flex").qposadr[0]  # finger closure (rad)
        self.carrier = m.body_mocapid[m.body("carrier").id]
        self.obj_body = {o: m.body(o).id for o in OBJECTS}
        self.obj_q = {o: m.joint(f"{o}_free").qposadr[0] for o in OBJECTS}
        self.obj_v = {o: m.joint(f"{o}_free").dofadr[0] for o in OBJECTS}
        self.weld = {o: m.equality(f"hold_{o}").id for o in OBJECTS}
        names = [m.body(b).name for b in range(m.nbody)]
        self.hand_bodies = {b for b, n in enumerate(names) if n.startswith("hand_")}
        self.arm_bodies = self.hand_bodies | {b for b, n in enumerate(names) if n.startswith("arm_")}
        self.finger_bodies = {b for b in self.hand_bodies if any(f in names[b] for f in
                                                                 ("index", "middle", "ring", "pinky", "thumb"))}
        self.action_space = gym.spaces.Box(-1.0, 1.0, (5,), np.float32)
        self.observation_space = gym.spaces.Box(-np.inf, np.inf, (OBS_SIZE,), np.float32)
        self.goal_predictor = goal_predictor  # (pos, vel, acc) -> (stop point, interval); see trajectory.py
        self.goal_noise = None if goal_noise is None else np.load(goal_noise)  # rows: distance left, error xyz
        self.hold_predictor = hold_predictor  # TabPFN classifier: P(hold) at first touch; see grasp_outcome.py
        self.eye = DepthCamera(m) if camera else None  # camera=True: detection and TabPFN see the object in pixels

    # ------------------------------------------------------------------ helpers
    def _ik(self, pos, wrist):
        """Joint targets putting the hand at pos with that wrist turn, starting from the current targets."""
        self.ik_d.qpos[:] = self.d.qpos
        self.ik_d.qpos[self.qadr] = self.q_cmd
        q, _ = solve(self.m, self.ik_d, self.qadr, self.dofs, pos, hand_frame(wrist), iters=12)
        return q

    def _set_ctrl(self, grip):
        self.d.ctrl[self.arm_act] = self.q_cmd
        for name, v in grip_ctrl(self.m, grip).items():
            self.d.ctrl[self.m.actuator(name).id] = v

    def hold_pose(self, obj, hand_pos, hand_R):
        """Object position and rotation when held by a hand at (hand_pos, hand_R)."""
        A = A_BOX if obj == "box" else A_LONG
        return hand_pos + hand_R @ HOLD_OFFSET[obj], hand_R @ A

    def _update_giver(self, t):
        """Pose the visual person: hand behind the object (or pulling back after letting go), torso
        following at arm's reach, elbow from two-segment arm geometry."""
        m, d = self.m, self.d
        obj = self.carrier_at(t) if self.weld_on else self.giver_hand_at_release
        back = self.S - self.G
        back[2] = 0.0
        back /= max(np.linalg.norm(back), 1e-9)
        hand = obj + back * (GRASP_RADIUS[self.obj] + 0.035)
        if not self.weld_on:  # let go: the hand withdraws
            hand = hand + back * min(0.25, 0.8 * (t - self.t_release))
        torso = np.array([hand[0], hand[1], 1.15]) + back * 0.42
        side = np.cross([0.0, 0.0, 1.0], back)
        shoulder = torso + [0, 0, 0.21] + side * 0.17
        L1, L2 = 0.30, 0.28
        v = hand - shoulder
        dist = float(np.clip(np.linalg.norm(v), 0.05, L1 + L2 - 1e-3))
        n = v / max(np.linalg.norm(v), 1e-9)
        hand = shoulder + n * dist
        a = (L1 * L1 - L2 * L2 + dist * dist) / (2 * dist)
        pole = np.array([0.0, 0.0, -1.0]) + 0.5 * side
        pole -= (pole @ n) * n
        elbow = shoulder + a * n + np.sqrt(max(L1 * L1 - a * a, 0.0)) * pole / max(np.linalg.norm(pole), 1e-9)
        yaw = np.zeros(4)
        mujoco.mju_axisAngle2Quat(yaw, [0.0, 0.0, 1.0], math.atan2(-back[1], -back[0]))
        poses = {"giver_torso": (torso, yaw), "giver_upper": ((shoulder + elbow) / 2, elbow - shoulder),
                 "giver_fore": ((elbow + hand) / 2, hand - elbow), "giver_hand": (hand, yaw)}
        for name, (pos, rot) in poses.items():
            mid = m.body_mocapid[m.body(name).id]
            d.mocap_pos[mid] = pos
            if len(rot) == 3:  # a segment: capsule z axis along it
                q = np.zeros(4)
                mujoco.mju_quatZ2Vec(q, rot)
                rot = q
            d.mocap_quat[mid] = rot

    def carrier_at(self, t):
        """Where the person's hand holds the object at time t: a quadratic arc S -> C -> G, timed by a
        minimum-jerk profile (how people move: smooth start, smooth slow-down before stopping)."""
        if self.t_carrier0 is None:  # live mode before "ready": still at the start
            return self.S
        tau = min(1.0, max(0.0, (t - self.t_carrier0) / self.travel_time))
        u = tau ** 3 * (10 - 15 * tau + 6 * tau * tau)
        return (1 - u) ** 2 * self.S + 2 * (1 - u) * u * self.C + u * u * self.G

    def sensor(self, name):
        return self.d.sensor(name).data.copy()

    # ------------------------------------------------------------------ reset
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        rng, m, d = self.np_random, self.m, self.d
        mujoco.mj_resetData(m, d)
        mujoco.mj_resetData(m, self.ik_d)  # same seed -> same start: IK starts from the same joints
        for o in OBJECTS:
            d.eq_active[self.weld[o]] = 0

        self.obj = obj = OBJECTS[rng.integers(len(OBJECTS))]
        self.period = SYMMETRY[obj]

        # arm start: hand in front of the body, forearm forward, wrist turned a little
        self.q_cmd = np.zeros(7)
        self.p_cmd = np.array([rng.uniform(0.15, 0.30), rng.uniform(-0.30, -0.15), rng.uniform(0.95, 1.10)])
        self.w_cmd = rng.uniform(-0.5, 0.5)
        self.grip = 0.0
        self.q_cmd = solve(m, self.ik_d, self.qadr, self.dofs, self.p_cmd, hand_frame(self.w_cmd), iters=300)[0]
        d.qpos[self.qadr] = self.q_cmd
        self._set_ctrl(0.0)

        # hand-over: where the hand should be (reachable by construction), and the true wrist target
        self.q_star = rng.uniform(math.radians(-80), math.radians(80))
        # people hand things over at about arm's length: on a shell 44-52 cm from the shoulder, in front
        az, el = math.radians(rng.uniform(-25, 15)), math.radians(rng.uniform(-45, -10))
        self.hand_target = SHOULDER + rng.uniform(0.44, 0.52) * np.array(
            [math.cos(el) * math.cos(az), math.cos(el) * math.sin(az), math.sin(el)])
        hR = hand_frame(self.q_star)
        self.G, Robj = self.hold_pose(obj, self.hand_target, hR)
        spin = rng.uniform(0, 2 * math.pi) if obj in ("cup", "bottle", "pen") else 0.0
        if obj != "box":  # turn about its own long axis (or anywhere, for the ball): looks the same
            axis = Robj[:, 2]
            q = np.zeros(4)
            mujoco.mju_axisAngle2Quat(q, axis, spin)
            Rs = np.zeros(9)
            mujoco.mju_quat2Mat(Rs, q)
            Robj = Rs.reshape(3, 3) @ Robj
        quat = mat_to_quat(Robj)

        # the object comes from in front, toward the palm
        direction = 0.8 * np.array([1.0, 0.0, 0.0]) + 0.6 * hR[:, 0] + rng.normal(0, 0.2, 3)
        direction /= np.linalg.norm(direction)
        self.S = self.G + direction * rng.uniform(0.4, 0.7)
        # a person's hand-over: an arc (lifted, swung a little sideways) that slows down smoothly at the end
        side = np.cross(self.G - self.S, [0.0, 0.0, 1.0])
        side /= max(np.linalg.norm(side), 1e-9)
        self.C = (self.S + self.G) / 2 + [0, 0, rng.uniform(0.03, 0.18)] + side * rng.uniform(-0.12, 0.12)
        self.travel_time = 1.4 * np.linalg.norm(self.G - self.S) / rng.uniform(0.15, 0.35)
        self.qa = self.obj_q[obj]
        d.qpos[self.qa:self.qa + 3] = self.S
        d.qpos[self.qa + 3:self.qa + 7] = quat
        d.mocap_pos[self.carrier] = self.S
        d.mocap_quat[self.carrier] = quat
        d.eq_active[self.weld[obj]] = 1
        mujoco.mj_forward(m, d)

        h, w, dd = TRUE_DIMS[obj]
        f = np.array([h, w, dd]) * (1 + rng.normal(0, 0.05, 3))
        self.features = (*f, f.max() / f.min(), float(np.clip(ROUNDNESS[obj] + rng.normal(0, 0.12), 0, 1)))

        self.t_ready = rng.uniform(0.2, 1.0)
        self.stop_window = None
        if rng.random() < 0.05:
            t0 = rng.uniform(self.t_ready + 0.3, self.t_ready + 2.0)
            self.stop_window = (t0, t0 + 0.8)
        self.live = bool((options or {}).get("live"))
        self.manual_voice = "idle" if self.live else None
        self.t_carrier0 = None if self.live else 0.0

        self.G_seen, self.G_range, self.G_pred = self.G, None, None
        self.t_look, self.looks, self.settled = -1e9, 0, False
        self.k = 0
        self.detected = False
        self.predicted = ""
        self.weld_on = True
        self.t_touch = None
        self.err_at_touch = float("nan")
        self.held_time = 0.0
        self.held = False
        self.grasp_bonus_given = False
        self.gentle_bonus_given = False
        self.expert_closing = False
        self.voice = "idle"
        self.prev_action = np.zeros(5)
        self.t_release, self.giver_hand_at_release = None, None
        self.hold_ps, self.regrasps, self.regrasp_until = [], 0, -1.0
        self.tracker, self.cam_rng = Tracker(), np.random.default_rng(seed)  # own stream: same episode with/without camera
        self._update_giver(0.0)
        mujoco.mj_kinematics(m, d)
        self._update_state()
        return self._obs(), {}

    # ------------------------------------------------------------------- step
    def step(self, action):
        m, d = self.m, self.d
        t = self.k * self.dt
        if self.weld_on and self.t_carrier0 is not None:
            d.mocap_pos[self.carrier] = self.carrier_at(t)
        self._update_giver(t)

        a = np.clip(np.asarray(action, dtype=float), -1, 1)
        self.p_cmd = np.clip(self.p_cmd + a[:3] * MAX_VEL * self.dt, *WORKSPACE)
        self.w_cmd = float(np.clip(self.w_cmd + a[3] * MAX_WRIST * self.dt, *WRIST_RANGE))
        if t < self.regrasp_until:  # re-grasp reflex: the hand opens while the person keeps holding it
            a[4] = -1.0
        self.grip = float(np.clip(self.grip + a[4] * MAX_GRIP * self.dt, 0.0, 1.0))
        self.q_cmd = self._ik(self.p_cmd, self.w_cmd)
        self._set_ctrl(self.grip)
        mujoco.mj_step(m, d, nstep=self.substeps)
        self.k += 1
        t = self.k * self.dt

        if self.manual_voice is not None:
            self.voice = self.manual_voice
            if self.t_carrier0 is None and self.voice == "ready":
                self.t_carrier0 = t
        elif self.stop_window and self.stop_window[0] <= t < self.stop_window[1]:
            self.voice = "stop"
        else:
            self.voice = "ready" if t >= self.t_ready else "idle"

        self._update_state()

        # hand-over: the person lets go once the hand has closed on the object for a moment
        if self.weld_on and self.t_touch is None and self.finger_f > 0.2 and t >= self.regrasp_until:
            self.t_touch = t
            self.err_at_touch = self.werr
            if self.hold_predictor is not None:  # TabPFN: will this grasp hold? If not, open and try again
                from grasp_outcome import touch_features
                p = float(self.hold_predictor.predict_proba([touch_features(self)])[0, 1])
                self.hold_ps.append(p)
                if p < REGRASP_BELOW and self.regrasps < MAX_REGRASPS:
                    self.regrasps += 1
                    self.regrasp_until = t + REGRASP_TIME
                    self.t_touch, self.expert_closing = None, False
        if (self.weld_on and self.t_touch is not None and t - self.t_touch >= LET_GO_DELAY
                and self.grip >= LET_GO_GRIP and self.finger_f > 0.2):
            self.weld_on = False
            d.eq_active[self.weld[self.obj]] = 0
            self.t_release, self.giver_hand_at_release = t, self.carrier_at(t)

        self.held = (not self.weld_on) and self.finger_f > 0.2 and self.grasp_dist < HELD_DIST
        if self.held:
            self.held_time += self.dt

        # reward: same terms and weights as the old env (common.REWARD)
        r = 0.0
        if self.voice == "ready":
            arriving = self.t_carrier0 is not None and t - self.t_carrier0 < self.travel_time
            if self.period is not None and self.t_touch is None and arriving:
                r += REWARD["align"] * (1 - min(self.werr, math.pi / 2) / (math.pi / 2))
            if self.weld_on:
                r -= REWARD["approach"] * self.grasp_dist
        elif self.arm_speed > 0.1:
            r += REWARD["no_command"]
        aligned = self.period is None or self.werr < ALIGN_TOL
        if self.contact and aligned and self.total_f < FORCE_LIMIT / 2 and not self.gentle_bonus_given:
            r += REWARD["gentle_touch"]
            self.gentle_bonus_given = True
        if not self.grasp_bonus_given and self.grip > GRIP_HOLD and self.contact and aligned:
            r += REWARD["grasp"]
            self.grasp_bonus_given = True
        if self.held:
            r += REWARD["hold"]
        if self.total_f > FORCE_LIMIT:
            r += REWARD["impact"]
        r += REWARD["jerk"] * float(np.abs(a - self.prev_action).sum()) * 0.25
        self.prev_action = a
        if (self.grip > GRIP_HOLD and not self.contact and self.grasp_dist > EARLY_GRASP_DIST
                and self.t_touch is None):
            r += REWARD["early_grasp"]

        terminated, success = False, False
        if self.held_time >= HOLD_TIME_FOR_SUCCESS:
            terminated = True
            success = self.period is None or self.err_at_touch < ALIGN_TOL
            r += SUCCESS_BONUS if success else 0.0
        truncated = self.k >= self.n_steps
        info = {"success": success, "object": self.obj, "predicted": self.predicted}
        return self._obs(), float(r), terminated, truncated, info

    # ----------------------------------------------------------------- expert
    def expert_action(self):
        """Scripted controller with privileged info (the hand-over pose); for warm-starting the agent."""
        a = np.zeros(5, np.float32)
        a[4] = -1.0
        if self.voice != "ready" or not self.detected:
            return a
        a[:3] = np.clip((self.hand_target - self.p_cmd) / (MAX_VEL * self.dt), -1, 1)
        period = SYMMETRY.get(self.predicted)
        if period is not None:
            target = nearest_equivalent(self.w_cmd, self.q_star, period, *WRIST_RANGE)
            a[3] = np.clip((target - self.w_cmd) / (MAX_WRIST * self.dt), -1, 1)
        if self.grasp_dist < 0.03 or self.t_touch is not None:
            self.expert_closing = True
        a[4] = 1.0 if self.expert_closing else -1.0
        return a

    # ---------------------------------------------------------------- state
    def contact_forces(self):
        """Normal force object-arm/hand and object-fingers (N)."""
        m, d = self.m, self.d
        total, fingers = 0.0, 0.0
        f6 = np.zeros(6)
        ob = self.obj_body[self.obj]
        for i in range(d.ncon):
            c = d.contact[i]
            b1, b2 = m.geom_bodyid[c.geom1], m.geom_bodyid[c.geom2]
            if ob in (b1, b2):
                other = b2 if b1 == ob else b1
                if other in self.arm_bodies:
                    mujoco.mj_contactForce(m, d, i, f6)
                    total += abs(f6[0])
                    if other in self.finger_bodies:
                        fingers += abs(f6[0])
        return total, fingers

    def _update_state(self):
        d = self.d
        self.opos = d.qpos[self.qa:self.qa + 3].copy()
        vel = d.qvel[self.obj_v[self.obj]:self.obj_v[self.obj] + 3].copy()
        self.oacc = (vel - self.ovel) / self.dt if self.k else np.zeros(3)
        self.ovel = vel
        self.hand_pos = d.site_xpos[self.site].copy()
        self.hand_R = d.site_xmat[self.site].reshape(3, 3).copy()
        self.hand_vel = np.zeros(6)
        mujoco.mj_objectVelocity(self.m, d, mujoco.mjtObj.mjOBJ_SITE, self.site, self.hand_vel, 0)
        self.arm_speed = float(np.abs(d.qvel[self.dofs]).sum())
        self.wrist = wrist_angle(self.hand_R)
        grasp_point, _ = self.hold_pose(self.obj, self.hand_pos, self.hand_R)
        self.grasp_dist = float(np.linalg.norm(self.opos - grasp_point))
        to_hand = self.hand_pos - self.opos
        self.dist = float(np.linalg.norm(to_hand))
        # what the arm knows about the object: the simulator's truth, or (camera=True) what the depth camera sees
        t = self.k * self.dt
        if self.eye is None:
            self.seen = (self.opos, self.ovel, self.oacc)
        elif self.weld_on and self.t_touch is None:  # after touch the fingers take over: no frames needed
            self.tracker.add(t, self.eye.see(d, self.obj_body[self.obj], self.cam_rng, min(self.features[1:3]) / 2))  # radius from shape features
            self.seen = self.tracker.estimate(t)
        if (not self.detected and self.voice == "ready" and self.seen is not None
                and np.linalg.norm(self.hand_pos - self.seen[0]) < DETECT_DIST
                and float(self.seen[1] @ (self.hand_pos - self.seen[0])) > 0):
            self.detected = True
            self.predicted = classify(self.features)
            if self.goal_predictor is not None or self.goal_noise is not None:
                self.G_seen = self.seen[0].copy()  # never the true point; where it is until the first look
        # where it will stop: TabPFN looks again every LOOK_EVERY s while it moves (position, velocity,
        # acceleration: the arc and the slow-down); in training, the true point + TabPFN's real error
        stop_speed = 0.08 if self.eye else 0.02  # the camera's speed noise is ~6 cm/s: below 8 it may stand still
        moving = self.seen is not None and not self.settled and float(np.linalg.norm(self.seen[1])) >= stop_speed
        if (self.detected and self.weld_on and self.t_touch is None and moving and self.t_carrier0 is not None
                and t - self.t_carrier0 >= 0.2 and t - self.t_look >= LOOK_EVERY - 1e-9):
            self.t_look, self.looks = t, self.looks + 1
            if self.goal_predictor is not None:
                self.G_seen, self.G_range = self.goal_predictor(*self.seen)
                self.G_pred = self.G_seen.copy()
            elif self.goal_noise is not None:
                left = float(np.linalg.norm(self.G - self.opos))  # only to pick an error of the right size
                near = np.flatnonzero(np.abs(self.goal_noise[:, 0] - left) < 0.05)
                pool = near if len(near) else np.arange(len(self.goal_noise))
                self.G_seen = self.G + self.goal_noise[self.np_random.choice(pool), 1:]
        if self.detected and self.weld_on and self.t_touch is None and not moving and self.looks and self.seen is not None:
            self.G_seen = self.seen[0].copy()  # stopped: seen, not predicted
            self.settled = self.eye is not None  # camera: once stopped, speed noise must not restart the looks
                                                 # (it did: a 0.19 m/s phantom sent a look 20 cm off)
        self.werr = 0.0 if self.period is None else abs(wrap_err(self.wrist, self.q_star, self.period))
        self.total_f, self.finger_f = self.contact_forces()
        self.contact = self.total_f > 0.05

    def _obs(self):
        d, R = self.d, self.hand_R
        if self.detected:
            rel_pos = R.T @ (self.opos - self.hand_pos)
            rel_quat = mat_to_quat(R.T @ d.xmat[self.obj_body[self.obj]].reshape(3, 3))
            rel_vel = R.T @ self.ovel
            rel_goal = R.T @ (self.G_seen - self.hand_pos)
        else:
            rel_pos, rel_quat, rel_vel, rel_goal = np.zeros(3), np.zeros(4), np.zeros(3), np.zeros(3)
        onehot = np.array([float(self.predicted == o) for o in OBJECTS])
        pred_period = SYMMETRY.get(self.predicted) if self.predicted else None
        wrist_target = [0.0, 0.0] if pred_period is None else [wrap_err(self.wrist, self.q_star, pred_period), 1.0]
        return np.concatenate([
            self.hand_pos - SHOULDER,                              # hand relative to the shoulder
            self.hand_vel[3:],                                     # hand linear velocity
            [self.wrist, self.w_cmd], [self.grip, float(d.qpos[self.grip_q])],
            self.sensor("thumb_acc") / 10.0, self.sensor("thumb_gyro"),
            rel_pos, rel_quat, rel_vel, rel_goal,
            onehot, wrist_target,
            [float(self.detected)],
            [float(self.voice == v) for v in VOICES],
            [float(self.contact), self.finger_f / FORCE_LIMIT],
        ]).astype(np.float32)


if __name__ == "__main__":
    import time
    from gymnasium.utils.env_checker import check_env
    env = ArmEnv()
    check_env(env, skip_render_check=True)
    obs, _ = env.reset(seed=0)
    total, done, steps, t0 = 0.0, False, 0, time.time()
    while not done:
        obs, r, term, trunc, info = env.step(env.action_space.sample())
        assert np.all(np.isfinite(obs)) and math.isfinite(r)
        total, steps, done = total + r, steps + 1, term or trunc
    print(f"OK: random episode, {info['object']}, {steps} steps, return {total:.1f}, "
          f"{steps / (time.time() - t0):.0f} steps/s")
    wins = 0
    for ep in range(20):
        env.reset(seed=10_000 + ep)
        done = False
        while not done:
            _, _, term, trunc, info = env.step(env.expert_action())
            done = term or trunc
        wins += info["success"]
    print(f"expert {wins}/20")
