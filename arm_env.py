"""
Gymnasium environment for the HackLab #2 prosthetic arm (spec sections 4-6).

Same scene, object hand-over, detection, classifier and REWARD weights as
generate_dataset.py, but the RL agent picks the 4 actions instead of the script.

Action (4, each -1..1): speed command for shoulder, elbow, wrist, grip.
Episode ends on success (object held 1 s) or after MAX_EPISODE_TIME.
Dropping the object does NOT end the episode: per-step rewards are mostly
negative, so ending early would otherwise look attractive to the agent.
"""

import math

import gymnasium as gym
import mujoco
import numpy as np

from generate_dataset import (
    ACTUATORS, ALIGN_TOL, ARM_JOINTS, DETECT_DIST, FORCE_LIMIT, GRASP_TRIGGER_DIST, GRIP_CLOSED,
    HOLD_TIME_FOR_SUCCESS, LET_GO_DELAY, MAX_EPISODE_TIME, OBJECTS, REWARD, ROUNDNESS, SQUEEZE,
    SYMMETRY, TRUE_DIMS,
    ArmSim, classify, grasp_local, mat_to_quat, nearest_equivalent, wrap_err,
)

SUCCESS_BONUS = 200.0                          # "stable hold: biggest bonus" (spec 6); must dominate every shaping term
GRIP_HOLD = 0.3                                # rad, grip command above this keeps the grasp assist on
EARLY_GRASP_DIST = 0.10                       # m, early_grasp penalty only when the object is farther than this
MAX_RATE = np.array([1.5, 1.5, 3.0, 3.5])      # rad/s at action = +-1 (shoulder, elbow, wrist, grip)
VOICES = ["idle", "ready", "stop"]             # "release" is never reached: success ends the episode
OBS_SIZE = 3 + 3 + 1 + 6 + 3 + 4 + 3 + 3 + 5 + 2 + 1 + 3 + 2


class ArmEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, scene_path="scene.xml", rate_hz=25.0):
        self.sim = ArmSim(scene_path, rate_hz)
        self.n_steps = int(MAX_EPISODE_TIME / self.sim.dt)
        self.action_space = gym.spaces.Box(-1.0, 1.0, (4,), np.float32)
        self.observation_space = gym.spaces.Box(-np.inf, np.inf, (OBS_SIZE,), np.float32)
        self.ctrl_lo = np.array([self.sim.m.actuator_ctrlrange[self.sim.act[a]][0] for a in ACTUATORS])
        self.ctrl_hi = np.array([self.sim.m.actuator_ctrlrange[self.sim.act[a]][1] for a in ACTUATORS])

    # ------------------------------------------------------------------ reset
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        rng, sim = self.np_random, self.sim
        m, d = sim.m, sim.d
        mujoco.mj_resetData(m, d)
        for o in OBJECTS:
            d.eq_active[sim.weld[o]] = 0
            d.eq_active[sim.grip_weld[o]] = 0
        self.assisted = False
        self.held = False

        # Object, arm start pose and hand-over point: same sampling as generate_dataset.run_episode
        self.obj = obj = OBJECTS[rng.integers(len(OBJECTS))]
        self.period = SYMMETRY[obj]
        q0 = [rng.uniform(-0.2, 0.3), rng.uniform(-0.3, 0.3), rng.uniform(-0.5, 0.5)]
        for j, q in zip(ARM_JOINTS, q0):
            d.qpos[sim.jq[j]] = q
        self.ctrl = np.array(q0 + [0.0])
        d.ctrl[:] = self.ctrl

        self.q_star = rng.uniform(math.radians(-80), math.radians(80))
        self.q_target = q_target = [rng.uniform(-0.4, 0.9), rng.uniform(-0.5, 1.2), self.q_star]
        hpos, hR = sim.forward_at(q_target)
        self.G = hpos + hR @ grasp_local(obj)

        align = np.array([[0, 1, 0], [0, 0, 1], [1, 0, 0]], float) if obj == "box" else np.eye(3)
        spin = rng.uniform(0, 2 * math.pi) if obj in ("cup", "bottle", "pen") else 0.0
        cs, sn = math.cos(spin), math.sin(spin)
        Rspin = np.array([[cs, -sn, 0], [sn, cs, 0], [0, 0, 1]])
        quat = mat_to_quat(hR @ Rspin if obj == "ball" else hR @ align @ Rspin)

        direction = hR[:, 1] + rng.normal(0, 0.25, 3)
        direction /= np.linalg.norm(direction)
        self.S = self.G + direction * rng.uniform(0.4, 0.7)
        self.travel_time = np.linalg.norm(self.G - self.S) / rng.uniform(0.15, 0.35)

        self.qa = sim.obj_q[obj]
        d.qpos[self.qa:self.qa + 3] = self.S
        d.qpos[self.qa + 3:self.qa + 7] = quat
        d.mocap_pos[sim.carrier_mocap] = self.S
        d.mocap_quat[sim.carrier_mocap] = quat
        d.eq_active[sim.weld[obj]] = 1
        mujoco.mj_forward(m, d)

        h, w, dd = TRUE_DIMS[obj]
        f = np.array([h, w, dd]) * (1 + rng.normal(0, 0.05, 3))
        self.features = (*f, f.max() / f.min(), float(np.clip(ROUNDNESS[obj] + rng.normal(0, 0.12), 0, 1)))

        self.t_ready = rng.uniform(0.2, 1.0)
        self.stop_window = None
        if rng.random() < 0.05:
            t0 = rng.uniform(self.t_ready + 0.3, self.t_ready + 2.0)
            self.stop_window = (t0, t0 + 0.8)

        # Live mode (web app): voice comes from the user via set_voice(), and the object only starts
        # approaching after the first "ready" (detection needs it moving toward the hand)
        self.live = bool((options or {}).get("live"))
        self.manual_voice = "idle" if self.live else None
        self.t_carrier0 = None if self.live else 0.0

        self.k = 0
        self.detected = False
        self.predicted = ""
        self.weld_on = True
        self.t_touch = None
        self.grip_at_touch = 0.0
        self.expert_closing = False
        self.err_at_touch = float("nan")
        self.held_time = 0.0
        self.grasp_bonus_given = False
        self.gentle_bonus_given = False
        self.voice = "idle"
        self._update_state()
        return self._obs(), {}

    # ------------------------------------------------------------------- step
    def step(self, action):
        sim, d = self.sim, self.sim.d
        t = self.k * sim.dt

        if self.weld_on and self.t_carrier0 is not None:
            frac = min(1.0, (t - self.t_carrier0) / self.travel_time)
            d.mocap_pos[sim.carrier_mocap] = self.S + (self.G - self.S) * frac

        prev_ctrl = self.ctrl.copy()
        a = np.clip(np.asarray(action, dtype=float), -1, 1)
        self.ctrl = np.clip(self.ctrl + a * MAX_RATE * sim.dt, self.ctrl_lo, self.ctrl_hi)
        d.ctrl[:] = self.ctrl
        mujoco.mj_step(sim.m, d, nstep=sim.substeps)
        self.k += 1
        t = self.k * sim.dt

        # voice timeline (same as the dataset; the agent only sees it), or the user's command in live mode
        if self.manual_voice is not None:
            self.voice = self.manual_voice
            if self.t_carrier0 is None and self.voice == "ready":
                self.t_carrier0 = t
        elif self.stop_window and self.stop_window[0] <= t < self.stop_window[1]:
            self.voice = "stop"
        else:
            self.voice = "ready" if t >= self.t_ready else "idle"

        self._update_state()

        # hand-over: carrier lets go LET_GO_DELAY after the fingers first touch
        if self.weld_on and self.t_touch is None and self.finger_f > 0.2:
            self.t_touch = t
            self.grip_at_touch = d.qpos[sim.grip_q]  # real finger angle: the command runs ~0.5 rad ahead
            self.err_at_touch = self.werr
        if self.weld_on and self.t_touch is not None and t - self.t_touch >= LET_GO_DELAY:
            self.weld_on = False
            d.eq_active[sim.weld[self.obj]] = 0
            # grasp assist (see objects.xml): only if the fingers are still closed on the object
            if self.finger_f > 0.2 and self.ctrl[3] > GRIP_HOLD:
                sim.attach(self.obj, True)
                self.assisted = True
        if self.assisted and self.ctrl[3] < GRIP_HOLD:  # hand opened ("release", or the agent let go)
            sim.attach(self.obj, False)
            self.assisted = False

        held = self.held = (not self.weld_on) and (
            self.assisted or (self.finger_f > 0.2 and self.grasp_dist < 0.06))
        if held:
            self.held_time += sim.dt

        # reward: same terms and weights as generate_dataset.py
        r = 0.0
        moving = float(np.abs(d.qvel[[sim.jdof[j] for j in ARM_JOINTS]]).sum())
        if self.voice == "ready":
            # align only guides the approach, until the object arrives: paid all episode it was worth
            # more (~77) than grasping
            arriving = self.t_carrier0 is not None and t - self.t_carrier0 < self.travel_time
            if self.period is not None and self.t_touch is None and arriving:
                r += REWARD["align"] * (1 - min(self.werr, math.pi / 2) / (math.pi / 2))
            if self.weld_on:  # after hand-over a dropped object lies ~1 m away: chasing it would punish every grasp try
                r -= REWARD["approach"] * self.dist
        elif moving > 0.1:
            r += REWARD["no_command"]
        aligned = self.period is None or self.werr < ALIGN_TOL
        if self.contact and aligned and self.total_f < FORCE_LIMIT / 2 and not self.gentle_bonus_given:
            r += REWARD["gentle_touch"]  # once per episode (spec: "bonus"); per step it paid to rest on the object
            self.gentle_bonus_given = True
        if not self.grasp_bonus_given and self.ctrl[3] > 0.3 and self.contact and aligned:
            r += REWARD["grasp"]
            self.grasp_bonus_given = True
        if held:
            r += REWARD["hold"]
        if self.total_f > FORCE_LIMIT:
            r += REWARD["impact"]
        r += REWARD["jerk"] * float(np.abs(self.ctrl - prev_ctrl).sum())
        for j in ARM_JOINTS:
            lo, hi = sim.jrange[j]
            q = d.qpos[sim.jq[j]]
            if q - lo < math.radians(2) or hi - q < math.radians(2):
                r += REWARD["joint_limit"]
        # Penalised everywhere, closing taught the agent to never close; near the object it must be free to try
        # only BEFORE the first touch (spec: closing before contact); after a drop it punished every try
        if (self.ctrl[3] > 0.3 and not self.contact and self.grasp_dist > EARLY_GRASP_DIST
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
        """The scripted controller from generate_dataset.py, as an action for this env.
        Uses privileged info (the hand-over pose); only for warm-starting the agent."""
        target = self.ctrl.copy()
        if self.voice == "ready" and self.detected:
            target[:2] = self.q_target[:2]
            period = SYMMETRY.get(self.predicted)
            if period is not None:
                wlo, whi = self.sim.jrange["wrist_rot"]
                target[2] = nearest_equivalent(self.ctrl[2], self.q_star, period, wlo, whi)
            if self.grasp_dist <= GRASP_TRIGGER_DIST:
                self.expert_closing = True
            if self.t_touch is not None:
                target[3] = min(self.grip_at_touch + SQUEEZE, GRIP_CLOSED)
            elif self.expert_closing:
                target[3] = GRIP_CLOSED
        return np.clip((target - self.ctrl) / (MAX_RATE * self.sim.dt), -1, 1).astype(np.float32)

    # ---------------------------------------------------------------- helpers
    def _update_state(self):
        sim, d = self.sim, self.sim.d
        self.opos = d.qpos[self.qa:self.qa + 3].copy()
        self.oquat = d.qpos[self.qa + 3:self.qa + 7].copy()
        self.ovel = d.qvel[sim.obj_v[self.obj]:sim.obj_v[self.obj] + 3].copy()
        self.hand_pos, self.hand_R = sim.hand_pose()
        to_hand = self.hand_pos - self.opos
        self.dist = float(np.linalg.norm(to_hand))
        if (not self.detected and self.voice == "ready" and self.dist < DETECT_DIST
                and float(self.ovel @ to_hand) > 0):
            self.detected = True
            self.predicted = classify(self.features)

        wrist = d.qpos[sim.jq["wrist_rot"]]
        wlo, whi = sim.jrange["wrist_rot"]
        self.werr = 0.0 if self.period is None else abs(
            wrist - nearest_equivalent(wrist, self.q_star, self.period, wlo, whi))
        self.grasp_dist = float(np.linalg.norm(self.opos - sim.grasp_point(self.obj)))
        self.total_f, self.finger_f = sim.contact_forces(self.obj)
        self.contact = self.total_f > 0.05

    def _obs(self):
        sim, d = self.sim, self.sim.d
        R = self.hand_R
        # Object info is only visible once detected (spec 3)
        if self.detected:
            rel_pos = R.T @ (self.opos - self.hand_pos)
            rel_quat = mat_to_quat(R.T @ d.xmat[sim.obj_body[self.obj]].reshape(3, 3))
            rel_vel = R.T @ self.ovel
            # ASSUMPTION: perception predicts where the approaching object will stop (the hand-over
            # point). Without it the agent only learned 0-3%: the expert it copies knows this point.
            rel_goal = R.T @ (self.G - self.hand_pos)
        else:
            rel_pos, rel_quat, rel_vel, rel_goal = np.zeros(3), np.zeros(4), np.zeros(3), np.zeros(3)
        # Classifier output and its wrist target (spec 5: "identified type and ideal wrist angle")
        onehot = np.array([float(self.predicted == o) for o in OBJECTS])
        pred_period = SYMMETRY.get(self.predicted) if self.predicted else None
        wrist = d.qpos[sim.jq["wrist_rot"]]
        if pred_period is None:
            wrist_target = [0.0, 0.0]          # no target yet, or ball
        else:
            wrist_target = [wrap_err(wrist, self.q_star, pred_period), 1.0]
        return np.concatenate([
            [d.qpos[sim.jq[j]] for j in ARM_JOINTS],
            [d.qvel[sim.jdof[j]] for j in ARM_JOINTS],
            [d.qpos[sim.grip_q]],
            sim.sensor("thumb_acc") / 10.0, sim.sensor("thumb_gyro"),
            rel_pos, rel_quat, rel_vel, rel_goal,
            onehot, wrist_target,
            [float(self.detected)],
            [float(self.voice == v) for v in VOICES],
            [float(self.contact), self.finger_f / FORCE_LIMIT],
        ]).astype(np.float32)


if __name__ == "__main__":
    # Self-check: Gymnasium API conformance plus one random episode
    from gymnasium.utils.env_checker import check_env
    env = ArmEnv()
    check_env(env, skip_render_check=True)
    obs, _ = env.reset(seed=0)
    total, steps, done = 0.0, 0, False
    while not done:
        obs, r, term, trunc, info = env.step(env.action_space.sample())
        assert np.all(np.isfinite(obs)) and math.isfinite(r)
        total, steps, done = total + r, steps + 1, term or trunc
    print(f"OK: random episode, {info['object']}, {steps} steps, return {total:.1f}")
    # The task must be solvable: the scripted expert reached 80/100 with the grasp assist (24/100 without)
    wins = 0
    for ep in range(20):
        env.reset(seed=10_000 + ep)
        done = False
        while not done:
            _, _, term, trunc, info = env.step(env.expert_action())
            done = term or trunc
        wins += info["success"]
    assert wins >= 12, f"expert only {wins}/20 - grasp assist broken?"
    print(f"OK: expert {wins}/20")
