# Technical documentation

How the prosthetic-arm simulator, the reinforcement-learning agent and the three TabPFN-3.5 models fit together, how each result was measured, and what we learned on the way. The README has the quick start and the headline numbers; this file has the details.

## 1. System overview

```
           person's hand moves the object along an arc, slowing down (minimum-jerk)
                                        │
                ┌───────────────────────┴────────────────────────┐
                │ MuJoCo scene (sim/scene.xml + sim/build.py)     │
                │ Kinova Gen3 (x0.6) + TetherIA Aero Hand Open    │
                └───────┬───────────────────────────────┬────────┘
       object state     │ (truth, or depth camera:       │ contacts, finger forces,
       25 Hz            │  sim/camera.py)                │ wrist angle, IMU
                        ▼                                ▼
   ┌─────────────────────────────┐        ┌─────────────────────────────┐
   │ TabPFN: what is it?         │        │ TabPFN: will it hold?        │
   │ classifier.py (shape)       │        │ grasp_outcome.py, at touch   │
   │ TabPFN: where will it stop? │        │ P(hold) < 0.5 → re-grasp     │
   │ trajectory.py, every 0.2 s  │        └──────────────┬──────────────┘
   └──────────────┬──────────────┘                       │
                  ▼ predicted stop point + object type   │
   ┌─────────────────────────────────────────────┐       │
   │ PPO agent (train.py, models/ppo_arm.zip)     │◄──────┘ (env opens the hand)
   │ action: hand velocity xyz, wrist rate, grip   │
   └──────────────┬──────────────────────────────┘
                  ▼ damped-least-squares IK (sim/ik.py) → 7 joint targets
```

Everything runs on a laptop CPU. The control loop runs at 25 Hz (`sim/env.py RATE_HZ`), physics at 500 Hz (timestep 2 ms, 20 substeps).

## 2. Simulator (`sim/`)

| Piece | Where | Details |
|---|---|---|
| Arm | `sim/build.py` | Kinova Gen3 from MuJoCo Menagerie, 7 joints, every length and mass scaled by 0.6 (≈0.7 m shoulder to wrist), hanging from a shoulder site; position actuators. |
| Hand | `sim/build.py` | TetherIA Aero Hand Open (tendon-driven). One grip value 0..1 drives all finger tendons and the thumb together (a prosthetic-style power-grip synergy); tendon force capped at `GRIP_FORCE` = 40 N. |
| IK | `sim/ik.py` | Damped least squares on position + orientation of `hand_grasp_site`; 12 iterations per control step from the previous solution. |
| Hand-over | `sim/env.py carrier_at` | Start S, control point C, stop G: a quadratic Bezier S→C→G, timed by the minimum-jerk profile 10τ³−15τ⁴+6τ⁵. G is on a shell 44–52 cm from the shoulder (people hand things over at about arm's length). A weld holds the object to the person's hand until the robot hand has closed on it for 0.15 s. |
| Person | `sim/env.py _update_giver` | Mocap bodies; torso follows the hand, elbow from two-link geometry. Visual only, no collisions. |
| Objects | `sim/objects.xml` | cup, bottle, box, pen, ball. Real contact physics: held by friction alone, no grasp assist. |
| Depth camera | `sim/camera.py` | See section 4. |

**Episode** (`ArmEnv.reset/step`): random object, random start pose, random hand-over target and wrist angle; "ready" comes at 0.2–1.0 s (5% of episodes also get a "stop" window). Success = held for 1 s with the wrist within 15° of the object's symmetry (the ball has none). "Held" = finger contact and the object within 10 cm of the grasp point.

**Observation** (42 numbers): hand position and velocity, wrist angle and command, grip command and finger closure, thumb IMU, object position/orientation/velocity relative to the hand, the stop point the arm believes in (`G_seen`) relative to the hand, predicted object one-hot, wrist error to the predicted object's symmetry, detected flag, voice one-hot, contact flag and finger force.

**Action** (5, each −1..1): hand velocity xyz (≤0.5 m/s), wrist rate (≤3 rad/s), grip rate.

**Reward** (`common.REWARD`): wrist alignment while the object arrives, approach, gentle aligned first touch, grasp, per-step hold, impact above 150 N, jerk, closing too early, moving without a command; +200 on success.

## 3. TabPFN models

All three use TabPFN-3.5 through the `tabpfn` package. They learn **in context** from logged rows: no training run, no hyper-parameter search. Live, they use the fast checkpoint with `fit_mode="fit_with_cache"`, so each prediction is one forward pass.

### 3.1 What is it? (`classifier.py`)
- Rows: five noisy shape features (height, width, depth, elongation, roundness) from `data/run1_features.csv`.
- Result: 100% on 600 held-out episodes vs 98.3% for the hand-written rule (`common.classify`).
- `tabpfn_generate.py` also tests TabPFN as a synthetic-data generator (train on synthetic, test on real).

### 3.2 Where will it stop? (`trajectory.py`)
- One row per **look** (every 0.2 s while the object moves after detection): position, speed, direction, acceleration in the direction-of-travel frame, and two physics hints (`s_shell`: where a straight path reaches arm's length; `s_brake`: stop distance at the current braking).
- Targets: stop point minus current position, in the travel frame (ahead, side, up). The travel frame makes the problem nearly rotation-invariant, which helps with a few thousand rows.
- Evaluation: held-out **hand-overs** (whole episodes, so no leakage between looks of one hand-over); 3,000 context looks, ~1,500 test looks; three separate regressors; the 10–90% quantiles of "ahead" give the range.
- Live: one stacked model (each look becomes 3 rows plus a categorical "which direction" column), `n_estimators=2`, 600 looks of context → ~0.1 s per look. The arm **waits** for each look: answering from a stale background prediction dropped live success from 25/30 to 5/30.

| Sensing | TabPFN | Linear regression | Physics (constant braking) | Straight to arm's length | Range coverage (ideal 80%) |
|---|---|---|---|---|---|
| Simulator truth | **0.6 cm** median, 3.8 cm p90 | 3.6 / 9.1 cm | 4.6 / 77 cm | 8.7 / 22.2 cm | 71% |
| Depth camera | **2.1 cm** median, 6.1 cm p90 | 4.6 / 9.7 cm | 66.7 / 86.6 cm | 9.3 / 24.3 cm | 72% |

### 3.3 Will it hold? (`grasp_outcome.py`)
- One row per grasp at the moment the fingers first touch: predicted object, wrist error, distance to the grasp point, total and finger force, finger closure, grip command, object speed, object position in the hand frame, arm speed. Label: did the episode end in a stable, aligned hold.
- 2,400 episodes with a mix of controllers (deterministic agent, stochastic agent, scripted expert) so both holds and failures occur; 75/25 stratified split.
- Result: ROC AUC **0.961**, Brier 0.055 (logistic regression 0.837 / 0.109, wrist-angle rule 0.551 / 0.161). Calibrated: predicted 98% → 97% held, 70% → 72% held.
- **Acting on it**: if P(hold) < 0.5 at first touch, the environment opens the hand for 0.4 s while the person keeps holding, and the agent grasps again (once). `regrasp_eval.py`: same 200 episodes, reflex off vs on: 89% → 92%; of 21 re-grasps, 8 saved a drop, 2 spoiled a hold, 11 changed nothing.

## 4. Depth camera (`sim/camera.py`)

The camera is worn at the user's head (`eye` in `sim/scene.xml`, 60° vertical field of view), like smart glasses.

1. Each control step (25 Hz) MuJoCo renders a 240×180 **depth** image and a **segmentation** image (which geom each pixel shows). Sites are hidden and anti-aliasing is off for this renderer (see section 7).
2. The object's pixels are back-projected to 3D with the pinhole model; depth noise σ = 2 mm × depth² (stereo-like).
3. The camera only sees the near surface, so the centre of those points is pushed back along the line of sight by ⅔ of the object's radius (exact for a sphere; the radius comes from the shape features).
4. `Tracker` fits a quadratic per axis to the last 0.4 s of positions → position, velocity and acceleration now. Frames with fewer than 8 object pixels are skipped (hidden by a hand).
5. With `ArmEnv(camera=True)`, detection and TabPFN's stop-point inputs use these estimates instead of the simulator's truth; TabPFN's camera model is trained on rows logged the same way (`python trajectory.py --camera --collect 1200`), with the true stop point as the target, so it learns the camera's errors in context.

Tracking error over 5 hand-overs (`python -m sim.camera`): position 1.1 cm median, 1.7 cm at the 90th percentile.

Still from the simulator in camera mode: the agent's relative object pose in its observation (its close-range view), shape features and contact sensing.

## 5. The agent (`train.py`)

1. **Behaviour cloning** from 300 episodes of the scripted expert (which knows the hand-over pose): 72% alone.
2. **PPO** (Stable-Baselines3, MLP policy, n_steps 1024, batch 256, γ 0.99), 1M steps on 12 processes, about 30 minutes on CPU.
3. **Prediction-aware training**: `--goal-noise data/trajectory_residuals.npy` replaces the true stop point with truth + a TabPFN held-out error of the same distance-left, so the agent learns to act on predictions rather than on the simulator's answer.

## 6. Evaluation protocol

- Agent: 200 episodes with fixed seeds 10000–10199, deterministic policy, TabPFN in the loop (`--tabpfn-goal`). Seed ranges are kept apart: evaluation 10 000+, behaviour cloning 20 000+, demo 50 000+, grasp rows 100 000+, trajectory rows 200 000+.
- TabPFN models: split by episode, never by row.

| Controller | Success |
|---|---|
| Scripted expert (knows the hand-over pose) | 80% (16/20) |
| Imitation only, on TabPFN's stop point | 72% (144/200) |
| PPO, on TabPFN's stop point | 89% (178/200) |
| + re-grasp when TabPFN says the grasp is risky | 92% (184/200) |
| PPO with the depth camera, on TabPFN's camera stop point | 81% (162/200) |

## 7. Problems we hit (and the fix)

| Problem | Cause | Fix |
|---|---|---|
| Live success fell from 25/30 to 5/30 | TabPFN ran in a background thread; the agent acted on answers up to a second old | Synchronous, smaller stacked model (~0.1 s); `env.step` in a worker thread so the WebSocket stays alive |
| Grip force readings of 300–500 N | The person's push at hand-over was counted | Count force only after the person lets go |
| Camera crash in MuJoCo's renderer | Anti-aliasing blends segmentation colours at object edges into invalid IDs | Camera renderer with `offsamples = 0` |
| Camera never saw some objects (success 58–62%) | The translucent reach-zone *site* is drawn solid in segmentation and depth, hiding objects behind it | Hide all site groups in the camera's render; success 81% |
| Camera position biased 2.7 cm toward the camera | Only the near surface is visible | Push back by ⅔ radius along the ray: 1.1 cm |
| Live view at half speed | Shadows and a reflective floor | Shadows off in the live stream, plain floor |
| Camera mode 8 points below perfect sensing (81% vs 89%) | The agent was trained on stop-point errors of the truth-based model | Tried: 500k more PPO steps with the camera's errors. It got worse (77% with the camera, 77% without), so we kept the original agent. Open: retrain from the start with camera errors, or with the camera in the loop |

## 8. Data files (`data/`)

| File | Rows | Made by |
|---|---|---|
| `run1_features.csv` | shape features per episode (first simulator; shape doesn't depend on the simulator) | git history |
| `run1_episodes.csv` | per-episode sample for the web app's CSV upload | git history |
| `trajectory.csv` | one row per look, simulator truth | `trajectory.py --collect 1200` |
| `trajectory_camera.csv` | one row per look, seen by the depth camera | `trajectory.py --camera --collect 1200` |
| `trajectory_residuals.npy`, `trajectory_camera_residuals.npy` | held-out stop-point errors with distance left | `trajectory.py [--camera]` |
| `grasp_touch.csv` | one row per first touch | `grasp_outcome.py --collect 2400` |
| `tabpfn_synthetic.csv` | TabPFN-generated shape rows | `tabpfn_generate.py` |

## 9. Web app (`server.py`, `index.html`)

FastAPI with one WebSocket. The browser sends `start` (or `start camera`), `ready`, `release`, `stop`; the server streams a JPEG of the `demo` camera and a JSON state per frame (25 fps target), plus, in camera mode, the depth camera's picture with the object in pink. The three TabPFN models load in background threads at start-up; until then the app falls back to the rule classifier and the simulator's stop point. "Stop" and "release" override the policy (`STOP_ACTION`, `RELEASE_ACTION`): a safety layer the agent cannot overrule.
