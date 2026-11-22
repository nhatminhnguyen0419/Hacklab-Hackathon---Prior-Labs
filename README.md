# Prosthetic Arm × TabPFN

**How far can a table take a prosthetic arm?**
A simulated robot arm and hand for someone who has lost an arm at the shoulder. It sees an object being handed over, **TabPFN-3.5 predicts what it is, where it will stop and whether the grasp will hold**, and a reinforcement-learning agent reaches and grasps it, only after the user says “ready”.

Built at **HackLab Berlin × TabPFN** (Prior Labs track) by **The Big Gang**: Minh Nguyen, Quan Vo, Kazuki Dozono.

![A person hands a bottle to the prosthetic arm](docs/images/hero.png)

**One hand-over, step by step** (the pink dot is TabPFN's predicted stop point):

![Four steps of a hand-over](docs/images/sequence.png)

**All five objects held by contact physics alone, no grasp assist:**

![The hand holding a cup, bottle, box, pen and ball](docs/images/grasps.png)

**Or seen only through a depth camera at the user's head** (`--camera`): the 3D view, and what the arm sees, with the object in pink:

![A hand-over seen through the depth camera](docs/images/camera.png)

---

## What it does

1. **Ready:** nothing moves until the user says “ready” (browser speech recognition, with buttons as a fallback). “Stop” freezes every joint; “release” opens the hand.
2. **Detect:** an object counts as detected within 50 cm of the hand while moving toward it.
3. **What is it?** TabPFN classifies the object (cup, bottle, box, pen, ball) from five noisy shape features.
4. **Where is it going?** The person moves the object along a curve and slows down before stopping. Every 0.2 s while it moves, TabPFN predicts from its position, velocity and acceleration where it will stop, with a 10–90% range. A pink marker shows the prediction in the 3D view.
5. **Reach and grasp:** a PPO agent moves the hand, turns the wrist to suit the object and closes the grip. It acts on **TabPFN’s predicted stop point, never the simulator’s true one**.
6. **What happens next?** The moment the fingers touch, TabPFN predicts the chance that the grasp ends in a stable hold. **If it says under 50%, the hand opens and grasps again** while the person keeps holding the object.

Every robot step, detection and grasp is a row in a table, and TabPFN learns from past rows in context, with no training run.

## Results

All numbers are on held-out data that the models never saw (fixed evaluation seeds for the agent).

**TabPFN predictions**

| Question | TabPFN-3.5 | Baselines | Held-out set |
|---|---|---|---|
| What is it? | **100%** accuracy | hand-written rule 98.3% | 600 episodes |
| Where will it stop? | **0.6 cm** median error, 3.8 cm at the 90th percentile | linear regression 3.6 / 9.1 cm; physics (constant braking) 4.6 / 77 cm; straight line to arm’s length 8.7 / 22.2 cm | 300 hand-overs, 1,500 looks |
| Will the grasp hold? | **ROC AUC 0.961**, Brier 0.055, accuracy 0.936 | logistic regression 0.837 / 0.109 / 0.839; wrist-angle rule 0.551 / 0.161 / 0.839 | 598 grasps |

The stop-point error shrinks as the object approaches: 2.5 cm with more than 25 cm to go, 0.1 cm in the last 10 cm. Its 10–90% range is a little too narrow (holds the truth 71% of the time, ideally 80%). Live, one combined TabPFN model answers all three directions in about 0.1 s (median 0.8 cm). The hold prediction is well calibrated (says 98% → 97% held; 70% → 72% held).

**The arm** (grasp success, 200 held-out episodes, real contact physics, no grasp assist)

| Controller | Success |
|---|---|
| Scripted expert (knows the hand-over pose) | 80% (16/20) |
| Agent imitating the expert, on TabPFN’s stop point | 72% (144/200) |
| Agent after 1M PPO steps, on TabPFN’s stop point | 89% (178/200) |
| **+ re-grasp when TabPFN says the grasp is risky** | **92% (184/200)** |

Of 21 re-grasps, 8 turned a drop into a hold, 2 spoiled a grasp that would have held, 11 changed nothing (`regrasp_eval.py`: the same episodes with the reflex off and on).

**Seeing it with a camera** (`--camera`). The numbers above use the object's position as the simulator knows it. With a depth camera worn at the user's head instead (`sim/camera.py`: 240×180 depth + segmentation at 25 Hz, depth noise growing with distance, position/velocity/acceleration fitted over the last 0.4 s), the arm sees the object only in pixels:

| | Perfect sensing | Depth camera |
|---|---|---|
| Camera position error | – | 1.1 cm median, 1.7 cm at the 90th percentile |
| TabPFN stop point | 0.6 cm | **2.1 cm** median (linear regression 4.6, straight line to arm's length 9.3); 1.2 cm in the last 10 cm |
| Agent on TabPFN's stop point | 89% (178/200) | **81% (162/200)**, same agent, not retrained for the camera |

TabPFN learns the camera's errors in context: it is trained on rows logged from camera views (`data/trajectory_camera.csv`), with the true stop point as the target.

Live predictions take about **0.1 s on a laptop CPU** (fast TabPFN checkpoint with a cached context); the arm waits for each one, because acting on a stale prediction cut success from 25/30 to 5/30.

## The simulator

- **MuJoCo** with real robot models from **[MuJoCo Menagerie](https://github.com/google-deepmind/mujoco_menagerie)**, attached in code (`sim/build.py`):
  - **Kinova Gen3** arm, 7 joints, scaled to 60% so shoulder-to-wrist is about 0.7 m like a human arm, hanging from the shoulder.
  - **TetherIA Aero Hand Open**, a tendon-driven humanoid hand (0.45 kg), closed as one prosthetic-style power-grip synergy, with the tendon motors capped at 40 N.
  - An IMU (accelerometer + gyro) in the thumb.
  - Optional depth camera at the user's head, like smart glasses (`--camera`): the object is found by segmentation, back-projected to 3D, and tracked.
- **Actions** (5): hand velocity x/y/z, wrist turn rate, grip rate. Damped-least-squares inverse kinematics (`sim/ik.py`) turns them into the 7 joint targets.
- **Hand-over:** a visible person brings the object along an arc to a point at about arm’s length from the shoulder, slowing down smoothly (a minimum-jerk profile, like human reaching), and lets go once the hand has closed on it.
- **Grasps are physical:** objects are held by contact forces only.
- **Re-grasp reflex:** a safety layer like “stop”: if TabPFN gives the grasp under 50% at first touch, the hand opens for 0.4 s and the agent grasps again (once).

## Quick start

Tested on Windows 11, CPU only, with Python 3.14, MuJoCo 4.0.1, Stable-Baselines3 2.9.2, Gymnasium 1.4.0 and TabPFN 10.2.0.

```bash
pip install -r requirements.txt
```

TabPFN downloads its model weights on first use and may ask you to accept its licence.

**Run the web app** (live simulation, voice commands, results, CSV upload):

```bash
python -m uvicorn server:app --port 8000
```

Open http://localhost:8000, press **New episode**, then **Ready** (or 🎤 Voice and say “ready”). Start it about a minute before a demo: the three TabPFN models load in the background (the header shows “TabPFN ready”). The page works offline. Tick **Depth camera** before **New episode** to let the arm find the object in a depth image; a picture-in-picture shows what it sees (object in pink). If the live demo fails, `demo_replays/` has five recorded episodes.

**Self-checks**

```bash
python sim/build.py        # builds the model, renders a preview image
python sim/ik.py           # inverse kinematics reaches the handshake pose
python -m sim.env          # Gymnasium API check, random episode, expert success, speed
```

**Evaluate the agent**

```bash
python train.py --eval models/ppo_arm.zip --episodes 200 --tabpfn-goal   # acting on TabPFN's stop point
python train.py --eval models/ppo_arm.zip --episodes 200                 # given the true stop point
```

**TabPFN experiments**

```bash
python classifier.py                    # object ID vs the rule
python trajectory.py --collect 1200     # log a row per look while objects approach -> data/trajectory.csv
python trajectory.py                    # stop-point prediction vs baselines (saves the error vectors)
python -m sim.camera                    # depth-camera tracking error vs the simulator's truth
python trajectory.py --camera --collect 1200 && python trajectory.py --camera   # the same, seen by the camera
python train.py --eval models/ppo_arm.zip --episodes 200 --tabpfn-goal --camera
python regrasp_eval.py                  # the re-grasp reflex off vs on, same 200 episodes
python grasp_outcome.py --collect 2400  # log grasps at first touch -> data/grasp_touch.csv
python grasp_outcome.py                 # hold prediction vs baselines, with calibration
python tabpfn_generate.py --n 500       # TabPFN synthetic shape data, train-on-synthetic / test-on-real
```

**Train the agent** (imitation of the scripted expert, then PPO; about 30 minutes with 12 processes)

```bash
python train.py --bc 300 --steps 1000000 --envs 12 --goal-noise data/trajectory_residuals.npy --out models/v3/ppo_arm
```

`--goal-noise` adds TabPFN’s real held-out stop-point errors during training, so the agent learns to act on predictions. Copy the result to `models/ppo_arm.zip` for the web app.

**Record demo videos**

```bash
python record_demo.py --n 5 --rl
```

## Project layout

| Path | What it is |
|---|---|
| `sim/scene.xml`, `sim/objects.xml` | Floor, torso, camera, the five objects and the hand-over carrier |
| `sim/build.py` | Loads the Menagerie arm and hand, scales and attaches them, adds the IMU, grip synergy |
| `sim/ik.py` | Inverse kinematics for the hand |
| `sim/camera.py` | Depth camera: segmentation, back-projection, quadratic-fit tracker |
| `sim/env.py` | Gymnasium environment: hand-over, voice timeline, detection, rewards, scripted expert, TabPFN hooks |
| `sim/menagerie/` | Kinova Gen3 and TetherIA Aero Hand Open models, with their licences |
| `trajectory.py` | TabPFN: where the object will stop |
| `grasp_outcome.py` | TabPFN: will the grasp hold |
| `classifier.py` | TabPFN: which object |
| `tabpfn_generate.py` | TabPFN synthetic data for the shape features |
| `train.py` | Imitation + PPO training and evaluation |
| `server.py`, `index.html`, `static/` | Web app (FastAPI + one page, Chart.js served locally) |
| `record_demo.py` | Records MP4 replays with the predicted-stop marker |
| `make_pitch_video.py` | Renders the pitch video `demo_replays/pitch_demo.mp4` (needs ffmpeg) |
| `docs/TECHNICAL.md` | Technical documentation: architecture, models, evaluation protocol, problems and fixes |
| `docs/VIDEO_SCRIPT.md`, `make_broll.py`, `make_final_video.py` | The 3-minute submission video: timed script, 2:20 of footage, and the tool that stitches in the team's camera clips and voice-over |
| `make_pictures.py` | Renders the README pictures in `docs/images/` |
| `regrasp_eval.py` | Measures the re-grasp reflex: the same episodes with it off and on |
| `common.py` | Shared definitions: objects, the rule classifier, reward weights, dataset CSV columns |
| `models/ppo_arm.zip` | The trained agent used by the web app (`models/v3/` has the imitation model and the training log) |
| `data/` | Logged rows for the TabPFN models |

## Limitations

- The hand-over is scripted: the person is a visual figure whose hand follows a generated arc, and lets go once the hand has closed.
- With `--camera`, detection and TabPFN's inputs come from the depth camera; the agent's close-range view of the object (relative position in its observation), the shape features and contact sensing still come from the simulator. The web app has a camera switch; the recorded videos run without the camera. Fine-tuning the agent on the camera's errors made it worse (77%), so the camera numbers use the original agent.
- The 7-joint arm’s elbow sometimes swings out sideways.
- The live view runs at about 0.7× real time on a laptop CPU, because the arm waits for each TabPFN look (shadows are off in the live stream; recorded videos keep them).
- “Held” means the object is in finger contact and within 10 cm of the grasp point; the cup is the hardest object (about 75–80%).
- `data/run1_*` come from our first, homemade simulator (in the git history): the object classifier still learns from `data/run1_features.csv` (shape features don't depend on the simulator), and `data/run1_episodes.csv` is a sample for the web app’s CSV upload. A per-step dataset export for the new simulator is not written yet.

## Credits and licences

- **TabPFN** by [Prior Labs](https://priorlabs.ai): tabular foundation model, used through the `tabpfn` and `tabpfn-extensions` packages.
- **[MuJoCo](https://mujoco.org)** and **[MuJoCo Menagerie](https://github.com/google-deepmind/mujoco_menagerie)** (Google DeepMind). The Kinova Gen3 model is © Kinova inc. under its BSD-style licence; the TetherIA Aero Hand Open model is under Apache-2.0. Both licence files are in `sim/menagerie/`.
- **[Stable-Baselines3](https://github.com/DLR-RM/stable-baselines3)** (PPO), **[Gymnasium](https://gymnasium.farama.org)**, **FastAPI**, **Chart.js**.
