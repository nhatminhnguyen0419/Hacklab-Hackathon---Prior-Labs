"""
Web app backend (spec section 9, minimum version): FastAPI + one WebSocket.

  python -m uvicorn server:app --port 8000      then open http://localhost:8000

- POST /upload   CSV dataset (per-step or per-episode file) -> column check + statistics
- WS   /ws/sim   live episode: JPEG frames from MuJoCo's "demo" camera + JSON state;
                 the browser sends "start" / "ready" / "release" / "stop"
The arm is driven by the trained agent (models/ppo_arm.zip), or by the scripted controller
when started with ARM_CONTROLLER=scripted or when the model file is missing. The identified object comes from TabPFN-3.5 (falls back to the rule).
"""

import asyncio
import io
import json
import math
import os
import threading
import time

import mujoco
import numpy as np
import pandas as pd
from fastapi import FastAPI, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image

from sim.env import FORCE_LIMIT, RELEASE_ACTION, STOP_ACTION, ArmEnv  # the new simulator
from generate_dataset import EPISODE_COLUMNS, STEP_COLUMNS
from grasp_outcome import touch_features

MODEL_PATH = "models/ppo_arm.zip"
FPS = 25
MAX_LIVE_SECONDS = 30
app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")  # chart.js kept local: demo works offline

# ---------------------------------------------------------------- classifier (loaded once, in background)
tabpfn = {"clf": None, "status": "loading"}


def _load_tabpfn():
    try:
        import torch
        from classifier import load_classifier
        tabpfn["clf"] = load_classifier(fast=not torch.cuda.is_available())
        tabpfn["status"] = "ready"
    except Exception as e:  # missing license/package: the app still runs with the rule classifier
        tabpfn["status"] = f"unavailable ({type(e).__name__})"


threading.Thread(target=_load_tabpfn, daemon=True).start()

# "What happens next": P(stable hold) predicted at first touch (grasp_outcome.py)
grasp = {"clf": None, "status": "loading"}


def _load_grasp():
    try:
        from grasp_outcome import FEATURES, load_predictor
        clf = load_predictor()
        clf.predict_proba(np.zeros((1, len(FEATURES))))  # warm-up: the first call is ~15x slower
        grasp["clf"], grasp["status"] = clf, "ready"
    except Exception as e:  # missing data file/package: the app still runs without the prediction
        grasp["status"] = f"unavailable ({type(e).__name__})"


threading.Thread(target=_load_grasp, daemon=True).start()

# "Where it's going": stop point predicted at detection (trajectory.py); the agent uses it
traj = {"predict": None, "status": "loading"}


def _load_traj():
    try:
        from trajectory import load_predictor
        traj["predict"], traj["status"] = load_predictor(), "ready"
    except Exception as e:  # without it the agent falls back to the simulator's true stop point
        traj["status"] = f"unavailable ({type(e).__name__})"


threading.Thread(target=_load_traj, daemon=True).start()


def load_policy():
    # RL by default: it beats the scripted controller (92% vs 80%). ARM_CONTROLLER=scripted to switch back.
    if os.environ.get("ARM_CONTROLLER") != "scripted" and os.path.exists(MODEL_PATH):
        from stable_baselines3 import PPO
        model = PPO.load(MODEL_PATH, device="cpu")
        return "RL agent (PPO)", lambda env, obs: model.predict(obs, deterministic=True)[0]
    return "scripted controller", lambda env, obs: env.expert_action()


# ---------------------------------------------------------------- pages and upload
@app.get("/")
def index():
    return FileResponse("index.html")


@app.post("/upload")
async def upload(file: UploadFile):
    try:
        df = pd.read_csv(io.BytesIO(await file.read()))
    except Exception as e:
        return {"ok": False, "error": f"not a readable CSV: {e}"}
    cols = set(df.columns)
    if set(STEP_COLUMNS) <= cols:
        kind = "per-step"
        # one row per episode: the last step holds the final success flag; add per-episode stats
        g = df.groupby("episode_id")
        ep = g.last()[["object_type", "success"]].join(g["contact_force"].max().rename("max_contact_force"))
        ep["time_to_grasp"] = df[df.phase == "grasping"].groupby("episode_id")["timestamp"].min()
        ep["detected"] = g["object_detected"].max()
    elif set(EPISODE_COLUMNS) <= cols:
        kind = "per-episode"
        ep = df.set_index("episode_id")
        ep["detected"] = ep["predicted_object_type"].notna().astype(int)
    else:
        return {"ok": False, "error": "columns do not match spec section 8",
                "missing_step_columns": sorted(set(STEP_COLUMNS) - cols)[:10]}

    per_obj = ep.groupby("object_type").agg(episodes=("success", "size"), successes=("success", "sum"))
    return {
        "ok": True, "kind": kind, "rows": len(df), "episodes": len(ep),
        "success_rate": round(float(ep["success"].mean()), 3),
        "detection_rate": round(float(ep["detected"].mean()), 3),
        "mean_time_to_grasp": round(float(pd.to_numeric(ep["time_to_grasp"], errors="coerce").mean()), 2),
        "mean_max_force": round(float(ep["max_contact_force"].mean()), 1),
        "per_object": {o: {k: int(v) for k, v in r.items()} for o, r in per_obj.iterrows()},
    }


# ---------------------------------------------------------------- live simulation
@app.websocket("/ws/sim")
async def sim_socket(ws: WebSocket):
    await ws.accept()
    policy_name, policy = load_policy()
    env = ArmEnv()
    renderer = mujoco.Renderer(env.m, 480, 640)
    cmd = {"voice": "idle", "start": False}

    async def receive():
        while True:
            msg = (await ws.receive_text()).strip().lower()
            if msg == "start":
                cmd["start"] = True
            elif msg in ("ready", "release", "stop"):
                cmd["voice"] = msg

    receiver = asyncio.create_task(receive())
    try:
        while True:
            # wait for "start", then run one live episode
            while not cmd["start"]:
                await asyncio.sleep(0.05)
            cmd.update(start=False, voice="idle")
            await run_live_episode(ws, env, renderer, policy, policy_name, cmd)
    except (WebSocketDisconnect, RuntimeError):
        pass
    finally:
        receiver.cancel()
        renderer.close()


async def run_live_episode(ws, env, renderer, policy, policy_name, cmd):
    env.goal_predictor = traj["predict"]
    obs, _ = env.reset(seed=time.time_ns() % 2**31, options={"live": True})  # ns: two episodes in one second differ
    marker = env.m.site("goal_pred").id
    env.m.site_pos[marker] = [0, 0, -5]  # hidden until TabPFN has predicted
    ident = {"name": None}
    success, t_success, t_release, max_force = False, None, None, 0.0

    async def identify():  # TabPFN takes seconds on CPU: start at reset, the UI shows it only after detection
        if tabpfn["clf"] is not None:
            pred = await asyncio.to_thread(tabpfn["clf"].predict, [list(env.features)])
            ident["name"], ident["by"] = str(pred[0]), "TabPFN-3.5"
    asyncio.create_task(identify())
    hold = {"p": None, "asked": False}

    async def predict_hold(row):
        hold["p"] = float(await asyncio.to_thread(lambda: grasp["clf"].predict_proba([row])[0, 1]))

    for k in range(MAX_LIVE_SECONDS * FPS):
        t0 = time.perf_counter()
        if cmd["start"]:
            break  # user restarted
        voice = cmd["voice"]
        env.manual_voice = "ready" if voice == "release" else voice
        if voice == "stop":
            action = STOP_ACTION  # safety layer: "stop" freezes every joint, whatever the policy says
        elif voice == "release":
            action = RELEASE_ACTION  # open the hand, keep the arm still
            t_release = t_release or k / FPS
        else:
            action = policy(env, obs)
        obs, _, terminated, _, info = env.step(action)
        if not env.weld_on:  # the hand's own grip, after the person lets go (their push at hand-over peaks 300-500 N)
            max_force = max(max_force, env.total_f)
        if env.G_range is not None:
            env.m.site_pos[marker] = env.G_pred
        if env.t_touch is not None and not hold["asked"] and grasp["clf"] is not None:
            hold["asked"] = True  # the row is taken now, at first touch; the answer arrives ~0.1 s later
            asyncio.create_task(predict_hold(touch_features(env)))
        if terminated and info["success"] and not success:  # a 1 s hold with the wrist misaligned is not a success
            success, t_success = True, k / FPS

        renderer.update_scene(env.d, camera="demo")
        renderer.scene.flags[mujoco.mjtRndFlag.mjRND_SHADOW] = 0  # shadows doubled the frame time: live view fell to half speed
        buf = io.BytesIO()
        Image.fromarray(renderer.render()).save(buf, format="JPEG", quality=75)
        await ws.send_bytes(buf.getvalue())
        await ws.send_text(json.dumps({
            "t": round(k / FPS, 2), "voice": voice, "controller": policy_name,
            "object": env.obj if env.detected else None,
            # until TabPFN answers, show the rule classifier's result (it picks the wrist rule anyway)
            "identified": (ident["name"] or env.predicted) if env.detected else None,
            "identified_by": ident.get("by", "rule"), "tabpfn": tabpfn["status"],
            "detected": env.detected, "held": bool(env.held),
            "success": success,
            "wrist_error_deg": None if math.isnan(env.err_at_touch) else round(math.degrees(env.err_at_touch), 1),
            "time_to_grasp": None if env.t_touch is None or env.t_carrier0 is None
            else round(env.t_touch - env.t_carrier0, 2),
            "max_force": round(max_force, 1), "force_limit": FORCE_LIMIT,
            "hold_p": None if hold["p"] is None else round(hold["p"], 3), "grasp_model": grasp["status"],
            # stop point: TabPFN's guess vs where the object really stops (the simulator knows), and the 10-90% range
            "goal_err_cm": None if env.G_range is None else round(100 * float(np.linalg.norm(env.G_pred - env.G)), 1),
            "goal_range_cm": None if env.G_range is None else round(100 * (env.G_range[1] - env.G_range[0]), 1),
            "arrive_in": None if env.G_range is None or env.t_carrier0 is None
            else round(max(0.0, env.travel_time - (env.k * env.dt - env.t_carrier0)), 1),
            "traj_model": traj["status"],
        }))
        if t_release is not None and k / FPS - t_release > 1.0:
            break
        await asyncio.sleep(max(0.0, 1 / FPS - (time.perf_counter() - t0)))
    await ws.send_text(json.dumps({"episode_over": True, "success": success}))
