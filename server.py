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

from arm_env import ArmEnv
from generate_dataset import EPISODE_COLUMNS, STEP_COLUMNS

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
    renderer = mujoco.Renderer(env.sim.m, 480, 640)
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
    obs, _ = env.reset(seed=int(time.time()), options={"live": True})
    ident = {"name": None}
    success, t_success, t_release, max_force = False, None, None, 0.0

    async def identify():  # TabPFN takes seconds on CPU: start at reset, the UI shows it only after detection
        if tabpfn["clf"] is not None:
            pred = await asyncio.to_thread(tabpfn["clf"].predict, [list(env.features)])
            ident["name"], ident["by"] = str(pred[0]), "TabPFN-3.5"
    asyncio.create_task(identify())

    for k in range(MAX_LIVE_SECONDS * FPS):
        t0 = time.perf_counter()
        if cmd["start"]:
            break  # user restarted
        voice = cmd["voice"]
        env.manual_voice = "ready" if voice == "release" else voice
        if voice == "stop":
            action = np.zeros(4)  # safety layer: "stop" freezes every joint, whatever the policy says
        elif voice == "release":
            action = np.array([0, 0, 0, -1.0])  # open the hand, keep the arm still
            t_release = t_release or k / FPS
        else:
            action = policy(env, obs)
        obs, _, terminated, _, _ = env.step(action)
        max_force = max(max_force, env.total_f)
        if terminated and not success:
            success, t_success = True, k / FPS

        renderer.update_scene(env.sim.d, camera="demo")
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
            "max_force": round(max_force, 1),
        }))
        if t_release is not None and k / FPS - t_release > 1.0:
            break
        await asyncio.sleep(max(0.0, 1 / FPS - (time.perf_counter() - t0)))
    await ws.send_text(json.dumps({"episode_over": True, "success": success}))
