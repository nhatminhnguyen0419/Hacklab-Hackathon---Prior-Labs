"""
Demo safety net (spec section 9): record a few successful episodes as MP4 videos
that play without the backend, e.g. if the live demo fails on stage.

  python record_demo.py                 # 3 successes with the scripted controller
  python record_demo.py --n 5 --rl      # 5 successes with the trained agent
"""

import argparse
import os

import cv2
import mujoco
import numpy as np

from arm_env import ArmEnv

FPS = 25


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=3, help="successful episodes to keep")
    ap.add_argument("--rl", action="store_true", help="use models/ppo_arm.zip instead of the script")
    ap.add_argument("--out", default="demo_replays")
    ap.add_argument("--max-tries", type=int, default=100)
    args = ap.parse_args()

    if args.rl:
        from stable_baselines3 import PPO
        model = PPO.load("models/ppo_arm.zip", device="cpu")
        act, who = (lambda env, obs: model.predict(obs, deterministic=True)[0]), "RL agent"
    else:
        act, who = (lambda env, obs: env.expert_action()), "scripted controller"

    os.makedirs(args.out, exist_ok=True)
    env = ArmEnv()
    renderer = mujoco.Renderer(env.sim.m, 480, 640)
    kept, seen = 0, set()
    for seed in range(args.max_tries):
        obs, _ = env.reset(seed=50_000 + seed)
        if env.obj in seen and len(seen) < 4:
            continue  # prefer a different object each time (ball never succeeds)
        frames, success, released_steps = [], False, 0
        for k in range(env.n_steps):
            if success:
                action = np.array([0, 0, 0, -1.0])  # "release": open the hand for 1 s, then end
                released_steps += 1
                if released_steps > FPS:
                    break
            else:
                action = act(env, obs)
            obs, _, terminated, truncated, _ = env.step(action)
            success = success or terminated
            renderer.update_scene(env.sim.d, camera="demo")
            img = cv2.cvtColor(renderer.render(), cv2.COLOR_RGB2BGR)
            ident = env.predicted if env.detected else "-"
            status = ("SUCCESS - released" if released_steps else "SUCCESS - holding") if success else ("grasping" if env.detected else env.voice)
            cv2.putText(img, f"{who} | object: {ident} | {status} | t={k / FPS:.1f}s",
                        (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
            frames.append(img)
        if not success:
            continue
        path = os.path.join(args.out, f"{kept + 1}_{env.obj}.mp4")
        writer = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"mp4v"), FPS, (640, 480))
        for f in frames:
            writer.write(f)
        writer.release()
        assert os.path.getsize(path) > 10_000, f"video {path} looks empty"
        print(f"saved {path} ({len(frames)} frames)")
        kept += 1
        seen.add(env.obj)
        if kept == args.n:
            break
    renderer.close()
    print(f"{kept} replays in {args.out}/")


if __name__ == "__main__":
    main()
