"""
README pictures, rendered from the real simulator with the trained agent and TabPFN in the loop:
  docs/images/hero.png       a person handing a bottle over; pink = TabPFN's predicted stop point
  docs/images/grasps.png     the five objects held by the hand (contact physics only)
  docs/images/sequence.png   one hand-over in four steps
  docs/images/camera.png     a hand-over seen through the depth camera: the 3D view beside what the arm sees

  python make_pictures.py
"""

import os

import mujoco
import numpy as np
from PIL import Image, ImageDraw
from stable_baselines3 import PPO

from grasp_outcome import load_predictor as load_hold
from make_pitch_video import F
from sim.env import ArmEnv
from trajectory import CAMERA_DATA, load_predictor as load_goal

OUT = "docs/images"
SEEDS = {"bottle": 60002, "cup": 60000, "pen": 60005, "ball": 60007}  # held hand-overs (make_pitch_video picks)


def camera(lookat, distance, azimuth, elevation):
    cam = mujoco.MjvCamera()
    cam.lookat[:], cam.distance, cam.azimuth, cam.elevation = lookat, distance, azimuth, elevation
    return cam


def renderer(env, h, w):
    env.m.vis.global_.offwidth, env.m.vis.global_.offheight = max(w, 1920), max(h, 1080)
    env.m.vis.quality.offsamples = 8  # anti-aliasing
    return mujoco.Renderer(env.m, h, w)


def play(env, model, seed, stop):
    """Run the agent until stop(env, k) is true; the pink marker shows TabPFN's latest stop-point guess."""
    marker = env.m.site("goal_pred").id
    obs, _ = env.reset(seed=seed)
    for k in range(env.n_steps):
        obs, _, term, trunc, _ = env.step(model.predict(obs, deterministic=True)[0])
        env.m.site_pos[marker] = env.G_pred if env.G_pred is not None else [0, 0, -5]
        if stop(env, k) or term or trunc:
            break


def held_object_seed(env, model, obj, first=60_000):
    for seed in range(first, first + 300):
        env.reset(seed=seed)
        if env.obj != obj:
            continue
        play(env, model, seed, lambda e, k: e.held_time > 0.6)
        if env.held_time > 0.6:
            return seed
    raise RuntimeError(f"no held {obj} found")


def label(img, text, xy=(24, 20), size="h2"):
    d = ImageDraw.Draw(img)
    x, y = xy
    w = d.textlength(text, font=F[size])
    d.rounded_rectangle((x - 12, y - 8, x + w + 12, y + F[size].size + 14), 10, fill=(11, 19, 34))
    d.text((x, y), text, font=F[size], fill=(238, 242, 248))
    return img


def main():
    os.makedirs(OUT, exist_ok=True)
    model = PPO.load("models/ppo_arm.zip", device="cpu")
    env = ArmEnv(goal_predictor=load_goal(), hold_predictor=load_hold())
    env.m.site_rgba[env.m.site("reach_zone").id, 3] = 0  # the green debug box hides the hand in stills

    # 1. hero: mid hand-over, the object a hand's width from the prosthetic hand
    r = renderer(env, 1080, 1920)
    play(env, model, SEEDS["bottle"], lambda e, k: e.detected and e.G_pred is not None and e.dist < 0.16)
    r.update_scene(env.d, camera(np.array([0.65, -0.25, 1.25]), 1.45, 58, -10))
    Image.fromarray(r.render()).save(f"{OUT}/hero.png")

    # 2. grasp gallery: each object held, close up on the hand
    tiles = []
    rt = renderer(env, 560, 560)
    for obj in ["cup", "bottle", "box", "pen", "ball"]:
        seed = SEEDS.get(obj) or held_object_seed(env, model, obj)
        play(env, model, seed, lambda e, k: e.held_time > 0.6)
        env.m.site_rgba[env.m.site("goal_pred").id, 3] = 0  # marker off in close-ups
        hand = env.d.site_xpos[env.site].copy()
        rt.update_scene(env.d, camera(hand + [0, 0, -0.02], 0.42, -60, -18))
        tiles.append(label(Image.fromarray(rt.render()), obj, size="h3"))
    gallery = Image.new("RGB", (560 * 5 + 16 * 4, 560), (11, 19, 34))
    for i, t in enumerate(tiles):
        gallery.paste(t, (i * (560 + 16), 0))
    gallery.save(f"{OUT}/grasps.png")
    env.m.site_rgba[env.m.site("goal_pred").id, 3] = 0.75

    # 3. one hand-over in four steps, on the demo camera
    rs = renderer(env, 540, 960)
    steps = [("1  “Ready”: a person hands over a cup", lambda e, k: k >= 22),
             ("2  TabPFN predicts where it stops (pink)", lambda e, k: e.looks >= 3),
             ("3  First touch: TabPFN asks “will it hold?”", lambda e, k: e.t_touch is not None),
             ("4  Held by contact physics alone", lambda e, k: e.held_time > 0.6)]
    shots = []
    for text, stop in steps:
        play(env, model, SEEDS["cup"], stop)
        rs.update_scene(env.d, camera(np.array([0.6, -0.25, 1.15]), 1.8, 62, -12))
        shots.append(label(Image.fromarray(rs.render()), text, size="h3"))
    seq = Image.new("RGB", (960 * 2 + 12, 540 * 2 + 12), (11, 19, 34))
    for i, s in enumerate(shots):
        seq.paste(s, ((i % 2) * (960 + 12), (i // 2) * (540 + 12)))
    seq.save(f"{OUT}/sequence.png")

    # 4. the depth camera: what the arm sees (object in pink) beside the 3D view, mid hand-over
    cam = ArmEnv(goal_predictor=load_goal(CAMERA_DATA), hold_predictor=load_hold(), camera=True)
    cam.m.site_rgba[cam.m.site("reach_zone").id, 3] = 0
    play(cam, model, 61_000, lambda e, k: e.looks >= 3)  # make_pitch_video's camera hand-over
    rc = renderer(cam, 720, 960)
    rc.update_scene(cam.d, camera(np.array([0.6, -0.25, 1.15]), 1.8, 62, -12))
    left = label(Image.fromarray(rc.render()), "3D view (pink dot: TabPFN's stop point)", size="h3")
    right = label(Image.fromarray(cam.eye.picture()).resize((960, 720), Image.NEAREST),
                  "What the arm sees: depth, object in pink", size="h3")
    both = Image.new("RGB", (960 * 2 + 12, 720), (11, 19, 34))
    both.paste(left, (0, 0))
    both.paste(right, (960 + 12, 0))
    both.save(f"{OUT}/camera.png")

    for f in sorted(os.listdir(OUT)):
        print(f"{OUT}/{f}: {Image.open(os.path.join(OUT, f)).size}")


if __name__ == "__main__":
    main()
