"""
Footage for the 3-minute video's voice-over part: exactly 2:20, sections timed to docs/VIDEO_SCRIPT.md.

   0-10 s  the problem             10-22 s  robots run on tables     22-32 s  three questions
  32-112 s five real hand-overs, 16 s each (bottle, cup, pen, one seen through the depth camera, a re-grasp
           TabPFN triggered)
 112-140 s results: stop point, will it hold, grasp success

  python make_broll.py          # -> demo_replays/broll.mp4  (then: python make_final_video.py)
"""

import os
import subprocess
import tempfile

import cv2
import mujoco
import numpy as np
from PIL import Image, ImageDraw
from stable_baselines3 import PPO

from classifier import load_classifier
from grasp_outcome import load_predictor as load_hold
from make_pitch_video import (BG, CARD, CYAN, F, FPS, GREEN, H, LINE, MUTED, PINK, TEXT, W, caption_frames,
                              pick_held_seed, pick_seeds, run_episode)
from sim.env import ArmEnv
from trajectory import CAMERA_DATA, load_predictor as load_goal

OUT = "demo_replays/broll.mp4"
EPISODE_S = 16  # each hand-over is stretched or held to exactly this, so the script's timings hold


def table_card(eyebrow, title, header, rows, notes, seconds, win=0):
    """A card with a small table (first column wide), highlighting row `win`."""
    img = Image.new("RGB", (W, H), BG)
    d = ImageDraw.Draw(img)
    d.text((110, 110), eyebrow, font=F["label"], fill=CYAN)
    d.text((110, 140), title, font=F["h2"], fill=TEXT)
    x = [110, 560, 830]
    y = 230
    for i, h in enumerate(header):
        d.text((x[i], y), h, font=F["label"], fill=MUTED)
    y += 34
    for r, row in enumerate(rows):
        if r == win:
            d.rounded_rectangle((96, y - 6, W - 110, y + 40), 10, fill=CARD, outline=CYAN, width=2)
        for i, cell in enumerate(row):
            d.text((x[i], y), cell, font=F["h3"] if r == win else F["body"], fill=TEXT if r == win else MUTED)
        y += 54
    y += 20
    for line in notes:
        d.text((110, y), line, font=F["body"], fill=TEXT)
        y += 36
    frame = np.array(img)
    return [(frame * min(1.0, i / 8)).astype(np.uint8) for i in range(int(seconds * FPS))]


def rows_card(seconds):
    """'Robots run on tables': a few real-looking rows with the three question columns in pink."""
    img = Image.new("RGB", (W, H), BG)
    d = ImageDraw.Draw(img)
    d.text((110, 100), "ROBOTS RUN ON TABLES", font=F["label"], fill=CYAN)
    d.text((110, 130), "Every moment of the arm is one row", font=F["h2"], fill=TEXT)
    cols = ["time", "object speed", "obj → hand", "wrist", "grip", "finger force", "what is it?", "where stops?",
            "will it hold?"]
    xs = [110, 200, 340, 480, 570, 650, 790, 940, 1090]
    rows = [["0.48 s", "0.21 m/s", "−0.41 m", "−0.47", "0.00", "0.0 N", "bottle", "…", "…"],
            ["2.12 s", "0.06 m/s", "0.04 m", "0.19", "0.96", "7.8 N", "cup", "…", "held"],
            ["…", "…", "…", "…", "…", "…", "…", "…", "…"],
            ["now", "0.23 m/s", "−0.12 m", "0.31", "0.00", "0.0 N", "?", "?", "?"]]
    y = 240
    for c, x in zip(cols, xs):
        d.text((x, y), c, font=F["label"], fill=PINK if c.endswith("?") else MUTED)
    y += 40
    for r, row in enumerate(rows):
        if r == len(rows) - 1:
            d.rounded_rectangle((96, y - 6, W - 60, y + 36), 10, fill=CARD, outline=PINK, width=2)
        for cell, x in zip(row, xs):
            d.text((x, y), cell, font=F["body"], fill=PINK if cell == "?" else TEXT)
        y += 54
    d.text((110, y + 30), "Past rows are TabPFN’s context: it fills in the question marks in one forward pass.",
           font=F["body"], fill=MUTED)
    frame = np.array(img)
    return [(frame * min(1.0, i / 8)).astype(np.uint8) for i in range(int(seconds * FPS))]


def fit(frames, seconds, max_slow=2.0):
    """Resample an episode to exactly `seconds`: slow it evenly (at most max_slow x), then hold the last frame."""
    n = int(seconds * FPS)
    s = min(max_slow, n / len(frames))
    out = [frames[min(len(frames) - 1, int(i / s))] for i in range(min(n, int(len(frames) * s)))]
    return out + [out[-1]] * (n - len(out))


def encode(frames, out):
    with tempfile.TemporaryDirectory() as tmp:
        raw = os.path.join(tmp, "raw.mp4")
        w = cv2.VideoWriter(raw, cv2.VideoWriter_fourcc(*"mp4v"), FPS, (W, H))
        for f in frames:
            w.write(cv2.cvtColor(f, cv2.COLOR_RGB2BGR))
        w.release()
        subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", raw, "-c:v", "libx264", "-pix_fmt", "yuv420p",
                        "-crf", "20", "-movflags", "+faststart", out], check=True)


def main():
    model = PPO.load("models/ppo_arm.zip", device="cpu")
    env = ArmEnv(goal_predictor=load_goal(), hold_predictor=load_hold())
    ident_clf = load_classifier(fast=True)
    good, risky = pick_seeds(env, model)
    assert len(good) == 4 and risky is not None, f"not enough episodes found: {good}, {risky}"
    renderer = mujoco.Renderer(env.m, 585, 840)
    cam_env = ArmEnv(goal_predictor=load_goal(CAMERA_DATA), hold_predictor=load_hold(), camera=True)
    cam_seed = pick_held_seed(cam_env, model)

    frames = caption_frames([
        ("THE PROBLEM", "label", CYAN, 22),
        ("Someone who lost an arm at the shoulder", "h2", TEXT, 6),
        ("is handed an object.", "h2", TEXT, 36),
        ("The arm must catch it safely, and only after the user says “ready”.", "body", TEXT, 14),
        ("The object is still moving in another person’s hand.", "body", MUTED, 0)], 10)
    frames += rows_card(12)
    frames += caption_frames([
        ("THREE QUESTIONS, ANSWERED IN CONTEXT", "label", CYAN, 22),
        ("TabPFN-3.5, no training run", "h2", TEXT, 40),
        ("1   What is it?", "h2", PINK, 18),
        ("2   Where will it stop?   (it looks again every 0.2 s)", "h2", PINK, 18),
        ("3   Will this grasp hold?   (if not: re-grasp)", "h2", PINK, 0)], 10)
    episodes = [(env, renderer, s, f"Hand-over {i + 1} of 4 · slowed for narration") for i, s in enumerate(good[:3])]
    episodes.append((cam_env, mujoco.Renderer(cam_env.m, 585, 840), cam_seed,
                     "Hand-over 4 of 4 · seen through the depth camera · slowed"))
    episodes.append((env, renderer, risky, "When TabPFN says “risky” · slowed for narration"))
    for e, r, seed, title in episodes:
        ep, _, _ = run_episode(e, model, seed, ident_clf, r, title)
        frames += fit(ep, EPISODE_S)
    frames += table_card("RESULTS · 300 HAND-OVERS THE MODEL NEVER SAW", "Where will it stop?",
                         ["model", "median error", "90% under"],
                         [["TabPFN-3.5", "0.6 cm", "3.8 cm"], ["linear regression", "3.6 cm", "9.1 cm"],
                          ["physics: constant braking", "4.6 cm", "77 cm"],
                          ["straight line to arm’s length", "8.7 cm", "22.2 cm"],
                          ["TabPFN, from the depth camera", "2.1 cm", "6.1 cm"]],
                         ["Each look sharpens it: 2.5 cm far away, 0.1 cm in the last 10 cm.",
                          "From camera pixels: still twice as good as linear regression (4.6 cm)."], 9)
    frames += table_card("RESULTS · 598 GRASPS THE MODEL NEVER SAW", "Will this grasp hold?  (at first touch)",
                         ["model", "ROC AUC", "Brier ↓"],
                         [["TabPFN-3.5", "0.961", "0.055"], ["logistic regression", "0.837", "0.109"],
                          ["rule: wrist error under 15°", "0.551", "0.161"]],
                         ["Calibrated: when it says 70%, 72% held."], 9)
    frames += caption_frames([
        ("RESULTS · 200 HELD-OUT HAND-OVERS, REAL CONTACT PHYSICS, NO GRASP ASSIST", "label", CYAN, 30),
        ("92%", "h1", GREEN, 4),
        ("grasp success: the arm acts on TabPFN’s predictions and re-grasps on its warning", "body", TEXT, 26),
        ("89% without the re-grasp.  21 re-grasps: 8 saved a grasp, 2 spoiled one, 11 changed nothing.", "body",
         MUTED, 12),
        ("Seeing the object only through the depth camera: 81%.", "body", MUTED, 0)], 10)

    assert len(frames) == 140 * FPS, f"{len(frames) / FPS:.2f} s, the script expects 140 s"
    encode(frames, OUT)
    print(f"saved {OUT}: {len(frames) / FPS:.0f} s, {os.path.getsize(OUT) / 1e6:.1f} MB")


if __name__ == "__main__":
    main()
