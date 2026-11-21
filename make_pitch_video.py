"""
Pitch demo video: title, how it works, real hand-overs with TabPFN's predictions filled in live
beside the 3D view, results, closing. 1280x720, 25 fps, H.264 (needs ffmpeg on PATH).

  python make_pitch_video.py          # -> demo_replays/pitch_demo.mp4 (about 75 s)
"""

import os
import subprocess
import tempfile

import cv2
import mujoco
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from stable_baselines3 import PPO

from classifier import load_classifier
from grasp_outcome import load_predictor as load_hold, touch_features
from sim.env import RELEASE_ACTION, ArmEnv
from trajectory import load_predictor as load_goal

OUT = "demo_replays/pitch_demo.mp4"
W, H, FPS = 1280, 720, 25
BG, CARD, LINE = (11, 19, 34), (22, 35, 58), (42, 59, 88)
TEXT, MUTED, CYAN, PINK, GREEN, AMBER, RED = ((238, 242, 248), (155, 170, 192), (47, 182, 240), (224, 56, 159),
                                              (52, 211, 153), (251, 191, 36), (248, 113, 113))
FONTS = "C:/Windows/Fonts/"


def font(size, bold=False):
    try:
        return ImageFont.truetype(FONTS + ("segoeuib.ttf" if bold else "segoeui.ttf"), size)
    except OSError:  # not Windows: PIL's own font (smaller, but readable)
        return ImageFont.load_default(size)


F = {k: font(*v) for k, v in {"h1": (58, True), "h2": (36, True), "h3": (26, True), "body": (24,), "small": (19,),
                              "label": (16, True), "big": (40, True)}.items()}


def card(lines, fade=1.0):
    """A full-screen text card. lines: (text, font key, colour, gap below)."""
    img = Image.new("RGB", (W, H), BG)
    d = ImageDraw.Draw(img)
    y = 150
    for text, f, col, gap in lines:
        for i, part in enumerate(text.split("\t")):  # a tab starts a second, aligned column
            d.text((110 + 330 * i, y), part, font=F["h3" if i == 0 and "\t" in text else f], fill=col if i else (
                CYAN if "\t" in text else col))
        y += F[f].size + gap
    return np.array(img) * fade


def caption_frames(lines, seconds):
    frames = [card(lines, min(1.0, i / 8)) for i in range(int(seconds * FPS))]  # fade in over 0.3 s
    return [f.astype(np.uint8) for f in frames]


# ------------------------------------------------------------------ episodes
def panel(img, s):
    """Right-hand panel: the voice command and TabPFN's three answers as they arrive."""
    d = ImageDraw.Draw(img)
    x0, y = 910, 70

    def box(title, tag, h):
        nonlocal y
        d.rounded_rectangle((x0, y, W - 30, y + h), 14, fill=CARD, outline=LINE, width=2)
        d.text((x0 + 18, y + 12), title, font=F["label"], fill=MUTED)
        if tag:
            d.text((W - 48 - d.textlength(tag, font=F["label"]), y + 12), tag, font=F["label"], fill=PINK)
        top = y + 36
        y += h + 14
        return top

    t = box("VOICE COMMAND", "", 70)
    d.text((x0 + 18, t), s["voice"], font=F["h3"], fill=CYAN if s["voice"] != "idle" else MUTED)
    t = box("1  WHAT IS IT?", "TabPFN", 78)
    d.text((x0 + 18, t), s["ident"] or "—", font=F["h3"], fill=TEXT if s["ident"] else MUTED)
    t = box("2  WHERE WILL IT STOP?", "TabPFN", 104)
    if s["goal"]:
        d.text((x0 + 18, t), f"{s['goal'][0]:.1f} cm off", font=F["h3"], fill=TEXT)
        d.text((x0 + 18, t + 34), f"range {s['goal'][1]:.0f} cm · pink dot in view", font=F["small"], fill=MUTED)
    else:
        d.text((x0 + 18, t), "—", font=F["h3"], fill=MUTED)
    t = box("3  WILL IT HOLD?", "TabPFN", 104)
    if s["hold"] is not None:
        p = s["hold"]
        d.rounded_rectangle((x0 + 18, t + 4, W - 48, t + 18), 7, fill=LINE)
        d.rounded_rectangle((x0 + 18, t + 4, x0 + 18 + int((W - 66 - x0) * p), t + 18), 7, fill=CYAN if p >= 0.5 else RED)
        d.text((x0 + 18, t + 28), f"{p * 100:.0f}% chance", font=F["h3"], fill=TEXT)
        d.text((W - 48 - d.textlength("at first touch", font=F["small"]), t + 34), "at first touch",
               font=F["small"], fill=MUTED)
    else:
        d.text((x0 + 18, t), "—", font=F["h3"], fill=MUTED)
    t = box("RESULT", "", 78)
    d.text((x0 + 18, t), s["result"], font=F["h3"], fill=s["result_col"])


def run_episode(env, model, seed, ident_clf, hold_clf, renderer, title, render=True):
    """Play one episode; return (frames, outcome, hold chance), outcome "held" / "dropped" / "misaligned" /
    "missed". Frames only if render. A drop ends the clip 1.5 s later instead of at the 8 s timeout."""
    marker = env.m.site("goal_pred").id
    obs, _ = env.reset(seed=seed)
    s = {"voice": "idle", "ident": None, "goal": None, "hold": None, "result": "waiting", "result_col": MUTED}
    frames, success, released, caption = [], False, 0, "The arm waits. Nothing moves until the user says “ready”."
    outcome, k_drop = "missed", None
    for k in range(env.n_steps + FPS * 2):
        if success:
            action = RELEASE_ACTION
            released += 1
            s["voice"], caption = "“release”", "“Release”: the hand opens."
            if released > FPS:
                break
        elif k >= env.n_steps:
            break
        else:
            action = model.predict(obs, deterministic=True)[0]
        obs, _, term, trunc, info = env.step(action)
        if not success:
            s["voice"] = "“ready”" if env.voice == "ready" else env.voice
        if env.voice == "ready" and not env.detected and not success:
            caption = "“Ready”: an object is on its way."
        if env.detected and s["ident"] is None:
            s["ident"] = str(ident_clf.predict([list(env.features)])[0]) if render else env.predicted
            caption = f"Detected. TabPFN identifies it: a {s['ident']}."
        if env.G_pred is not None and s["goal"] is None:
            s["goal"] = (float(np.linalg.norm(env.G_pred - env.G)) * 100, (env.G_range[1] - env.G_range[0]) * 100)
            caption = "From one look, TabPFN predicts where it will stop (pink). The RL agent moves the hand there."
        if env.t_touch is not None and s["hold"] is None:
            s["hold"] = float(hold_clf.predict_proba([touch_features(env)])[0, 1])
            caption = "First touch: TabPFN predicts whether this grasp will hold."
            s["result"], s["result_col"] = "grasping…", AMBER
        if env.detected and env.t_touch is None and not success:
            s["result"], s["result_col"] = "reaching…", AMBER
        if term and info["success"] and not success:
            success, outcome = True, "held"
            s["result"], s["result_col"] = "held · real physics", GREEN
            caption = "Held by contact physics alone: no grasp assist."
        if term and not info["success"]:
            outcome = "misaligned"
            s["result"], s["result_col"] = "held, wrist misaligned", RED
        if not success and not env.weld_on and env.grasp_dist > 0.15 and k_drop is None:
            k_drop, outcome = k, "dropped"  # the person let go and it is clearly out of the hand
            s["result"], s["result_col"] = "dropped", RED
            if s["hold"] is not None and s["hold"] < 0.5:
                caption = f"It slipped, as TabPFN warned ({s['hold'] * 100:.0f}% chance)."
        if render:
            env.m.site_pos[marker] = env.G_pred if env.G_pred is not None else [0, 0, -5]
            renderer.update_scene(env.d, camera="demo")
            img = Image.new("RGB", (W, H), BG)
            img.paste(Image.fromarray(renderer.render()), (30, 70))
            d = ImageDraw.Draw(img)
            d.text((30, 22), title, font=F["h3"], fill=TEXT)
            d.text((870 - d.textlength(f"t = {k / FPS:.1f} s", font=F["body"]), 26), f"t = {k / FPS:.1f} s",
                   font=F["body"], fill=MUTED)
            d.rectangle((30, 655, 870, 700), fill=CARD)
            d.text((44, 663), caption, font=F["body"], fill=TEXT)
            panel(img, s)
            frames.append(np.array(img))
        if ((term or trunc) and not success) or (k_drop is not None and k - k_drop > 1.5 * FPS):
            break
    return frames, outcome, s["hold"]


def pick_seeds(env, model, hold_clf):
    """Successful hand-overs of bottle, cup, pen, ball; and a drop TabPFN called risky beforehand."""
    want, picks, risky = ["bottle", "cup", "pen", "ball"], {}, None
    for seed in range(60_000, 60_400):
        env.reset(seed=seed)
        obj = env.obj
        if obj in picks and risky is not None:
            continue
        _, outcome, hold = run_episode(env, model, seed, None, hold_clf, None, "", render=False)
        if outcome == "held" and obj in want and obj not in picks:
            picks[obj] = seed
        if outcome == "dropped" and risky is None and hold is not None and hold < 0.5:
            risky = seed
        if len(picks) == len(want) and risky is not None:
            break
    return [picks[o] for o in want if o in picks], risky


def main():
    model = PPO.load("models/ppo_arm.zip", device="cpu")
    env = ArmEnv(goal_predictor=load_goal())
    hold_clf, ident_clf = load_hold(), load_classifier(fast=True)
    good, risky = pick_seeds(env, model, hold_clf)
    print("seeds:", good, "risky drop:", risky)
    renderer = mujoco.Renderer(env.m, 585, 840)

    frames = caption_frames([
        ("HackLab Berlin × TabPFN · Prior Labs track", "label", CYAN, 22),
        ("How far can a table take", "h1", TEXT, 4), ("a prosthetic arm?", "h1", TEXT, 36),
        ("A simulated robot arm and hand that see an object coming, predict where it will stop", "body", MUTED, 6),
        ("and whether the grasp will hold, with TabPFN-3.5 learning in context.", "body", MUTED, 60),
        ("The Big Gang · Minh Nguyen, Quan Vo, Kazuki Dozono", "body", TEXT, 0)], 4.5)
    frames += caption_frames([
        ("HOW IT WORKS", "label", CYAN, 22),
        ("Someone hands an object to a person with a prosthetic arm.", "h2", TEXT, 30),
        ("1   The user says “ready”: nothing moves before that.", "body", TEXT, 14),
        ("2   TabPFN: what is it?  Where will it stop?", "body", PINK, 14),
        ("3   An RL agent moves a Kinova Gen3 arm and turns the wrist to suit the object.", "body", TEXT, 14),
        ("4   At first touch, TabPFN: will this grasp hold?", "body", PINK, 14),
        ("5   The TetherIA Aero Hand holds it by real contact physics. “Release” opens it.", "body", TEXT, 30),
        ("Every step, detection and grasp is a row of a table; TabPFN learns from past rows in context.", "small", MUTED, 0)],
        6.5)
    episodes = [(s, f"Hand-over {i + 1} of {len(good)}") for i, s in enumerate(good)]
    if risky is not None:
        episodes.append((risky, "When TabPFN says “risky”"))
    for seed, title in episodes:
        ep, _, _ = run_episode(env, model, seed, ident_clf, hold_clf, renderer, title)
        frames += ep
    frames += caption_frames([
        ("RESULTS · held-out tests the models never saw", "label", CYAN, 26),
        ("What is it?\t100% object identified  (hand-written rule: 98.3%)", "body", TEXT, 18),
        ("Where will it stop?\t2.6 cm median error from one look;  10–90% range covers 79%", "body", TEXT, 18),
        ("Will it hold?\tROC AUC 0.945 at first touch  (logistic regression: 0.795)", "body", TEXT, 34),
        ("84.5%", "h1", GREEN, 2),
        ("grasp success, the arm acting on TabPFN’s predictions, real contact physics, no assist", "body", TEXT, 6),
        ("(87.5% when given the simulator’s true stop point; 200 held-out episodes)", "small", MUTED, 0)], 7)
    frames += caption_frames([
        ("HOW FAR CAN A TABLE TAKE A ROBOT?", "label", CYAN, 22),
        ("From one look to a real grasp,", "h1", TEXT, 4), ("84.5% of the time.", "h1", TEXT, 46),
        ("github.com/nhatminhnguyen0419/Hacklab-Hackathon---Prior-Labs", "body", CYAN, 12),
        ("The Big Gang · Minh Nguyen, Quan Vo, Kazuki Dozono", "body", TEXT, 0)], 5)

    with tempfile.TemporaryDirectory() as tmp:
        raw = os.path.join(tmp, "raw.mp4")
        w = cv2.VideoWriter(raw, cv2.VideoWriter_fourcc(*"mp4v"), FPS, (W, H))
        for f in frames:
            w.write(cv2.cvtColor(f, cv2.COLOR_RGB2BGR))
        w.release()
        # H.264 + yuv420p: plays in PowerPoint, browsers and phones (OpenCV's mp4v often doesn't)
        subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", raw, "-c:v", "libx264", "-pix_fmt", "yuv420p",
                        "-crf", "20", "-movflags", "+faststart", OUT], check=True)
    assert os.path.getsize(OUT) > 100_000, "video looks empty"
    print(f"saved {OUT}: {len(frames) / FPS:.0f} s, {os.path.getsize(OUT) / 1e6:.1f} MB")


if __name__ == "__main__":
    main()
