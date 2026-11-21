"""
"What happens next": TabPFN-3.5 predicts, at the moment the fingers first touch the
object, the probability that the grasp ends in a stable, aligned hold (spec section 6
success). One row per episode, logged at first touch; TabPFN learns in-context from
past grasps, no training run.

  python grasp_outcome.py --collect 2000     # log touch rows -> data/grasp_touch.csv (parallel)
  python grasp_outcome.py                    # score on held-out episodes vs baselines

From other code:
  from grasp_outcome import load_predictor, touch_features
  clf = load_predictor()
  clf.predict_proba([touch_features(env)])[0, 1]   # P(stable hold), call right after env.t_touch is set
"""

import argparse
import math
import os
import sys
import time
from multiprocessing import Pool

import numpy as np
import pandas as pd

from generate_dataset import ARM_JOINTS, OBJECTS

DATA = "data/grasp_touch.csv"
FEATURES = ["object", "wrist_err_deg", "grasp_dist", "total_force", "finger_force", "grip_angle",
            "grip_cmd", "obj_speed", "rel_x", "rel_y", "rel_z", "arm_speed"]
CATEGORICAL = [FEATURES.index("object")]
LIVE_CONTEXT = 600  # rows kept for live use on CPU (fast checkpoint + cached context)


def touch_features(env):
    """Feature row at the first finger touch (what the robot can sense then; no future info)."""
    sim, d = env.sim, env.sim.d
    rel = env.hand_R.T @ (env.opos - env.hand_pos)
    return [
        OBJECTS.index(env.predicted) if env.predicted in OBJECTS else -1,
        math.degrees(env.err_at_touch) if not math.isnan(env.err_at_touch) else 0.0,
        env.grasp_dist, env.total_f, env.finger_f,
        float(d.qpos[sim.grip_q]), float(env.ctrl[3]),
        float(np.linalg.norm(env.ovel)), *rel,
        float(np.abs([d.qvel[sim.jdof[j]] for j in ARM_JOINTS]).sum()),
    ]


# --------------------------------------------------------------------- collection
_worker = {}


def _init():
    from stable_baselines3 import PPO
    from arm_env import ArmEnv
    _worker["env"] = ArmEnv()
    _worker["rl"] = PPO.load("models/ppo_arm.zip", device="cpu")


def _episode(seed):
    """One episode with a controller picked by seed; returns the touch row + label, or None."""
    env, rl = _worker["env"], _worker["rl"]
    mode = ("rl", "rl_noisy", "expert")[seed % 3]  # mix of controllers: both holds and failures
    obs, _ = env.reset(seed=seed)
    row = None
    while True:
        if mode == "expert":
            a = env.expert_action()
        else:
            a = rl.predict(obs, deterministic=(mode == "rl"))[0]
        obs, _, term, trunc, info = env.step(a)
        if row is None and env.t_touch is not None:
            row = touch_features(env)
        if term or trunc:
            break
    return None if row is None else [seed, mode, *row, int(info["success"])]


def collect(n, out=DATA):
    seeds = range(100_000, 100_000 + n)  # apart from eval (10_000+), BC (20_000+) and demo (50_000+) seeds
    start = time.time()
    with Pool(max(1, (os.cpu_count() or 4) - 2), initializer=_init) as pool:
        rows = [r for r in pool.imap_unordered(_episode, seeds, chunksize=8) if r is not None]
    df = pd.DataFrame(rows, columns=["seed", "controller", *FEATURES, "held"]).sort_values("seed")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    df.to_csv(out, index=False)
    print(f"{len(df)} touch rows from {n} episodes in {time.time() - start:.0f}s -> {out}"
          f"   held {df.held.mean():.0%}   by controller: {df.groupby('controller').held.mean().round(2).to_dict()}")


# --------------------------------------------------------------------- model
def make_model(fast=True, cache=False):
    from tabpfn import TabPFNClassifier
    from tabpfn.constants import ModelVersion
    version = ModelVersion.V3_5_FAST if fast else ModelVersion.V3_5
    return TabPFNClassifier.create_default_for_version(
        version, device="auto", categorical_features_indices=CATEGORICAL,
        **({"fit_mode": "fit_with_cache"} if cache else {}))


def load_predictor(path=DATA, n=LIVE_CONTEXT):
    """Live predictor: fast checkpoint, context cached once so each prediction is a quick forward pass."""
    df = pd.read_csv(path)
    df = df.sample(min(n, len(df)), random_state=0)
    return make_model(fast=True, cache=True).fit(df[FEATURES].to_numpy(float), df["held"].to_numpy())


def evaluate(path=DATA, fast="--full" not in sys.argv):
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import accuracy_score, brier_score_loss, roc_auc_score
    from sklearn.model_selection import train_test_split

    df = pd.read_csv(path)
    tr, te = train_test_split(df, test_size=0.25, stratify=df["held"], random_state=0)  # held-out episodes
    X_tr, y_tr, X_te, y_te = tr[FEATURES].to_numpy(float), tr["held"].to_numpy(), te[FEATURES].to_numpy(float), te["held"].to_numpy()

    t = time.time()
    p_tab = make_model(fast).fit(X_tr, y_tr).predict_proba(X_te)[:, 1]
    t_tab = time.time() - t
    lr = LogisticRegression(max_iter=2000).fit(pd.get_dummies(tr[FEATURES], columns=["object"]).to_numpy(float), y_tr)
    p_lr = lr.predict_proba(pd.get_dummies(te[FEATURES], columns=["object"]).reindex(
        columns=pd.get_dummies(tr[FEATURES], columns=["object"]).columns, fill_value=0).to_numpy(float))[:, 1]
    p_rule = (te["wrist_err_deg"] < 15).astype(float).to_numpy()  # "aligned at touch -> it will hold"
    p_base = np.full(len(te), y_tr.mean())                        # always the training hold rate

    print(f"{len(tr)} context grasps, {len(te)} held-out grasps (held {y_te.mean():.0%})")
    print(f"{'model':28s} {'accuracy':>9s} {'ROC AUC':>8s} {'Brier':>7s}   (Brier: lower = better calibrated)")
    for name, p in [(f"TabPFN-3.5{'-fast' if fast else ''}", p_tab), ("logistic regression", p_lr),
                    ("rule: wrist error < 15 deg", p_rule), ("always the base rate", p_base)]:
        auc = roc_auc_score(y_te, p) if len(set(p)) > 1 else float("nan")
        print(f"{name:28s} {accuracy_score(y_te, p > 0.5):9.3f} {auc:8.3f} {brier_score_loss(y_te, p):7.3f}")
    print(f"TabPFN predict time for {len(te)} rows: {t_tab:.1f}s")
    # calibration table: when TabPFN says ~x%, how often did it hold?
    bins = pd.cut(p_tab, [0, .2, .4, .6, .8, 1.0], include_lowest=True)
    print("\nTabPFN calibration (predicted -> actual hold rate, count):")
    print(pd.DataFrame({"pred": p_tab, "held": y_te, "bin": bins}).groupby("bin", observed=True)
          .agg(predicted=("pred", "mean"), actual=("held", "mean"), n=("held", "size")).round(2).to_string())
    assert roc_auc_score(y_te, p_tab) > 0.7, "TabPFN barely beats chance - check the data"


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--collect", type=int, help="episodes to run and log")
    ap.add_argument("--full", action="store_true", help="evaluate with the full TabPFN-3.5 (slow on CPU)")
    args = ap.parse_args()
    if args.collect:
        collect(args.collect)
    else:
        evaluate()
