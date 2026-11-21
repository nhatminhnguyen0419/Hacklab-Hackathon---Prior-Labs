"""
"Where it's going": TabPFN-3.5 predicts where an approaching object will stop (the
hand-over point), from one look at it when it is detected: its position and velocity.
The object travels in a straight line, so the model predicts the distance left along
that line; stop point = position + direction * distance. With a 10-90% interval.

  python trajectory.py --collect 3000     # log detection rows -> data/trajectory.csv (parallel)
  python trajectory.py                    # score on held-out episodes vs baselines

From other code:
  from trajectory import load_predictor
  predict = load_predictor()
  stop_point, (lo, hi) = predict(obj_pos, obj_vel)   # lo/hi: 10-90% interval of the distance left
"""

import argparse
import os
import sys
import time
from multiprocessing import Pool

import numpy as np
import pandas as pd

DATA = "data/trajectory.csv"
RESIDUALS = "data/trajectory_residuals.npy"
REACH_CENTER = np.array([0.0, -0.22, 1.40])  # the shoulder: hand-overs happen on a shell around it (sim/env.py)
ARM_LENGTH = 0.48  # m, typical hand-over distance from the shoulder
FEATURES = ["px", "py", "pz", "vx", "vy", "vz", "speed", "s_center", "s_shell"]
LIVE_CONTEXT = 800


def features(p, v):
    """Position + velocity, plus how far along the path the shoulder is closest (s_center) and where the
    path first reaches arm's length from it (s_shell): features before raw data."""
    p, v = np.asarray(p, float), np.asarray(v, float)
    speed = float(np.linalg.norm(v))
    u = v / max(speed, 1e-9)
    b = float((REACH_CENTER - p) @ u)
    disc = b * b - (float((p - REACH_CENTER) @ (p - REACH_CENTER)) - ARM_LENGTH ** 2)
    return [*p, *v, speed, b, b - np.sqrt(disc) if disc > 0 else b]  # path misses the shell: closest point


# --------------------------------------------------------------------- collection
_worker = {}


def _init():
    from sim.env import ArmEnv
    _worker["env"] = ArmEnv()


def _episode(seed):
    """Run until detection (the arm barely matters before it); return the row and the true stop point."""
    env = _worker["env"]
    env.reset(seed=seed)
    for _ in range(env.n_steps):
        env.step(env.expert_action())
        if env.detected:
            return [seed, *features(env.opos, env.ovel), float(np.linalg.norm(env.G - env.opos)), *env.G]
    return None


def collect(n, out=DATA):
    seeds = range(200_000, 200_000 + n)  # apart from every other seed range in the project
    start = time.time()
    with Pool(max(1, (os.cpu_count() or 4) - 2), initializer=_init) as pool:
        rows = [r for r in pool.imap_unordered(_episode, seeds, chunksize=16) if r is not None]
    df = pd.DataFrame(rows, columns=["seed", *FEATURES, "dist_left", "gx", "gy", "gz"]).sort_values("seed")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    df.to_csv(out, index=False)
    print(f"{len(df)} detection rows from {n} episodes in {time.time() - start:.0f}s -> {out}"
          f"   distance left {df.dist_left.mean():.2f} +- {df.dist_left.std():.2f} m")


# --------------------------------------------------------------------- model
def make_model(cache=False):
    from tabpfn import TabPFNRegressor
    from tabpfn.constants import ModelVersion
    return TabPFNRegressor.create_default_for_version(
        ModelVersion.V3_5_FAST, device="auto", **({"fit_mode": "fit_with_cache"} if cache else {}))


def _stop_point(row, dist):
    """row = features(); stop point = position + unit velocity * distance left."""
    p, v, speed = np.array(row[0:3]), np.array(row[3:6]), row[6]
    return p + v / max(speed, 1e-9) * dist


def load_predictor(path=DATA, n=LIVE_CONTEXT):
    """Fast checkpoint with a cached context: ~0.1 s per prediction on CPU after the first."""
    df = pd.read_csv(path).sample(frac=1, random_state=0).head(n)
    model = make_model(cache=True).fit(df[FEATURES].to_numpy(float), df["dist_left"].to_numpy())

    def predict(p, v):
        row = features(p, v)
        out = model.predict(np.array([row]), output_type="quantiles", quantiles=[0.1, 0.5, 0.9])
        lo, med, hi = (float(q[0]) for q in out)
        return _stop_point(row, med), (lo, hi)
    predict(REACH_CENTER + [0.4, 0, 0], [-0.25, 0, 0])  # warm-up: the first call is much slower
    return predict


def evaluate(path=DATA):
    from sklearn.linear_model import LinearRegression
    from sklearn.model_selection import train_test_split

    df = pd.read_csv(path)
    tr, te = train_test_split(df, test_size=0.25, random_state=0)  # held-out episodes
    X_tr, X_te = tr[FEATURES].to_numpy(float), te[FEATURES].to_numpy(float)
    G = te[["gx", "gy", "gz"]].to_numpy()

    t = time.time()
    lo, med, hi = make_model().fit(X_tr, tr["dist_left"]).predict(X_te, output_type="quantiles",
                                                                  quantiles=[0.1, 0.5, 0.9])
    t_tab = time.time() - t
    preds = {
        "TabPFN-3.5-fast": med,
        "linear regression": LinearRegression().fit(X_tr, tr["dist_left"]).predict(X_te),
        "geometry: reaches arm's length": te["s_shell"].to_numpy(),  # geometry only: stop at arm's length
        "average distance left": np.full(len(te), tr["dist_left"].mean()),
    }
    print(f"{len(tr)} context approaches, {len(te)} held-out; true distance left "
          f"{te.dist_left.mean() * 100:.0f} +- {te.dist_left.std() * 100:.0f} cm")
    print(f"{'model':30s} {'stop-point error: median':>25s} {'90th pct':>9s}")
    for name, dist in preds.items():
        err = np.linalg.norm(np.array([_stop_point(r, d) for r, d in zip(X_te, dist)]) - G, axis=1) * 100
        print(f"{name:30s} {np.median(err):22.1f} cm {np.percentile(err, 90):6.1f} cm")
    cover = np.mean((te["dist_left"] >= lo) & (te["dist_left"] <= hi))
    print(f"TabPFN 10-90% interval holds the true distance {cover:.0%} of the time (ideal 80%), "
          f"mean width {np.mean(hi - lo) * 100:.0f} cm;  predict time {t_tab:.1f}s for {len(te)} rows")
    resid = np.array([_stop_point(r, d) for r, d in zip(X_te, med)]) - G
    np.save(RESIDUALS, resid)  # TabPFN's real held-out errors: train.py --goal-noise trains the agent with them
    print(f"saved {len(resid)} held-out error vectors -> {RESIDUALS}")
    assert np.median(np.linalg.norm(resid, axis=1)) < 0.10, "stop point off by more than 10 cm - check the data"


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--collect", type=int, help="episodes to run and log")
    args = ap.parse_args()
    collect(args.collect) if args.collect else evaluate()
