"""
"Where it's going": TabPFN-3.5 predicts where an approaching object will stop (the hand-over
point). The person's hand moves along an arc and slows down before stopping (sim/env.py), so the
arm looks again every 0.2 s while the object moves: position, velocity and acceleration, written
in the object's own direction of travel. Three TabPFN regressors predict the stop point as
"how far ahead, how far to the side, how far up", with a 10-90% range.

  python trajectory.py --collect 1000     # log a row per look -> data/trajectory.csv (parallel)
  python trajectory.py                    # score on held-out episodes vs baselines

From other code:
  from trajectory import load_predictor
  predict = load_predictor()
  stop_point, (lo, hi) = predict(obj_pos, obj_vel, obj_acc)   # lo/hi: 10-90% of the distance ahead
"""

import argparse
import os
import time
from multiprocessing import Pool

import numpy as np
import pandas as pd

DATA = "data/trajectory.csv"
RESIDUALS = "data/trajectory_residuals.npy"
REACH_CENTER = np.array([0.0, -0.22, 1.40])  # the shoulder: hand-overs happen on a shell around it (sim/env.py)
ARM_LENGTH = 0.48  # m, typical hand-over distance from the shoulder
FEATURES = ["px", "py", "pz", "speed", "ux", "uy", "uz", "a_ahead", "a_side", "a_up", "s_shell", "s_brake"]
TARGETS = ["ahead", "side", "up"]  # stop point - position, in the direction-of-travel frame
LIVE_CONTEXT = 600  # looks (x3 stacked rows)
LOOK_STEPS = 5  # one look every 5 control steps = 0.2 s (sim/env.py LOOK_EVERY)


def frame(v):
    """Direction-of-travel frame: ahead (along v), side (horizontal, left of ahead), up."""
    e1 = np.asarray(v, float) / max(float(np.linalg.norm(v)), 1e-9)
    e2 = np.cross([0.0, 0.0, 1.0], e1)
    e2 = e2 / max(float(np.linalg.norm(e2)), 1e-9)
    return np.column_stack([e1, e2, np.cross(e1, e2)])


def features(p, v, a):
    """Raw state in the travel frame, plus two physics hints (features before raw data): where a
    straight path reaches arm's length from the shoulder, and the stop distance if it keeps braking."""
    p, v, a = (np.asarray(x, float) for x in (p, v, a))
    speed, E = float(np.linalg.norm(v)), frame(v)
    u, (a1, a2, a3) = E[:, 0], E.T @ a
    b = float((REACH_CENTER - p) @ u)
    disc = b * b - (float((p - REACH_CENTER) @ (p - REACH_CENTER)) - ARM_LENGTH ** 2)
    s_shell = b - np.sqrt(disc) if disc > 0 else b
    s_brake = min(speed * speed / (2 * -a1), 1.0) if a1 < -0.05 else 1.0  # not braking yet: "far"
    return [*p, speed, *u, a1, a2, a3, s_shell, s_brake]


# --------------------------------------------------------------------- collection
_worker = {}


def _init():
    from sim.env import ArmEnv
    _worker["env"] = ArmEnv()


def _episode(seed):
    """One row per look while the object moves after detection; target = where it really stops."""
    env = _worker["env"]
    env.reset(seed=seed)
    rows = []
    for k in range(env.n_steps):
        env.step(env.expert_action())
        t = env.k * env.dt
        if (env.detected and env.weld_on and env.t_touch is None and t >= 0.2 and env.k % LOOK_STEPS == 0
                and np.linalg.norm(env.ovel) >= 0.02):
            off = frame(env.ovel).T @ (env.G - env.opos)
            rows.append([seed, len(rows), *features(env.opos, env.ovel, env.oacc), *off,
                         float(np.linalg.norm(env.G - env.opos)), *env.G])
        if env.t_touch is not None or not env.weld_on:
            break
    return rows


def collect(n, out=DATA):
    seeds = range(200_000, 200_000 + n)  # apart from every other seed range in the project
    start = time.time()
    with Pool(max(1, (os.cpu_count() or 4) - 2), initializer=_init) as pool:
        rows = [r for ep in pool.imap_unordered(_episode, seeds, chunksize=8) for r in ep]
    df = pd.DataFrame(rows, columns=["seed", "look", *FEATURES, *TARGETS, "dist_left", "gx", "gy", "gz"])
    df = df.sort_values(["seed", "look"])
    os.makedirs(os.path.dirname(out), exist_ok=True)
    df.to_csv(out, index=False)
    print(f"{len(df)} looks from {df.seed.nunique()} of {n} episodes in {time.time() - start:.0f}s -> {out}"
          f"   {len(df) / df.seed.nunique():.1f} looks per hand-over")


# --------------------------------------------------------------------- model
def make_model(cache=False):
    from tabpfn import TabPFNRegressor
    from tabpfn.constants import ModelVersion
    return TabPFNRegressor.create_default_for_version(
        ModelVersion.V3_5_FAST, device="auto", **({"fit_mode": "fit_with_cache"} if cache else {}))


def _to_world(X, offsets):
    """Rows of features + predicted (ahead, side, up) -> stop points in the world."""
    out = []
    for x, o in zip(X, offsets):
        p, u = np.array(x[0:3]), np.array(x[4:7])
        out.append(p + frame(u) @ o)
    return np.array(out)


def _stack(X):
    """Each look -> 3 rows (features + which direction), so one model answers all three in one call."""
    return np.vstack([np.column_stack([X, np.full(len(X), i)]) for i in range(len(TARGETS))])


def load_predictor(path=DATA, n=LIVE_CONTEXT):
    """Live/agent predictor: ONE TabPFN model for ahead/side/up (stacked rows), 2 estimators, cached context:
    ~0.1 s per look on CPU, so the arm can wait for it every 0.2 s (three separate models: ~0.27 s; answering
    from a stale background prediction dropped live success from 25/30 to 5/30). Held-out: median 0.8 cm."""
    from tabpfn import TabPFNRegressor
    from tabpfn.constants import ModelVersion
    df = pd.read_csv(path).sample(frac=1, random_state=0).head(n)
    model = TabPFNRegressor.create_default_for_version(
        ModelVersion.V3_5_FAST, device="auto", fit_mode="fit_with_cache", n_estimators=2,
        categorical_features_indices=[len(FEATURES)])
    model.fit(_stack(df[FEATURES].to_numpy(float)), np.concatenate([df[c].to_numpy() for c in TARGETS]))

    def predict(p, v, a):
        row = np.array([features(p, v, a)])
        lo, med, hi = model.predict(_stack(row), output_type="quantiles", quantiles=[0.1, 0.5, 0.9])
        return _to_world(row, [med])[0], (float(lo[0]), float(hi[0]))  # range: of the distance ahead
    predict(REACH_CENTER + [0.4, 0, 0], [-0.25, 0, 0], [0.3, 0, 0])  # warm-up: the first call is much slower
    return predict


def evaluate(path=DATA, n_context=3000, n_test=1500):
    from sklearn.linear_model import LinearRegression

    df = pd.read_csv(path)
    seeds = df.seed.unique()
    rng = np.random.default_rng(0)
    test_seeds = set(rng.choice(seeds, size=len(seeds) // 4, replace=False))  # held-out hand-overs
    tr, te = df[~df.seed.isin(test_seeds)], df[df.seed.isin(test_seeds)]
    tr, te = tr.sample(min(n_context, len(tr)), random_state=0), te.sample(min(n_test, len(te)), random_state=0)
    X_tr, X_te = tr[FEATURES].to_numpy(float), te[FEATURES].to_numpy(float)
    G = te[["gx", "gy", "gz"]].to_numpy()

    t = time.time()
    q = make_model().fit(X_tr, tr["ahead"]).predict(X_te, output_type="quantiles", quantiles=[0.1, 0.5, 0.9])
    tab = np.column_stack([q[1]] + [make_model().fit(X_tr, tr[c]).predict(X_te) for c in TARGETS[1:]])
    t_tab = time.time() - t
    lin = np.column_stack([LinearRegression().fit(X_tr, tr[c]).predict(X_te) for c in TARGETS])
    brake = np.column_stack([te["s_brake"], np.zeros(len(te)), np.zeros(len(te))])  # physics: brakes in a line
    straight = np.column_stack([te["s_shell"], np.zeros(len(te)), np.zeros(len(te))])
    preds = {"TabPFN-3.5-fast": tab, "linear regression": lin,
             "physics: constant braking": brake, "geometry: straight to arm's length": straight}
    print(f"{tr.seed.nunique()} context hand-overs ({len(tr)} looks), {te.seed.nunique()} held-out "
          f"({len(te)} looks); true distance left {te.dist_left.mean() * 100:.0f} +- {te.dist_left.std() * 100:.0f} cm")
    print(f"{'model':36s} {'median error':>13s} {'90th pct':>9s}   by distance left: >25 cm / 10-25 / <10 cm")
    far, mid, near = (te.dist_left > 0.25).to_numpy(), te.dist_left.between(0.10, 0.25).to_numpy(), \
        (te.dist_left < 0.10).to_numpy()
    for name, off in preds.items():
        err = np.linalg.norm(_to_world(X_te, off) - G, axis=1) * 100
        print(f"{name:36s} {np.median(err):10.1f} cm {np.percentile(err, 90):6.1f} cm   "
              + " / ".join(f"{np.median(err[m]):.1f}" for m in (far, mid, near)) + " cm")
    cover = np.mean((te["ahead"].to_numpy() >= q[0]) & (te["ahead"].to_numpy() <= q[2]))
    print(f"TabPFN 10-90% range of the distance ahead holds the truth {cover:.0%} of the time (ideal 80%), "
          f"mean width {np.mean(q[2] - q[0]) * 100:.0f} cm;  predict time {t_tab:.1f}s for {len(te)} looks x 3")
    resid = _to_world(X_te, tab) - G
    np.save(RESIDUALS, np.column_stack([te.dist_left.to_numpy(), resid]))  # train.py --goal-noise
    print(f"saved {len(resid)} held-out errors (with distance left) -> {RESIDUALS}")
    assert np.median(np.linalg.norm(resid, axis=1)) < 0.10, "stop point off by more than 10 cm - check the data"


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--collect", type=int, help="episodes to run and log")
    args = ap.parse_args()
    collect(args.collect) if args.collect else evaluate()
