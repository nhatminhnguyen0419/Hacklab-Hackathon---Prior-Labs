"""
Does acting on TabPFN's "will it hold?" help? The same 200 held-out episodes, the agent acting on
TabPFN's stop point, with the re-grasp reflex off and on (sim/env.py: under 50% at first touch, the
hand opens and grasps again while the person keeps holding). Episodes are identical up to the first
touch, so each re-grasp can be judged: saved, spoiled, or no change.

  python regrasp_eval.py           # both runs in parallel, then the comparison
"""

import sys
from multiprocessing import Pool

import numpy as np
from stable_baselines3 import PPO

SEEDS = range(10_000, 10_200)  # train.py evaluate()'s held-out seeds


def run(reflex):
    from grasp_outcome import load_predictor as load_hold
    from sim.env import ArmEnv
    from trajectory import load_predictor as load_goal
    model = PPO.load("models/ppo_arm.zip", device="cpu")
    env = ArmEnv(goal_predictor=load_goal(), hold_predictor=load_hold() if reflex else None)
    out = {}
    for seed in SEEDS:
        obs, _ = env.reset(seed=seed)
        while True:
            obs, _, term, trunc, info = env.step(model.predict(obs, deterministic=True)[0])
            if term or trunc:
                break
        out[seed] = (bool(info["success"]), env.regrasps, env.obj)
    return out


if __name__ == "__main__":
    with Pool(2) as pool:
        off, on = pool.map(run, [False, True])
    s_off, s_on = sum(v[0] for v in off.values()), sum(v[0] for v in on.values())
    tried = [s for s in SEEDS if on[s][1]]
    saved = sum(on[s][0] and not off[s][0] for s in tried)
    spoiled = sum(off[s][0] and not on[s][0] for s in tried)
    print(f"reflex off: {s_off}/{len(SEEDS)}   reflex on: {s_on}/{len(SEEDS)}")
    print(f"re-grasps triggered: {len(tried)}  saved {saved}  spoiled {spoiled}  "
          f"unchanged {len(tried) - saved - spoiled}   by object: "
          + str({o: sum(on[s][2] == o for s in tried) for o in sorted({on[s][2] for s in tried})}))
    sys.exit(0)
