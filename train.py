"""
Train / evaluate the RL agent (PPO, Stable-Baselines3) on ArmEnv.

  python train.py --steps 200000                 # quick smoke run (~minutes)
  python train.py --steps 3000000 --envs 8       # real run (hours; Colab if needed)
  python train.py --eval models/ppo_arm.zip      # success rate per object
"""

import argparse
import os
from collections import Counter

import numpy as np
import torch as th

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv

from arm_env import ArmEnv


def evaluate(path, episodes):
    model = PPO.load(path)
    env = ArmEnv()
    ok, total = Counter(), Counter()
    for ep in range(episodes):
        obs, _ = env.reset(seed=10_000 + ep)  # fixed seeds: runs are comparable
        done = False
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, _, term, trunc, info = env.step(action)
            done = term or trunc
        total[info["object"]] += 1
        ok[info["object"]] += info["success"]
    print(f"success {sum(ok.values())}/{episodes}  "
          + "  ".join(f"{o} {ok[o]}/{total[o]}" for o in sorted(total)))


def warm_start(model, episodes, epochs=30):
    """Behaviour cloning: teach the policy the scripted controller's actions before PPO.
    Random exploration almost never finds the approach-close-hold sequence on its own."""
    env = ArmEnv()
    obs_l, act_l = [], []
    for ep in range(episodes):
        obs, _ = env.reset(seed=20_000 + ep)  # seeds apart from evaluate()'s
        done = False
        while not done:
            a = env.expert_action()
            obs_l.append(obs)
            act_l.append(a)
            obs, _, term, trunc, _ = env.step(a)
            done = term or trunc
    obs_t = th.as_tensor(np.array(obs_l), device=model.device)
    act_t = th.as_tensor(np.array(act_l), device=model.device)
    opt = th.optim.Adam(model.policy.parameters(), lr=1e-3)
    for epoch in range(epochs):
        perm = th.randperm(len(obs_t))
        for i in range(0, len(perm), 512):
            idx = perm[i:i + 512]
            loss = -model.policy.get_distribution(obs_t[idx]).log_prob(act_t[idx]).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
        if epoch % 10 == 9:
            print(f"warm start epoch {epoch + 1}/{epochs}  loss {loss.item():.3f}")
    print(f"warm start done on {len(obs_t)} expert steps")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=200_000)
    ap.add_argument("--envs", type=int, default=8)
    ap.add_argument("--out", default="models/ppo_arm")
    ap.add_argument("--resume", help="continue training from this .zip")
    ap.add_argument("--bc", type=int, default=0, help="warm-start from N scripted-expert episodes")
    ap.add_argument("--eval", help="evaluate this .zip instead of training")
    ap.add_argument("--episodes", type=int, default=100)
    args = ap.parse_args()

    if args.eval:
        evaluate(args.eval, args.episodes)
        return

    env = make_vec_env(ArmEnv, n_envs=args.envs, vec_env_cls=SubprocVecEnv,
                       monitor_kwargs={"info_keywords": ("success",)})
    if args.resume:
        model = PPO.load(args.resume, env=env)
    else:
        model = PPO("MlpPolicy", env, n_steps=1024, batch_size=256, gamma=0.99,
                    verbose=1, device="cpu")  # small MLP + MuJoCo: CPU is faster than GPU
    if args.bc:
        warm_start(model, args.bc)
        model.save(f"{args.out}_bc")
        evaluate(f"{args.out}_bc.zip", args.episodes)
    model.learn(
        total_timesteps=args.steps,
        callback=CheckpointCallback(max(100_000 // args.envs, 1), os.path.dirname(args.out) or ".",
                                    name_prefix="ppo_arm"),  # next to --out, so runs don't mix
        reset_num_timesteps=not args.resume,
    )
    model.save(args.out)
    print(f"Saved {args.out}.zip")
    evaluate(f"{args.out}.zip", args.episodes)


if __name__ == "__main__":  # required on Windows for SubprocVecEnv
    main()
