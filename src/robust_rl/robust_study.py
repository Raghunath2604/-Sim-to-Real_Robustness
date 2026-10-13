"""Train the baseline and domain-randomised policies for the robustness study (variants x seeds, resumable).

    python -m robust_rl.robust_study --study core --env-id HalfCheetah-v5 --seeds 5 \
        --total-timesteps 1000000 --jobs 4 --out runs/robust
"""

from __future__ import annotations

import argparse
import itertools
import multiprocessing as mp
import os

from .ablate import _run
from .config import PPOConfig

VARIANTS: dict[str, dict] = {
    "baseline": {},  # trained on nominal physics only
    "dr_0.2": {"dr_strength": 0.2},  # friction/mass/gain ~ logU[0.83, 1.2]
    "dr_0.5": {"dr_strength": 0.5},  # ~ logU[0.67, 1.5]
    "dr_0.5_sensing": {"dr_strength": 0.5, "train_obs_noise": 0.1, "train_action_delay": 2},
}
STUDIES: dict[str, list[str]] = {
    "core": list(VARIANTS),
    "tiny": ["baseline", "dr_0.5"],  # pipeline check
}


def build_jobs(study: str, env_id: str, seeds: int, total_timesteps: int, out: str, eval_episodes: int = 5):
    jobs = []
    for variant, seed in itertools.product(STUDIES[study], range(1, seeds + 1)):
        cfg = PPOConfig(env_id=env_id, total_timesteps=total_timesteps, seed=seed, eval_episodes=eval_episodes)
        for k, v in VARIANTS[variant].items():
            setattr(cfg, k, v)
        cfg.log_dir = os.path.join(out, variant, f"seed_{seed}")
        jobs.append((cfg.to_dict(), cfg.log_dir))
    return jobs


def main() -> None:
    ap = argparse.ArgumentParser(description="Train baseline and domain-randomised PPO policies.")
    ap.add_argument("--study", choices=list(STUDIES), default="core")
    ap.add_argument("--env-id", default="HalfCheetah-v5")
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--total-timesteps", type=int, default=1_000_000)
    ap.add_argument("--out", default="runs/robust")
    ap.add_argument("--jobs", type=int, default=1)
    args = ap.parse_args()
    jobs = build_jobs(args.study, args.env_id, args.seeds, args.total_timesteps, args.out)
    print(f"{len(jobs)} runs ({len(STUDIES[args.study])} variants x {args.seeds} seeds) -> {args.out}")
    if args.jobs <= 1:
        for j in jobs:
            print(_run(j))
    else:
        with mp.get_context("spawn").Pool(args.jobs) as pool:
            for msg in pool.imap_unordered(_run, jobs):
                print(msg)


if __name__ == "__main__":
    main()
