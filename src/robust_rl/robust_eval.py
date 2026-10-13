"""Evaluate trained policies across a grid of perturbations.

    python -m robust_rl.robust_eval --runs runs/robust --out results/robust/eval.csv --episodes 10

Every model is evaluated *deterministically* on the same episode seeds, once at nominal settings and once per
(axis, value) in the grid, with exactly one axis perturbed at a time.
"""

from __future__ import annotations

import argparse
import glob
import json
import multiprocessing as mp
import os

import pandas as pd
import torch

from .checkpoint import load_checkpoint
from .evaluate import evaluate_policy
from .perturb import PerturbationSpec, make_perturbed_env, spec_for

# Values are chosen to extend well beyond typical training ranges (e.g. DR strength 0.5 covers x0.67 .. x1.5)
DEFAULT_GRID: dict[str, list[float]] = {
    "friction": [0.25, 0.5, 0.75, 1.5, 2.0, 3.0],
    "mass": [0.5, 0.75, 1.5, 2.0, 3.0],
    "gain": [0.5, 0.75, 1.25, 1.5, 2.0],
    "obs_noise": [0.05, 0.1, 0.2, 0.3],
    "action_delay": [1, 2, 3, 5],
}
QUICK_GRID: dict[str, list[float]] = {
    "friction": [0.5, 2.0],
    "mass": [0.5, 2.0],
    "gain": [0.5, 1.5],
    "obs_noise": [0.2],
    "action_delay": [2],
}
EVAL_SEED = 10_000  # the same initial states for every model and every perturbation


def eval_one_model(job: tuple[str, str, str, int, dict, int]) -> list[dict]:
    """job = (variant, seed_name, model_path, episodes, grid, ...). Returns long-format rows."""
    variant, seed_name, model_path, episodes, grid, _ = job
    torch.set_num_threads(1)
    agent, obs_norm, cfg, lo, hi = load_checkpoint(model_path)
    settings: list[tuple[str, float, PerturbationSpec]] = [("nominal", float("nan"), PerturbationSpec())]
    for axis, values in grid.items():
        settings += [(axis, float(v), spec_for(axis, v)) for v in values]
    rows = []
    for axis, value, spec in settings:
        env = make_perturbed_env(cfg.env_id, fixed=spec, seed=EVAL_SEED)
        res = evaluate_policy(agent, obs_norm, cfg.env_id, lo, hi, episodes, seed=EVAL_SEED, env=env)
        env.close()
        rows.append(
            {
                "variant": variant,
                "seed": seed_name,
                "env_id": cfg.env_id,
                "axis": axis,
                "value": value,
                "mean_return": res["mean_return"],
                "std_return": res["std_return"],
                "mean_length": res["mean_length"],
                "episodes": episodes,
            }
        )
    return rows


def discover_models(runs_dir: str) -> list[tuple[str, str, str]]:
    out = []
    for path in sorted(glob.glob(os.path.join(runs_dir, "*", "seed_*", "model.pt"))):
        run_dir = os.path.dirname(path)
        out.append((os.path.basename(os.path.dirname(run_dir)), os.path.basename(run_dir), path))
    return out


def evaluate_all(runs_dir: str, episodes: int, grid: dict, jobs: int = 1) -> pd.DataFrame:
    models = discover_models(runs_dir)
    if not models:
        raise SystemExit(f"no trained models found under {runs_dir} (expected <variant>/seed_<k>/model.pt)")
    work = [(v, s, p, episodes, grid, 0) for v, s, p in models]
    n_settings = 1 + sum(len(v) for v in grid.values())
    print(f"{len(work)} models x {n_settings} settings x {episodes} episodes")
    if jobs <= 1:
        results = [eval_one_model(w) for w in work]
    else:
        with mp.get_context("spawn").Pool(jobs) as pool:
            results = pool.map(eval_one_model, work)
    return pd.DataFrame([r for rows in results for r in rows])


def main() -> None:
    ap = argparse.ArgumentParser(description="Evaluate trained policies under perturbations.")
    ap.add_argument("--runs", required=True, help="directory with <variant>/seed_<k>/model.pt")
    ap.add_argument("--out", default="results/robust/eval.csv")
    ap.add_argument("--episodes", type=int, default=10)
    ap.add_argument("--quick", action="store_true", help="use a tiny grid (pipeline check)")
    ap.add_argument("--grid-json", default=None, help="optional JSON file overriding the grid")
    ap.add_argument("--jobs", type=int, default=1)
    args = ap.parse_args()
    grid = QUICK_GRID if args.quick else DEFAULT_GRID
    if args.grid_json:
        grid = json.load(open(args.grid_json))
    df = evaluate_all(args.runs, args.episodes, grid, args.jobs)
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    df.to_csv(args.out, index=False)
    print(f"wrote {len(df)} rows to {args.out}")


if __name__ == "__main__":
    main()
