"""Closed-loop test of a *served* policy: step a (perturbed) simulator using actions fetched over HTTP.

    # terminal 1
    PPO_MODEL_PATH=runs/robust/dr_0.5/seed_1/model.pt python -m uvicorn --factory robust_rl.serve:create_app --port 8000
    # terminal 2
    python -m robust_rl.closed_loop --url http://localhost:8000 --env-id HalfCheetah-v5 --axis gain --value 1.5

The returned mean return should match offline evaluation of the same model under the same perturbation, which
verifies the whole deployment path (export -> normalisation -> HTTP -> action clipping) end to end.
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Callable

import numpy as np

from .perturb import PerturbationSpec, make_perturbed_env, spec_for

EVAL_SEED = 10_000  # same episode seeds as robust_eval, so numbers are directly comparable


def run_episodes(
    act_fn: Callable[[np.ndarray], np.ndarray], env_id: str, spec: PerturbationSpec, episodes: int
) -> dict:
    env = make_perturbed_env(env_id, fixed=spec, seed=EVAL_SEED)
    returns, latencies = [], []
    import time

    for ep in range(episodes):
        obs, _ = env.reset(seed=EVAL_SEED + ep)
        done, ret = False, 0.0
        while not done:
            t0 = time.perf_counter()
            action = act_fn(obs)
            latencies.append((time.perf_counter() - t0) * 1000)
            obs, r, term, trunc, _ = env.step(action)
            ret += float(r)
            done = term or trunc
        returns.append(ret)
    env.close()
    return {
        "episodes": episodes,
        "mean_return": float(np.mean(returns)),
        "returns": returns,
        "mean_act_latency_ms": float(np.mean(latencies)),
        "p95_act_latency_ms": float(np.percentile(latencies, 95)),
    }


def http_act_fn(client, base_url: str = "") -> Callable[[np.ndarray], np.ndarray]:
    """`client` is anything with a requests/httpx-style .post(url, json=...) (e.g. httpx.Client or a TestClient)."""

    def act(obs: np.ndarray) -> np.ndarray:
        resp = client.post(f"{base_url}/act", json={"observation": [float(x) for x in obs]})
        resp.raise_for_status()
        return np.asarray(resp.json()["action"], dtype=np.float32)

    return act


def main() -> None:
    ap = argparse.ArgumentParser(description="Closed-loop test of a served policy.")
    ap.add_argument("--url", default="http://localhost:8000")
    ap.add_argument("--env-id", required=True)
    ap.add_argument("--axis", default="nominal", help="friction|mass|gain|obs_noise|action_delay|nominal")
    ap.add_argument("--value", type=float, default=1.0)
    ap.add_argument("--episodes", type=int, default=3)
    args = ap.parse_args()
    import httpx

    spec = PerturbationSpec() if args.axis == "nominal" else spec_for(args.axis, args.value)
    with httpx.Client(timeout=10.0) as client:
        out = run_episodes(http_act_fn(client, args.url), args.env_id, spec, args.episodes)
    print(json.dumps({k: v for k, v in out.items() if k != "returns"}, indent=2))


if __name__ == "__main__":
    main()
