"""Visual rollout and comparative demonstration module for sim2real-robustness.

Renders trained policies under physical or sensing perturbations to MP4/GIF videos.
Supports side-by-side comparison between baseline and robust policies under identical disturbances.

Usage:
    # Single model rollout
    python -m robust_rl.demo --model runs/robust_HalfCheetah-v5/dr_0.5_sensing/seed_1/model.pt \
        --axis action_delay --value 2 --steps 300 --out demo_delayed.gif

    # Side-by-side comparison (Baseline vs Robust policy)
    python -m robust_rl.demo --compare \
        --baseline runs/robust_HalfCheetah-v5/baseline/seed_1/model.pt \
        --robust runs/robust_HalfCheetah-v5/dr_0.5_sensing/seed_1/model.pt \
        --axis action_delay --value 2 --steps 300 --out assets/demo_comparison.gif
"""

from __future__ import annotations

import argparse
from pathlib import Path

import imageio
import numpy as np
import torch
from PIL import Image, ImageDraw

from .checkpoint import load_checkpoint
from .perturb import PerturbationSpec, make_perturbed_env, spec_for


def render_policy_rollout(
    model_path: str,
    env_id: str,
    spec: PerturbationSpec,
    max_steps: int = 300,
    seed: int = 10_000,
) -> tuple[list[np.ndarray], float]:
    """Rolls out a policy in a perturbed environment and collects RGB frames."""
    torch.set_num_threads(1)
    agent, obs_norm, cfg, lo, hi = load_checkpoint(model_path)
    env = make_perturbed_env(env_id, fixed=spec, seed=seed, render_mode="rgb_array")
    obs, _ = env.reset(seed=seed)

    frames: list[np.ndarray] = []
    total_reward = 0.0

    for _ in range(max_steps):
        frame = env.render()
        if frame is not None:
            frames.append(frame)

        # Observation normalization and deterministic action
        x = obs_norm(obs[None, :]) if obs_norm is not None else obs[None, :].astype(np.float32)
        with torch.no_grad():
            act = agent.act_deterministic(torch.as_tensor(x)).cpu().numpy()[0]
        act = np.clip(act, lo, hi)

        obs, r, term, trunc, _ = env.step(act)
        total_reward += float(r)
        if term or trunc:
            break

    env.close()
    return frames, total_reward


def add_banner(frame: np.ndarray, title: str, subtitle: str, color=(30, 30, 30)) -> np.ndarray:
    """Adds a stylish header banner to an RGB frame."""
    h, w, c = frame.shape
    banner_height = 50
    composite = Image.new("RGB", (w, h + banner_height), color=color)

    # Draw header text
    draw = ImageDraw.Draw(composite)
    draw.text((12, 8), title, fill=(255, 255, 255))
    draw.text((12, 28), subtitle, fill=(200, 200, 200))

    # Paste simulation frame below banner
    frame_img = Image.fromarray(frame)
    composite.paste(frame_img, (0, banner_height))
    return np.array(composite)


def create_comparison_video(
    baseline_path: str,
    robust_path: str,
    env_id: str,
    spec: PerturbationSpec,
    axis_name: str,
    value: float,
    out_path: str,
    max_steps: int = 300,
    seed: int = 10_000,
    fps: int = 25,
) -> None:
    """Renders baseline and robust policies under identical disturbance and saves side-by-side video/GIF."""
    print(f"Rendering Baseline: {baseline_path} under {axis_name}={value}...")
    base_frames, base_ret = render_policy_rollout(baseline_path, env_id, spec, max_steps=max_steps, seed=seed)

    print(f"Rendering Robust:   {robust_path} under {axis_name}={value}...")
    rob_frames, rob_ret = render_policy_rollout(robust_path, env_id, spec, max_steps=max_steps, seed=seed)

    n_frames = min(len(base_frames), len(rob_frames))
    combined_frames = []

    sub_title = f"{axis_name} = {value}"
    base_label = f"Baseline PPO ({sub_title}) | Return: {base_ret:.1f}"
    rob_label = f"Domain-Randomized PPO ({sub_title}) | Return: {rob_ret:.1f}"

    for i in range(n_frames):
        b_frame = add_banner(base_frames[i], "BASELINE POLICY", base_label, color=(40, 20, 20))
        r_frame = add_banner(rob_frames[i], "DOMAIN-RANDOMIZED POLICY", rob_label, color=(20, 40, 30))
        # Concatenate horizontally
        side_by_side = np.concatenate([b_frame, r_frame], axis=1)
        combined_frames.append(side_by_side)

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    print(f"Saving {len(combined_frames)} frames to {out_path}...")
    if out_path.endswith(".gif"):
        imageio.mimsave(out_path, combined_frames, fps=fps, loop=0)
    else:
        imageio.mimsave(out_path, combined_frames, fps=fps)
    print(f"Successfully generated comparison: {out_path}")


def main() -> None:
    ap = argparse.ArgumentParser(description="Render visual demonstrations of policies under perturbation.")
    ap.add_argument("--model", default="runs/robust_HalfCheetah-v5/dr_0.5_sensing/seed_1/model.pt")
    ap.add_argument("--baseline", default="runs/robust_HalfCheetah-v5/baseline/seed_1/model.pt")
    ap.add_argument("--robust", default="runs/robust_HalfCheetah-v5/dr_0.5_sensing/seed_1/model.pt")
    ap.add_argument("--compare", action="store_true", help="Generate side-by-side comparison video")
    ap.add_argument("--env-id", default="HalfCheetah-v5")
    ap.add_argument("--axis", default="action_delay", help="friction|mass|gain|obs_noise|action_delay|nominal")
    ap.add_argument("--value", type=float, default=2.0)
    ap.add_argument("--steps", type=int, default=250)
    ap.add_argument("--fps", type=int, default=25)
    ap.add_argument("--out", default="assets/demo_comparison.gif")
    ap.add_argument("--seed", type=int, default=10_000)
    args = ap.parse_args()

    spec = PerturbationSpec() if args.axis == "nominal" else spec_for(args.axis, args.value)

    if args.compare:
        create_comparison_video(
            baseline_path=args.baseline,
            robust_path=args.robust,
            env_id=args.env_id,
            spec=spec,
            axis_name=args.axis,
            value=args.value,
            out_path=args.out,
            max_steps=args.steps,
            seed=args.seed,
            fps=args.fps,
        )
    else:
        print(f"Rendering rollout for {args.model} under {args.axis}={args.value}...")
        frames, ret = render_policy_rollout(args.model, args.env_id, spec, max_steps=args.steps, seed=args.seed)
        annotated = [
            add_banner(
                f,
                f"{args.env_id} Rollout",
                f"Perturbation: {args.axis}={args.value} | Total Return: {ret:.1f}",
            )
            for f in frames
        ]
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        imageio.mimsave(args.out, annotated, fps=args.fps)
        print(f"Saved rollout to {args.out} (Return: {ret:.2f})")


if __name__ == "__main__":
    main()
