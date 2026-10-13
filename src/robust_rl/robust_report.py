"""Turn eval.csv into robustness curves (mean +- 95% CI across seeds) and summary tables.

python -m robust_rl.robust_report --eval results/robust/eval.csv --runs runs/robust --out results/robust
"""

from __future__ import annotations

import argparse
import json
import os
import warnings

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy import stats  # noqa: E402

from .analysis import ci95_halfwidth, to_markdown  # noqa: E402

AXES = ["friction", "mass", "gain", "obs_noise", "action_delay"]
NOMINAL_X = {"friction": 1.0, "mass": 1.0, "gain": 1.0, "obs_noise": 0.0, "action_delay": 0.0}
LOG_AXES = {"friction", "mass", "gain"}
LABELS = {
    "friction": "friction scale (x nominal)",
    "mass": "mass scale (x nominal)",
    "gain": "actuator gain scale (x nominal)",
    "obs_noise": "sensor noise (fraction of obs std)",
    "action_delay": "action delay (control steps)",
}


def seed_metrics(df: pd.DataFrame) -> pd.DataFrame:
    """One row per (variant, seed): nominal return, robustness aggregates over the grid."""
    rows = []
    for (variant, seed), g in df.groupby(["variant", "seed"]):
        nominal = float(g.loc[g.axis == "nominal", "mean_return"].iloc[0])
        pert = g[g.axis != "nominal"]
        row = {
            "variant": variant,
            "seed": seed,
            "nominal": nominal,
            "mean_perturbed": float(pert.mean_return.mean()),
            "worst_case": float(pert.mean_return.min()),
            "retention": float(pert.mean_return.mean() / nominal) if nominal > 0 else np.nan,
        }
        for axis, ga in pert.groupby("axis"):
            row[f"mean_{axis}"] = float(ga.mean_return.mean())
        rows.append(row)
    return pd.DataFrame(rows)


def _mean_ci(x: pd.Series) -> tuple[float, float]:
    v = x.dropna().to_numpy(dtype=float)
    if len(v) == 0:
        return np.nan, np.nan
    return float(v.mean()), float(ci95_halfwidth(v[:, None])[0])


def _welch(a: np.ndarray, b: np.ndarray) -> float:
    if len(a) < 2 or len(b) < 2:
        return np.nan
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        return float(stats.ttest_ind(a, b, equal_var=False).pvalue)


def summarize(sm: pd.DataFrame, baseline: str = "baseline") -> tuple[pd.DataFrame, pd.DataFrame]:
    """Returns (overall, per_axis) tables with mean, 95% CI half-width across seeds and Welch p vs baseline."""
    variants = list(dict.fromkeys(sm.variant))
    if baseline in variants:
        variants = [baseline] + [v for v in variants if v != baseline]

    def block(metric: str, v: str) -> np.ndarray:
        return sm.loc[sm.variant == v, metric].dropna().to_numpy(dtype=float)

    overall = []
    for v in variants:
        row = {"variant": v, "n_seeds": int((sm.variant == v).sum())}
        for metric in ("nominal", "mean_perturbed", "worst_case", "retention"):
            m, ci = _mean_ci(sm.loc[sm.variant == v, metric])
            row[metric] = m
            row[f"{metric}_ci95"] = ci
        if baseline in variants and v != baseline:
            row["nominal_vs_baseline"] = row["nominal"] - float(np.mean(block("nominal", baseline)))
            row["perturbed_vs_baseline"] = row["mean_perturbed"] - float(np.mean(block("mean_perturbed", baseline)))
            row["perturbed_welch_p"] = _welch(block("mean_perturbed", v), block("mean_perturbed", baseline))
        overall.append(row)

    per_axis = []
    for v in variants:
        for axis in AXES:
            col = f"mean_{axis}"
            if col not in sm:
                continue
            m, ci = _mean_ci(sm.loc[sm.variant == v, col])
            row = {"variant": v, "axis": axis, "mean_return": m, "ci95": ci}
            if baseline in variants and v != baseline:
                row["delta_vs_baseline"] = m - float(np.mean(block(col, baseline)))
                row["welch_p"] = _welch(block(col, v), block(col, baseline))
            per_axis.append(row)
    return pd.DataFrame(overall), pd.DataFrame(per_axis)


def train_ranges(runs_dir: str | None, variants: list[str]) -> dict[str, tuple[float, float]]:
    """Widest perturbation range seen in training, per axis, read from each variant's config.json."""
    out: dict[str, tuple[float, float]] = {}
    if not runs_dir:
        return out
    for v in variants:
        path = os.path.join(runs_dir, v, "seed_1", "config.json")
        if not os.path.exists(path):
            continue
        cfg = json.load(open(path))
        s = float(cfg.get("dr_strength", 0.0))
        if s > 0:
            for p in str(cfg.get("dr_params", "")).split(","):
                if p in ("friction", "mass", "gain"):
                    lo, hi = 1 / (1 + s), 1 + s
                    old = out.get(p, (lo, hi))
                    out[p] = (min(old[0], lo), max(old[1], hi))
        if float(cfg.get("train_obs_noise", 0.0)) > 0:
            out["obs_noise"] = (0.0, max(out.get("obs_noise", (0, 0))[1], float(cfg["train_obs_noise"])))
        if int(cfg.get("train_action_delay", 0)) > 0:
            out["action_delay"] = (0.0, max(out.get("action_delay", (0, 0))[1], float(cfg["train_action_delay"])))
    return out


def axis_curves(df: pd.DataFrame, axis: str) -> dict[str, pd.DataFrame]:
    """variant -> DataFrame(x, mean, ci) with the nominal point inserted at the axis' nominal value."""
    pert = df[df.axis == axis][["variant", "seed", "value", "mean_return"]].copy()
    nom = df[df.axis == "nominal"][["variant", "seed", "mean_return"]].copy()
    nom["value"] = NOMINAL_X[axis]
    allx = pd.concat([pert, nom[["variant", "seed", "value", "mean_return"]]])
    curves = {}
    for v, g in allx.groupby("variant"):
        piv = g.pivot_table(index="seed", columns="value", values="mean_return")
        m = piv.mean(axis=0)
        ci = pd.Series(ci95_halfwidth(piv.to_numpy(dtype=float), axis=0), index=piv.columns)
        curves[v] = pd.DataFrame({"x": piv.columns.to_numpy(dtype=float), "mean": m.to_numpy(), "ci": ci.to_numpy()})
    return curves


def _draw(ax, curves, axis, ranges, show_legend=True):
    for v, c in curves.items():
        ax.plot(c["x"], c["mean"], marker="o", ms=4, label=v)
        ax.fill_between(c["x"], c["mean"] - c["ci"], c["mean"] + c["ci"], alpha=0.15)
    if axis in ranges:
        lo, hi = ranges[axis]
        ax.axvspan(lo, hi, color="gray", alpha=0.12, label="widest training range")
    ax.axvline(NOMINAL_X[axis], color="k", ls=":", lw=0.8)
    if axis in LOG_AXES:
        ax.set_xscale("log")
        xs = sorted({round(x, 3) for c in curves.values() for x in c["x"]})
        ax.set_xticks(xs)
        ax.set_xticklabels([f"{x:g}" for x in xs], fontsize=7)
        ax.minorticks_off()
    ax.set_xlabel(LABELS[axis])
    ax.set_ylabel("deterministic return")
    ax.grid(alpha=0.3)
    if show_legend:
        ax.legend(fontsize=7)


def plot_all(df: pd.DataFrame, out_dir: str, runs_dir: str | None = None) -> list[str]:
    os.makedirs(out_dir, exist_ok=True)
    present = [a for a in AXES if (df.axis == a).any()]
    ranges = train_ranges(runs_dir, list(dict.fromkeys(df.variant)))
    written = []
    for axis in present:
        curves = axis_curves(df, axis)
        fig, ax = plt.subplots(figsize=(6.5, 4.2))
        _draw(ax, curves, axis, ranges)
        ax.set_title(f"Return vs {axis} (mean, 95% CI across seeds)")
        fig.tight_layout()
        path = os.path.join(out_dir, f"robustness_{axis}.png")
        fig.savefig(path, dpi=140)
        plt.close(fig)
        written.append(path)
    ncols = 3
    nrows = int(np.ceil(len(present) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 3.8 * nrows), squeeze=False)
    for i, axis in enumerate(present):
        _draw(axes[i // ncols][i % ncols], axis_curves(df, axis), axis, ranges, show_legend=(i == 0))
        axes[i // ncols][i % ncols].set_title(axis)
    for j in range(len(present), nrows * ncols):
        axes[j // ncols][j % ncols].axis("off")
    fig.tight_layout()
    path = os.path.join(out_dir, "robustness_grid.png")
    fig.savefig(path, dpi=140)
    plt.close(fig)
    written.append(path)
    return written


def main() -> None:
    ap = argparse.ArgumentParser(description="Robustness report from eval.csv.")
    ap.add_argument("--eval", required=True)
    ap.add_argument("--runs", default=None, help="runs directory (to shade the training ranges)")
    ap.add_argument("--out", default="results/robust")
    args = ap.parse_args()
    df = pd.read_csv(args.eval)
    os.makedirs(args.out, exist_ok=True)
    sm = seed_metrics(df)
    overall, per_axis = summarize(sm)
    sm.to_csv(os.path.join(args.out, "per_seed.csv"), index=False)
    overall.to_csv(os.path.join(args.out, "summary.csv"), index=False)
    per_axis.to_csv(os.path.join(args.out, "per_axis.csv"), index=False)
    with open(os.path.join(args.out, "summary.md"), "w") as f:
        f.write(
            "## Overall\n\n" + to_markdown(overall) + "\n\n## Per perturbation axis\n\n" + to_markdown(per_axis) + "\n"
        )
    plots = plot_all(df, args.out, args.runs)
    print(to_markdown(overall))
    print()
    print(to_markdown(per_axis))
    print(f"\nwrote tables and {len(plots)} plots to {args.out}/")


if __name__ == "__main__":
    main()
