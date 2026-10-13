from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from robust_rl.config import PPOConfig
from robust_rl.robust_eval import QUICK_GRID, evaluate_all
from robust_rl.robust_report import AXES, plot_all, seed_metrics, summarize
from robust_rl.robust_study import STUDIES, VARIANTS, build_jobs
from robust_rl.train import train


def synthetic_eval(seeds=(1, 2, 3), grid=None):
    """baseline degrades under perturbation; dr degrades less but has a lower nominal return."""
    rng = np.random.default_rng(0)
    grid = grid or {
        "friction": [0.5, 2.0],
        "mass": [0.5, 2.0],
        "gain": [0.5, 1.5],
        "obs_noise": [0.2],
        "action_delay": [2],
    }
    rows = []
    for variant, nominal, drop in (("baseline", 100.0, 0.5), ("dr_0.5", 90.0, 0.15)):
        for s in seeds:
            n = nominal + rng.normal(0, 1)
            rows.append(
                dict(variant=variant, seed=f"seed_{s}", env_id="X", axis="nominal", value=np.nan, mean_return=n)
            )
            for axis, vals in grid.items():
                for v in vals:
                    rows.append(
                        dict(
                            variant=variant,
                            seed=f"seed_{s}",
                            env_id="X",
                            axis=axis,
                            value=float(v),
                            mean_return=n * (1 - drop) + rng.normal(0, 1),
                        )
                    )
    return pd.DataFrame(rows)


def test_seed_metrics_and_summary_recover_the_planted_effect():
    sm = seed_metrics(synthetic_eval())
    assert len(sm) == 6 and (sm.nominal > 80).all()
    overall, per_axis = summarize(sm)
    assert list(overall.variant) == ["baseline", "dr_0.5"]
    base, dr = overall.iloc[0], overall.iloc[1]
    assert dr["mean_perturbed"] > base["mean_perturbed"]  # DR more robust
    assert dr["nominal"] < base["nominal"]  # ...at a cost in nominal return
    assert dr["retention"] > base["retention"]
    assert dr["perturbed_welch_p"] < 0.01
    assert set(per_axis.axis) == {"friction", "mass", "gain", "obs_noise", "action_delay"}


def test_retention_is_nan_when_nominal_return_is_not_positive():
    df = synthetic_eval()
    df.loc[(df.variant == "baseline") & (df.axis == "nominal"), "mean_return"] = -5.0
    sm = seed_metrics(df)
    assert sm.loc[sm.variant == "baseline", "retention"].isna().all()


def test_plots_are_written(tmp_path):
    written = plot_all(synthetic_eval(), str(tmp_path))
    assert len(written) == len(AXES) + 1
    assert all((tmp_path / Path(f).name).exists() for f in written)


def test_build_jobs_wires_dr_settings(tmp_path):
    jobs = build_jobs("core", "HalfCheetah-v5", 2, 1000, str(tmp_path))
    assert len(jobs) == len(STUDIES["core"]) * 2
    cfgs = {Path(j[1]).parent.name: j[0] for j in jobs}
    assert cfgs["baseline"]["dr_strength"] == 0.0
    assert cfgs["dr_0.5"]["dr_strength"] == 0.5
    assert cfgs["dr_0.5_sensing"]["train_action_delay"] == 2 and VARIANTS["dr_0.5_sensing"]["train_obs_noise"] == 0.1


@pytest.fixture(scope="module")
def tiny_runs(tmp_path_factory):
    root = tmp_path_factory.mktemp("runs")
    for variant, kw in (
        ("baseline", {}),
        ("dr_0.5", {"dr_strength": 0.5, "train_obs_noise": 0.1, "train_action_delay": 1}),
    ):
        train(
            PPOConfig(
                env_id="InvertedPendulum-v5",
                total_timesteps=256,
                num_steps=128,
                num_minibatches=2,
                update_epochs=1,
                eval_episodes=1,
                log_dir=str(root / variant / "seed_1"),
                **kw,
            )
        )
    return root


def test_training_with_domain_randomisation_writes_artifacts(tiny_runs):
    assert (tiny_runs / "dr_0.5" / "seed_1" / "model.pt").exists()
    cfg = PPOConfig.from_dict(__import__("json").load(open(tiny_runs / "dr_0.5" / "seed_1" / "config.json")))
    assert cfg.randomization().enabled() and cfg.dr_strength == 0.5


def test_evaluate_all_on_trained_models(tiny_runs):
    df = evaluate_all(str(tiny_runs), episodes=1, grid=QUICK_GRID, jobs=1)
    n_settings = 1 + sum(len(v) for v in QUICK_GRID.values())
    assert len(df) == 2 * n_settings
    assert set(df.variant) == {"baseline", "dr_0.5"} and np.isfinite(df.mean_return).all()
    assert (df[df.axis == "nominal"].mean_length > 0).all()


def test_served_policy_matches_offline_evaluation_under_perturbation(tiny_runs):
    from fastapi.testclient import TestClient

    from robust_rl.checkpoint import load_checkpoint
    from robust_rl.closed_loop import EVAL_SEED, http_act_fn, run_episodes
    from robust_rl.evaluate import evaluate_policy
    from robust_rl.perturb import make_perturbed_env, spec_for
    from robust_rl.serve import create_app

    model = str(tiny_runs / "dr_0.5" / "seed_1" / "model.pt")
    spec = spec_for("gain", 1.5)
    served = run_episodes(http_act_fn(TestClient(create_app(model))), "InvertedPendulum-v5", spec, episodes=2)
    agent, norm, cfg, lo, hi = load_checkpoint(model)
    env = make_perturbed_env(cfg.env_id, fixed=spec, seed=EVAL_SEED)
    offline = evaluate_policy(agent, norm, cfg.env_id, lo, hi, episodes=2, seed=EVAL_SEED, env=env)
    assert abs(served["mean_return"] - offline["mean_return"]) < 1e-3
