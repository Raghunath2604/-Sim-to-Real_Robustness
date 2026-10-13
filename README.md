# sim2real-robustness

**Project 2  of 3.** Does domain randomisation make a PPO policy more robust to changes in physics and sensing, what does it
cost in nominal performance, and does robustness transfer from one kind of perturbation to another?

This repo trains a baseline and domain-randomised PPO policies (PPO implemented from scratch; a self-contained copy of
project 1's code is included), evaluates every policy across a grid of **friction, mass, actuator-gain, sensor-noise and
action-delay** perturbations, and reports robustness curves with 95% confidence intervals across seeds. It also deploys a
policy behind an HTTP API and checks it in closed loop against offline evaluation.

> **Start here:** [`docs/PROCESS.md`](docs/PROCESS.md): the complete guide from setup to write-up to deployment.

> This is a study of a *perturbed simulator*, a proxy for the sim-to-real gap. No real hardware is involved.

## Quickstart

```bash
python -m venv .venv && source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -e ".[dev]"

make test            # 44 tests (~25 s)
make robust-quick    # pipeline check on an easy task: train -> evaluate -> report (10-15 min)
```

The real study (HalfCheetah, 4 variants x 5 seeds) is `make robust-all RJOBS=4`; see the guide for budgets.
No `make`? Every target is a plain `python -m robust_rl....` command listed in [`docs/PROCESS.md`](docs/PROCESS.md).

## Pipeline

```
robust_study  ->  runs/<variant>/seed_<k>/model.pt      train baseline + DR variants x seeds (resumable)
robust_eval   ->  eval.csv                              every model x every perturbation, identical episode seeds
robust_report ->  summary.md, per_axis.csv, plots       mean +- 95% CI across seeds, Welch tests vs baseline
export/serve  ->  policy.ts, FastAPI /act               deployment
closed_loop   ->  served return vs offline return       end-to-end deployment check, latency
```

| New module | Purpose |
|---|---|
| `perturb.py` | `PerturbationSpec`, `DomainRandomization`, `PerturbedEnv` (MuJoCo physics + sensing + latency) |
| `robust_study.py` | variants: `baseline`, `dr_0.2`, `dr_0.5`, `dr_0.5_sensing` |
| `robust_eval.py` | evaluation grid (extends beyond training ranges), parallel over models |
| `robust_report.py` | per-axis robustness curves, nominal-cost vs robustness tables |
| `closed_loop.py` | steps a perturbed simulator with actions fetched over HTTP |

## Verified status (what was actually run)

Build environment: Python 3.12, Linux, shared CPU; torch 2.14.1, gymnasium 1.4.0, mujoco 3.15.0.

- `pytest`: **44 passed**; `ruff check` and `ruff format --check` clean.
- **Perturbation wrapper**, checked on HalfCheetah-v5 and InvertedPendulum-v5: the nominal wrapper is bit-identical to the
  plain environment; each axis changes the dynamics; mass/inertia/friction/gain scale exactly, do not accumulate across
  resets and restore to nominal; delayed actions are the older ones; DR samples stay in the intended range.
  InvertedPendulum has no ground contact, so the friction axis has no effect there (as expected).
- **End-to-end mini-study** (pipeline check, *not* the real experiment): InvertedPendulum-v5, `baseline` vs `dr_0.5`,
  3 seeds each, 61,440 steps, evaluated over the full grid with 5 episodes per setting.
  - Both variants reached the maximum return (1000) at nominal settings on every seed.
  - Both stayed at 1000 across friction x0.25..x3, mass x0.5..x3 and gain x0.5..x2: this task is simply insensitive to those ranges.
  - Sensor noise 0.2-0.3 and **any** action delay (>= 1 step) hurt both variants: with delay, both fell to a return of about 10.
  - Mean perturbed return: baseline 796 vs `dr_0.5` 795 (Welch p = 0.94, n = 3).
  - **Takeaway:** the machinery works end to end, and this easy environment cannot tell the variants apart. It is
    *not* evidence for or against domain randomisation.
- **Closed-loop deployment check:** a trained policy was served with uvicorn and a client stepped a sensor-noise-perturbed
  simulator through HTTP. The mean return over 5 episodes was 943.8, **identical** to offline evaluation of the same
  model and episode seeds. `/act` took about 3.6 ms on average (p95 about 5.0 ms) round trip on the build machine.

**Not run:** the real HalfCheetah/Hopper study (that is your job and the point of the project), Docker build, GitHub
Actions, Windows/macOS.

## Layout

```
src/robust_rl/    package (perturb, robust_*, closed_loop + project-1 modules)
tests/            44 tests
docs/PROCESS.md   full guide;  docs/RESULTS_TEMPLATE.md   write-up skeleton
Makefile  Dockerfile  pyproject.toml  .github/workflows/ci.yml
```
