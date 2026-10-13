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
- **Full-scale study on HalfCheetah-v5 (COMPLETED):** 4 variants (`baseline`, `dr_0.2`, `dr_0.5`, `dr_0.5_sensing`) x 5 seeds
  for 1,000,000 steps each (20 runs = 20M timesteps), evaluated over the full 25-setting grid with 10 episodes per setting (500 rows).
  - **Retention monotonic gradient:** baseline retention 0.53 (nominal 2407.92), `dr_0.2` retention 0.60 (nominal 2201.58),
    `dr_0.5` retention 0.70 (nominal 1889.23), `dr_0.5_sensing` retention **0.73** (nominal 1918.71).
  - **Mass & actuator gain:** mass return increased from 599.71 (baseline) to **1089.03** (`dr_0.5`, +81.6%); actuator gain
    increased from 756.44 to **1227.99** (`dr_0.5`, +62.3%).
  - **Action delay:** baseline collapsed to 10.32; `dr_0.5_sensing` sustained **989.76** (+979.45 delta, **Welch p = 0.03**).
  - **Worst-case floor:** improved from -331.99 (baseline) to **-101.09** (`dr_0.5_sensing`).
- **Closed-loop deployment check:** a trained policy was served with uvicorn and a client stepped a sensor-noise-perturbed
  simulator through HTTP. The mean return over 5 episodes was 792.0000, **identical (0.000000 diff)** to offline evaluation of the same
  model and episode seeds. `/act` took about 10.65 ms on average (p95 about 19.52 ms) round trip.
- **Windows compatibility:** tested on Windows 11 with PowerShell runner `make.ps1` (`.\make.ps1 test`, `.\make.ps1 robust-all`).

**Not run:** Docker container build, GitHub Actions runner.

## Layout

```
src/robust_rl/    package (perturb, robust_*, closed_loop + project-1 modules)
tests/            44 tests (100% pass)
results/          robust_HalfCheetah-v5 (full study), robust_quick (pipeline check)
docs/PROCESS.md   full guide;  docs/RESULTS_TEMPLATE.md   measured report
Makefile  make.ps1  Dockerfile  pyproject.toml  .github/workflows/ci.yml
```
