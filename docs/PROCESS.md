# Sim-to-real robustness study: the complete process, end to end

Project 2 of 3. It builds on the PPO implementation from project 1 (a self-contained copy is included, so you do **not**
need project 1 installed) and asks a research question:

> **Does training with domain randomisation make a PPO policy more robust to changes in physics and sensing, what does
> it cost in nominal performance, and does robustness to some perturbations transfer to others?**

You will train a baseline and several domain-randomised policies with several seeds, evaluate every policy across a
grid of perturbations (friction, mass, actuator strength, sensor noise, actuation delay), and report robustness curves
with confidence intervals. Then you deploy a policy and verify it in closed loop through the HTTP API.

> **Honesty rule.** This study uses a *perturbed simulator* as a proxy for the sim-to-real gap. It does not involve real
> hardware, and you should say so in your write-up. Only report numbers you measured yourself.

---

## 0. Why this project matters

Robot-learning teams care less about "reward in the training simulator" and more about whether a policy keeps working
when the world differs from the simulator. This project shows you can:

| Skill | Evidence in this repo |
|---|---|
| Model the sim-to-real gap | `perturb.py`: physics and sensing perturbations applied to MuJoCo models |
| Domain randomisation | `DomainRandomization` + training flags `--dr-strength`, `--train-obs-noise`, `--train-action-delay` |
| Rigorous evaluation | `robust_eval.py`: identical episode seeds, one axis at a time, deterministic policy, beyond-training-range grid |
| Honest statistics | `robust_report.py`: mean +- 95% CI across seeds, Welch tests, nominal-cost vs robustness trade-off |
| Integrating policies into systems | `export.py`, `serve.py`, `closed_loop.py`: served policy checked against offline evaluation |

---

## 1. Setup

Python 3.10+ (3.11 recommended), ~3 GB disk, CPU only.

```bash
unzip sim2real-robustness.zip && cd sim2real-robustness
python -m venv .venv && source .venv/bin/activate          # Windows: .venv\Scripts\Activate.ps1
pip install torch --index-url https://download.pytorch.org/whl/cpu   # macOS: plain `pip install torch`
pip install -e ".[dev]"
python -c "import torch, gymnasium, mujoco; print(torch.__version__, gymnasium.__version__, mujoco.__version__)"
```

The package is called `robust_rl` (not `ppo_scratch`), so it can live in the same environment as project 1.
No `make`? Every target in the Makefile is a plain `python -m robust_rl....` command shown in this guide.

---

## 2. What is new compared with project 1

```
src/robust_rl/
├── perturb.py         PerturbationSpec, DomainRandomization, PerturbedEnv wrapper         (NEW)
├── robust_study.py    trains baseline + domain-randomised variants x seeds (resumable)    (NEW)
├── robust_eval.py     evaluates every model across the perturbation grid                  (NEW)
├── robust_report.py   curves, summary tables, Welch tests                                 (NEW)
├── closed_loop.py     steps a perturbed simulator with actions fetched over HTTP          (NEW)
├── config.py          + dr_strength, dr_params, train_obs_noise, train_action_delay       (EXTENDED)
├── train.py           environments are wrapped with DomainRandomization when enabled      (EXTENDED)
└── (agent, gae, networks, normalize, evaluate, export, serve, analysis, ablate: as in project 1)
```

---

## 3. Concepts

### 3a. The sim-to-real gap and domain randomisation

A policy trained in one simulator setting overfits to its exact physics. Real robots differ: friction varies, payloads
change mass, motors weaken, sensors are noisy, control loops have latency. **Domain randomisation (DR)** trains on many
perturbed versions of the simulator so the policy cannot rely on exact values.

### 3b. The five perturbation axes

| Axis | Implementation (in `PerturbedEnv`) | Notes |
|---|---|---|
| `friction` | scales every geom's *sliding* friction (`geom_friction[:, 0]`) | only matters with ground contact (HalfCheetah, Hopper, Walker2d). **No effect on InvertedPendulum**, which never touches the ground |
| `mass` | scales `body_mass` **and** `body_inertia` together, then `mj_setConst` | scaling mass alone would make inertia physically inconsistent |
| `gain` | scales actuator gear ratios (`actuator_gear[:, 0]`) | models weaker/stronger motors |
| `obs_noise` | Gaussian sensor noise, std = level x per-dimension std of observations under a random policy | units are comparable across dimensions; level 0.2 = 20% of a typical signal std |
| `action_delay` | actions are applied `d` control steps late (a FIFO buffer, zeros at the start) | models communication/actuation latency |

Perturbations are applied to the model just before every reset, always relative to the *nominal* values captured at
construction, so they never accumulate (tested).

### 3c. How DR samples

Physical scales are **log-uniform in [1/(1+s), 1+s]**: strength `s = 0.5` means "x0.67 to x1.5", symmetric on a log scale
(a x2 change and a x0.5 change are equally far from nominal). Sensor-noise level is U(0, max); delay is uniform on {0..max}.

---

## 4. Experimental design

### Variants (policies that are compared)

| Variant | Training distribution |
|---|---|
| `baseline` | nominal physics only |
| `dr_0.2` | friction, mass, gain each ~ x0.83 .. x1.2 |
| `dr_0.5` | friction, mass, gain each ~ x0.67 .. x1.5 |
| `dr_0.5_sensing` | as `dr_0.5`, plus sensor noise up to 0.1 and delay up to 2 steps |

### Hypotheses (state them *before* you look at the data)

- **H1.** The baseline degrades as perturbations grow.
- **H2.** DR improves robustness on the axes it randomises, both inside and *beyond* the training range.
- **H3.** DR costs some nominal performance (the price of robustness).
- **H4.** Robustness to physics does not automatically transfer to unseen axes (noise, delay); only `dr_0.5_sensing`,
  which trains on them, should help there.

Any of these may turn out false. Reporting that is a result.

### Controls

Same training budget and seeds across variants; evaluation uses **identical episode seeds for every model and every
perturbation** and a **deterministic** policy; one axis is perturbed at a time; the grid extends beyond the widest
training range (e.g. friction x3 vs a training maximum of x1.5).

### Metrics (all computed per seed, then averaged across seeds)

| Metric | Definition | Caveat |
|---|---|---|
| `nominal` | deterministic return at nominal settings | |
| `mean_perturbed` | mean return over all perturbed grid points | depends on the grid you choose |
| `worst_case` | minimum return over the grid | noisy; driven by the most extreme point |
| `retention` | `mean_perturbed / nominal` | **only defined when nominal > 0** (otherwise NaN) |
| per-axis mean | mean over that axis' grid points | |

The unit of replication is the **seed** (one trained policy), not the episode.

---

## 5. Step-by-step

### Step 1. Tests (about 1 minute)

```bash
python -m pytest -q
```

They check, among other things: the nominal wrapper is *bit-identical* to the plain environment; each axis changes the
dynamics; scales are exact and do not accumulate; friction scales only the sliding component; delayed actions are the
older ones; sensor noise has the requested scale and is reproducible; DR ranges are right; the evaluation pipeline runs;
the report recovers a planted effect; and a served policy matches offline evaluation under perturbation.

### Step 2. Pipeline check on an easy task (a few minutes)

```bash
make robust-quick
```

This trains `baseline` and `dr_0.5` on `InvertedPendulum-v5` with 3 seeds, evaluates on a small grid, and writes the
report. It verifies the *machinery*, not the science: InvertedPendulum is too easy and has no ground contact, so the
friction axis is flat by construction.

### Step 3. Train the real study

Use **HalfCheetah-v5** as the primary environment: it never terminates early, so episode length does not confound
return. Hopper-v5 or Walker2d-v5 make good second environments, but they terminate when the robot falls, which mixes
"survived longer" into the return; say so if you use them.

```bash
python -m robust_rl.robust_study --study core --env-id HalfCheetah-v5 --seeds 5 \
    --total-timesteps 1000000 --jobs 4 --out runs/robust_hc
```

Budget (estimate): 4 variants x 5 seeds = 20 runs x ~20 min = ~7 h on one core, ~2 h with `--jobs 4`. The run is
resumable (finished runs are skipped). Preview with `--seeds 3 --total-timesteps 300000`.

### Step 4. Evaluate across the perturbation grid

```bash
python -m robust_rl.robust_eval --runs runs/robust_hc --out results/robust_hc/eval.csv --episodes 10 --jobs 4
```

20 models x 31 settings x 10 episodes x 1000 steps is ~6M environment steps: expect tens of minutes. Use `--quick`
for a tiny grid, or `--grid-json my_grid.json` for your own values.

### Step 5. Report

```bash
python -m robust_rl.robust_report --eval results/robust_hc/eval.csv --runs runs/robust_hc --out results/robust_hc
```

Outputs: `summary.md/.csv` (overall and per axis, with 95% CIs and Welch p-values vs baseline), `per_seed.csv`,
`robustness_<axis>.png` (return vs perturbation, mean and CI across seeds, gray band = widest training range) and
`robustness_grid.png` (all axes in one figure).

### Step 6. Interpret honestly

- **Look at the curves first**, tables second. The question is *where* robustness appears or fails.
- **Inside vs beyond the gray band.** Gains inside the band are expected; gains beyond it are the interesting finding.
- **Cost of robustness.** Compare `nominal` across variants. A DR policy that is worse at nominal but better under
  perturbation is the classic trade-off; quantify both.
- **Small samples.** With 5 seeds the CIs are wide (t = 2.78) and Welch tests are weak; with ~12 comparisons some will
  look "significant" by chance. Prefer effects that are large, consistent across seeds and visible in the curves.
- **Retention** is meaningless when nominal return is near zero or negative; look at raw returns then.
- **Do not claim sim-to-real transfer.** You measured robustness to *simulated* perturbations.
- Report failures and null results (e.g. "DR did not help on gain").

Use [`RESULTS_TEMPLATE.md`](RESULTS_TEMPLATE.md) for the write-up.

---

## 6. What was actually verified when this was built

(Also in the README.) 44 tests pass; the wrapper was checked on HalfCheetah and InvertedPendulum; a small end-to-end study
(InvertedPendulum, `baseline` vs `dr_0.5`, 3 seeds, 61k steps) ran through train -> evaluate -> report.

What that mini-study showed, and what it did not:

- Both variants reached the maximum return at nominal settings and stayed there across the whole friction, mass and gain grid:
  that task is insensitive to those ranges (and friction has no effect at all without ground contact).
- Sensor noise of 0.2-0.3 and any action delay of 1+ steps hurt both variants; with delay both fell to a return of ~10.
- Mean perturbed return was 796 (baseline) vs 795 (`dr_0.5`), Welch p = 0.94 with n = 3.
- **Interpretation:** the pipeline works, and an easy task cannot discriminate between variants. This is a pipeline check,
  not a finding about domain randomisation. It is also a good illustration of H4: randomising physics does nothing for
  delay, which `dr_0.5` never saw in training.
- Closed loop: a served policy and offline evaluation gave the *identical* mean return (943.8 over 5 episodes) under sensor
  noise 0.2; `/act` averaged ~3.6 ms (p95 ~5 ms) round trip on the build machine.

Not run: full HalfCheetah/Hopper studies, Docker, GitHub Actions, Windows/macOS. Treat those as yours to verify.

---

## 7. Deployment

Train, then export and serve exactly as in project 1, but deploy a **robust** policy and verify it in closed loop.

```bash
# export a self-contained TorchScript policy (normalisation + action clipping baked in)
python -m robust_rl.export --model runs/robust_hc/dr_0.5/seed_1/model.pt --out-dir export

# serve it
PPO_MODEL_PATH=runs/robust_hc/dr_0.5/seed_1/model.pt \
    python -m uvicorn --factory robust_rl.serve:create_app --port 8000

# in another terminal: step a perturbed simulator with actions fetched over HTTP
python -m robust_rl.closed_loop --url http://localhost:8000 --env-id HalfCheetah-v5 --axis gain --value 1.5 --episodes 3
```

The closed-loop mean return should match offline evaluation of the same model under the same perturbation (a test
asserts equality to 1e-3 on a small model). `closed_loop.py` also reports the mean and p95 latency of `/act`, which is
the number a robotics integration team would ask about.

Container and CI are the same as project 1 (`Dockerfile`, `.github/workflows/ci.yml`). **The Docker build and the
CI workflow have not been run in the build environment.**

---

## 8. Extending it (towards project 3)

- Add an off-policy learner (SAC/TD3) next to `agent.py` and run the *same* robustness evaluation: is PPO or SAC more
  robust, and at what sample cost? `robust_eval.py` only needs a checkpoint that `load_checkpoint` can read.
- Randomise more things (gravity, joint damping, contact softness) by extending `PerturbationSpec`.
- Try **adaptive** strategies: feed the perturbation parameters (or a history window) to the policy.
- Evaluate **combined** perturbations, not only one axis at a time.

---

## 9. Interview talking points

1. **What is domain randomisation, and why log-uniform?** Train on a distribution of simulators; log-uniform makes x2 and x0.5 equally likely.
2. **Why scale inertia together with mass?** Otherwise the body is physically inconsistent.
3. **Why does friction do nothing on InvertedPendulum?** No ground contact; I verified the flat curve.
4. **Why identical evaluation seeds?** Paired comparison removes initial-state variance from the difference between models.
5. **Why is the seed the unit of replication?** Training variance across seeds dominates episode variance.
6. **What does a wider DR range cost?** Often lower nominal return and slower learning; I measured it.
7. **Does physics DR transfer to sensor noise or delay?** I tested it with a held-out axis (`dr_0.5_sensing` vs `dr_0.5`).
8. **What can't you conclude?** Anything about real hardware, other environments, or other budgets.
9. **How did you validate the deployment?** A closed-loop test comparing served and offline returns, plus latency.

---

## 10. Troubleshooting

| Symptom | Likely cause / fix |
|---|---|
| Friction curve is flat | The environment has no ground contact (e.g. InvertedPendulum); use HalfCheetah/Hopper/Walker2d |
| `retention` is NaN | Nominal return <= 0 for that seed; use raw returns |
| Returns collapse for every variant at extreme values | Expected beyond the training range; that is the finding |
| Hopper/Walker2d returns look odd | Early termination shortens episodes; return mixes performance with survival |
| `RuntimeError: obs_scale is required` | Use `make_perturbed_env`, which estimates the scale for you |
| Evaluation is slow | Use `--jobs N`, `--episodes 5`, or `--quick` |
| Wide CIs | Add seeds instead of hiding the interval |

---

## 11. Limitations (state these)

- A perturbed simulator is a *proxy* for the real world; no hardware was used.
- One axis at a time; real gaps are combined and correlated.
- Perturbations are global scalings; real systems differ per body part and joint.
- Deterministic evaluation; stochastic evaluation could rank policies differently.
- One or two environments, default PPO hyper-parameters, small seed counts.
