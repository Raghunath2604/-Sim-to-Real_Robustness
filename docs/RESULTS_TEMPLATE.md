# Results write-up (TEMPLATE: replace every `<...>` with your own measured numbers)

> Do not publish with unfilled placeholders. Do not copy numbers from anywhere else.

## 1. Question and hypotheses

Does domain randomisation (DR) make PPO policies more robust to physics and sensing perturbations, at what cost in
nominal performance, and does robustness transfer across perturbation types?

- H1 (baseline degrades): `<supported / not supported, evidence>`
- H2 (DR helps on randomised axes, inside and beyond training range): `<...>`
- H3 (cost in nominal performance): `<...>`
- H4 (no transfer to unseen axes without training on them): `<...>`

## 2. Setup

- Environment(s): `<HalfCheetah-v5 ...>`; gymnasium `<ver>`, mujoco `<ver>`; hardware `<...>`
- Variants: baseline, dr_0.2, dr_0.5, dr_0.5_sensing; `<5>` seeds each; `<1,000,000>` steps
- Evaluation: deterministic, `<10>` episodes per setting, identical episode seeds, one axis at a time
- Grid: `<paste from robust_eval.py or your grid.json>`
- Uncertainty: 95% t-interval across seeds; Welch tests vs baseline (interpret cautiously)

## 3. Nominal performance (cost of robustness)

`<table: nominal return mean +- CI per variant>`

## 4. Robustness curves

![grid](../results/<study>/robustness_grid.png)

`<Per axis: where each variant degrades, inside vs beyond the training range.>`

## 5. Summary tables

`<paste results/<study>/summary.md>`

## 6. Findings

1. `<finding, with numbers and the figure it comes from>`
2. `<...>`

## 7. Deployment check

`<Closed-loop return vs offline return for the deployed policy under <axis, value>; mean and p95 /act latency.>`

## 8. Limitations

`<Simulated proxy only; one axis at a time; global scalings; environments, seeds and budget; multiple comparisons.>`

## 9. Next steps

`<Combined perturbations, adaptive policies, SAC comparison (project 3), more environments.>`
