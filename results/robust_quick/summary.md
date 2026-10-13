## Overall

| variant | n_seeds | nominal | nominal_ci95 | mean_perturbed | mean_perturbed_ci95 | worst_case | worst_case_ci95 | retention | retention_ci95 | nominal_vs_baseline | perturbed_vs_baseline | perturbed_welch_p |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| baseline | 3 | 1000.00 | 0.00 | 835.99 | 65.88 | 9.00 | 0.00 | 0.84 | 0.07 |  |  |  |
| dr_0.5 | 3 | 1000.00 | 0.00 | 868.40 | 16.71 | 9.89 | 3.13 | 0.87 | 0.02 | 0.00 | 32.42 | 0.16 |

## Per perturbation axis

| variant | axis | mean_return | ci95 | delta_vs_baseline | welch_p |
|---|---|---|---|---|---|
| baseline | friction | 1000.00 | 0.00 |  |  |
| baseline | mass | 1000.00 | 0.00 |  |  |
| baseline | gain | 1000.00 | 0.00 |  |  |
| baseline | obs_noise | 678.89 | 527.04 |  |  |
| baseline | action_delay | 9.00 | 0.00 |  |  |
| dr_0.5 | friction | 1000.00 | 0.00 | 0.00 |  |
| dr_0.5 | mass | 1000.00 | 0.00 | 0.00 |  |
| dr_0.5 | gain | 1000.00 | 0.00 | 0.00 |  |
| dr_0.5 | obs_noise | 937.33 | 134.82 | 258.44 | 0.16 |
| dr_0.5 | action_delay | 9.89 | 3.13 | 0.89 | 0.35 |
