## Overall

| variant | n_seeds | nominal | nominal_ci95 | mean_perturbed | mean_perturbed_ci95 | worst_case | worst_case_ci95 | retention | retention_ci95 | nominal_vs_baseline | perturbed_vs_baseline | perturbed_welch_p |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| baseline | 2 | -70.10 | 710.42 | -45.22 | 371.02 | -98.92 | 532.43 |  |  |  |  |  |
| dr_0.5 | 2 | 62.84 | 1064.00 | 17.32 | 534.94 | -91.34 | 709.90 | 0.41 | 0.00 | 132.94 | 62.54 | 0.36 |

## Per perturbation axis

| variant | axis | mean_return | ci95 | delta_vs_baseline | welch_p |
|---|---|---|---|---|---|
| baseline | friction | -46.87 | 386.74 |  |  |
| baseline | mass | 8.39 | 119.96 |  |  |
| baseline | gain | -69.81 | 598.33 |  |  |
| baseline | obs_noise | -73.71 | 623.71 |  |  |
| baseline | action_delay | -71.49 | 614.26 |  |  |
| dr_0.5 | friction | 26.08 | 636.13 | 72.96 | 0.36 |
| dr_0.5 | mass | -58.73 | 478.40 | -67.11 | 0.31 |
| dr_0.5 | gain | 97.81 | 1675.24 | 167.62 | 0.41 |
| dr_0.5 | obs_noise | 32.57 | 635.30 | 106.28 | 0.27 |
| dr_0.5 | action_delay | -24.38 | 21.77 | 47.10 | 0.51 |
