## Overall

| variant | n_seeds | nominal | nominal_ci95 | mean_perturbed | mean_perturbed_ci95 | worst_case | worst_case_ci95 | retention | retention_ci95 | nominal_vs_baseline | perturbed_vs_baseline | perturbed_welch_p |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| baseline | 5 | 2407.92 | 1725.80 | 1177.06 | 462.26 | -331.99 | 98.90 | 0.53 | 0.13 |  |  |  |
| dr_0.2 | 5 | 2201.58 | 1562.02 | 1312.62 | 864.78 | -291.97 | 226.98 | 0.60 | 0.05 | -206.34 | 135.56 | 0.71 |
| dr_0.5 | 5 | 1889.23 | 1287.78 | 1295.92 | 785.79 | -252.68 | 134.41 | 0.70 | 0.05 | -518.69 | 118.85 | 0.73 |
| dr_0.5_sensing | 5 | 1918.71 | 1682.59 | 1311.14 | 969.66 | -101.09 | 259.05 | 0.73 | 0.11 | -489.21 | 134.08 | 0.74 |

## Per perturbation axis

| variant | axis | mean_return | ci95 | delta_vs_baseline | welch_p |
|---|---|---|---|---|---|
| baseline | friction | 2245.18 | 1357.41 |  |  |
| baseline | mass | 599.71 | 290.11 |  |  |
| baseline | gain | 756.44 | 436.78 |  |  |
| baseline | obs_noise | 1989.09 | 928.36 |  |  |
| baseline | action_delay | 10.32 | 194.31 |  |  |
| dr_0.2 | friction | 2215.64 | 1526.40 | -29.55 | 0.97 |
| dr_0.2 | mass | 984.24 | 904.04 | 384.52 | 0.31 |
| dr_0.2 | gain | 1008.89 | 322.88 | 252.45 | 0.24 |
| dr_0.2 | obs_noise | 2000.97 | 1252.32 | 11.88 | 0.98 |
| dr_0.2 | action_delay | 59.92 | 234.14 | 49.60 | 0.66 |
| dr_0.5 | friction | 1934.82 | 1177.84 | -310.36 | 0.64 |
| dr_0.5 | mass | 1089.03 | 735.27 | 489.31 | 0.14 |
| dr_0.5 | gain | 1227.99 | 607.54 | 471.56 | 0.12 |
| dr_0.5 | obs_noise | 1806.37 | 1167.02 | -182.72 | 0.74 |
| dr_0.5 | action_delay | 170.62 | 187.88 | 160.30 | 0.14 |
| dr_0.5_sensing | friction | 1707.58 | 1151.47 | -537.61 | 0.43 |
| dr_0.5_sensing | mass | 1037.09 | 893.38 | 437.37 | 0.25 |
| dr_0.5_sensing | gain | 932.50 | 510.52 | 176.06 | 0.49 |
| dr_0.5_sensing | obs_noise | 1853.76 | 1579.85 | -135.33 | 0.84 |
| dr_0.5_sensing | action_delay | 989.76 | 849.25 | 979.45 | 0.03 |
