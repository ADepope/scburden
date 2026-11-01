
### S1_main (5 replicates)

| method | R2_B | gainx1e4_B | r2_true | prog_cos | link_auc | celltype_cor | eta |
|---|---|---|---|---|---|---|---|
| coupled_refit | 0.388 ± 0.150 | 5.112 ± 2.266 | 0.385 ± 0.119 | 0.630 ± 0.042 | 0.817 ± 0.074 | 0.805 ± 0.071 | 8.600 ± 1.400 |
| coupled_joint | 0.296 ± 0.136 | 4.057 ± 2.171 | 0.282 ± 0.110 | 0.608 ± 0.057 | 0.916 ± 0.040 | 0.754 ± 0.050 | 3.000 ± 0.000 |
| two_step | 0.278 ± 0.146 | 3.865 ± 2.330 | 0.260 ± 0.105 | 0.416 ± 0.091 | 0.676 ± 0.106 | 0.645 ± 0.046 | – |
| scitd | 0.131 ± 0.090 | 1.969 ± 1.434 | 0.093 ± 0.059 | 0.261 ± 0.022 | – | – | – |
| random | -0.016 ± 0.001 | -0.195 ± 0.020 | -0.017 ± 0.002 | – | – | – | – |
| null | 0.000 ± 0.000 | 0.000 ± 0.000 | 0.000 ± 0.000 | – | – | – | – |

paired coupled_refit − two_step, R2_B: 0.110 ± 0.048 (better in 5/5)

paired coupled_refit − two_step, gainx1e4_B: 1.247 ± 0.439 (better in 5/5)

paired coupled_refit − two_step, prog_cos: 0.214 ± 0.053 (better in 5/5)

### S2_aligned (5 replicates)

| method | R2_B | gainx1e4_B | r2_true | prog_cos | link_auc | celltype_cor | eta |
|---|---|---|---|---|---|---|---|
| coupled_refit | 0.551 ± 0.112 | 9.394 ± 3.319 | 0.492 ± 0.104 | 0.619 ± 0.038 | 0.714 ± 0.083 | 0.823 ± 0.060 | 5.000 ± 2.074 |
| coupled_joint | 0.471 ± 0.103 | 8.324 ± 3.463 | 0.481 ± 0.093 | 0.702 ± 0.012 | 0.811 ± 0.038 | 0.878 ± 0.032 | 3.000 ± 0.000 |
| two_step | 0.465 ± 0.100 | 8.287 ± 3.211 | 0.425 ± 0.105 | 0.503 ± 0.088 | 0.654 ± 0.097 | 0.731 ± 0.102 | – |
| scitd | 0.180 ± 0.080 | 3.501 ± 2.201 | 0.167 ± 0.073 | 0.385 ± 0.038 | – | – | – |
| random | -0.013 ± 0.003 | -0.184 ± 0.022 | -0.012 ± 0.002 | – | – | – | – |
| null | 0.000 ± 0.000 | 0.000 ± 0.000 | 0.000 ± 0.000 | – | – | – | – |

paired coupled_refit − two_step, R2_B: 0.086 ± 0.068 (better in 4/5)

paired coupled_refit − two_step, gainx1e4_B: 1.107 ± 0.819 (better in 4/5)

paired coupled_refit − two_step, prog_cos: 0.116 ± 0.097 (better in 3/5)

### S3_null (5 replicates)

| method | R2_B | gainx1e4_B | prog_cos | celltype_cor | eta |
|---|---|---|---|---|---|
| coupled_refit | 0.027 ± 0.164 | -0.081 ± 0.023 | 0.532 ± 0.055 | 0.773 ± 0.044 | 5.400 ± 1.913 |
| coupled_joint | 0.345 ± 0.773 | -0.291 ± 0.046 | 0.596 ± 0.075 | 0.799 ± 0.063 | 1.000 ± 0.000 |
| two_step | 0.147 ± 0.337 | -0.128 ± 0.036 | 0.600 ± 0.080 | 0.800 ± 0.063 | – |
| scitd | 0.150 ± 0.480 | -0.181 ± 0.034 | 0.291 ± 0.013 | – | – |
| random | 0.053 ± 0.244 | -0.138 ± 0.025 | – | – | – |
| null | 0.000 ± 0.000 | 0.000 ± 0.000 | – | – | – |

paired coupled_refit − two_step, R2_B: -0.120 ± 0.175 (better in 3/5)

paired coupled_refit − two_step, gainx1e4_B: 0.046 ± 0.020 (better in 4/5)

paired coupled_refit − two_step, prog_cos: -0.068 ± 0.098 (better in 1/5)
