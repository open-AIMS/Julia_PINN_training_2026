::: {.cell-output .cell-output-stdout}
```text
==================================================================
Unit 7 — Poisson-PINN training-step throughput: CPU vs GPU
==================================================================
GPU available: true  (NVIDIA L4)
net 2-128-128-128-1; one step = 5 batched forward passes + reverse-mode AD

N (colloc)       CPU (ms)     GPU (ms)    speedup      CPU pts/s      GPU pts/s
------------------------------------------------------------------------------
1000                 21.4          5.1       4.2x       4.67e+04       1.96e+05
10000               328.3          5.5      59.9x       3.05e+04       1.82e+06
100000             2299.5         41.7      55.2x       4.35e+04       2.40e+06
500000             8566.5        367.2      23.3x       5.84e+04       1.36e+06

At N=500000 the GPU evaluates 39.2x more collocation points per second.
```
:::
