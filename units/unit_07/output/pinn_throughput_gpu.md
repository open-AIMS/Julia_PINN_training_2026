::: {.cell-output .cell-output-stdout}
```text
==================================================================
Unit 7 — Poisson-PINN training-step throughput: CPU vs GPU
==================================================================
GPU available: true  (NVIDIA A10G)
net 2-128-128-128-1; one step = 5 batched forward passes + reverse-mode AD

N (colloc)       CPU (ms)     GPU (ms)    speedup      CPU pts/s      GPU pts/s
------------------------------------------------------------------------------
1000                 31.2          6.3       5.0x       3.20e+04       1.59e+05
10000               412.1          6.9      59.3x       2.43e+04       1.44e+06
100000             2563.1         29.7      86.2x       3.90e+04       3.36e+06
500000             9826.9        141.6      69.4x       5.09e+04       3.53e+06

At N=500000 the GPU evaluates 69.4x more collocation points per second;
the GPU throughput peaks at N=500000 (69.4x the CPU there).
```
:::
