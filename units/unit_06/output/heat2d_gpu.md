::: {.cell-output .cell-output-stdout}
```text
================================================================
Unit 6 — 2-D heat equation (FTCS finite differences) on CPU vs GPU
================================================================
GPU available: true  (NVIDIA L4)

N        grid         cells      steps      CPU (s)    GPU (s)   speedup
--------------------------------------------------------------------------
128      128x128      16384      2000          0.09       0.04       2.1x
256      256x256      65536      2000          0.37       0.04       9.1x
512      512x512      262144     2000          1.49       0.04      36.9x
1024     1024x1024    1048576    2000          6.07       0.10      63.4x
2048     2048x2048    4194304    2000     —  (skip)       0.28   GPU-only
4096     4096x4096    16777216   2000     —  (skip)       1.40   GPU-only

CPU and GPU fields agree to < 1e-3 at every size (same stencil, same code).
```
:::
