::: {.cell-output .cell-output-stdout}
```text
================================================================
Unit 1 — Moreton Bay shallow-water surge on CPU vs GPU
================================================================
geometry: real Moreton Bay bathymetry (190x100)
GPU available: true  (NVIDIA A10G, 22.0 GiB)

refine   grid         cells     steps      CPU (s)    GPU (s)   speedup
------------------------------------------------------------------------
1        190x100      19000     1462          1.77       0.37       4.7x
2        380x200      76000     2925          6.86       0.63      10.9x
3        570x300      171000    4388         26.99       1.39      19.5x
4        760x400      304000    5851         69.04       2.90      23.8x

CPU and GPU fields agree to < 1e-2 m at every resolution (same code, same physics).
wrote figures/surge_gpu_field.png
```
:::
