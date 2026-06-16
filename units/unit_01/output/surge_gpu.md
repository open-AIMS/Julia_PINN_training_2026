::: {.cell-output .cell-output-stdout}
```text
================================================================
Unit 1 — Moreton Bay shallow-water surge on CPU vs GPU
================================================================
geometry: real Moreton Bay bathymetry (190x100)
GPU available: true  (NVIDIA L4, 22.0 GiB)

refine   grid         cells     steps      CPU (s)    GPU (s)   speedup
------------------------------------------------------------------------
1        190x100      19000     1462          0.55       0.25       2.2x
2        380x200      76000     2925          3.60       0.54       6.7x
3        570x300      171000    4388         12.21       0.79      15.5x
4        760x400      304000    5851         36.81       1.65      22.4x

CPU and GPU fields agree to < 1e-2 m at every resolution (same code, same physics).
wrote figures/surge_gpu_field.png
```
:::
