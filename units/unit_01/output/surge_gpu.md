::: {.cell-output .cell-output-stdout}
```text
================================================================
Unit 1 — Moreton Bay shallow-water surge on CPU vs GPU
================================================================
geometry: real Moreton Bay bathymetry (190x100)
GPU available: true  (NVIDIA A10G, 22.0 GiB)

refine   grid         cells     steps      CPU (s)    GPU (s)   speedup
------------------------------------------------------------------------
1        190x100      19000     1462          1.06       0.31       3.5x
2        380x200      76000     2925          7.71       0.64      12.0x
3        570x300      171000    4388         26.31       1.36      19.3x
4        760x400      304000    5851         56.48       2.84      19.9x

CPU and GPU fields agree to < 1e-2 m at every resolution (same code, same physics).
wrote figures/surge_gpu_field.png
```
:::
