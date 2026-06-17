::: {.cell-output .cell-output-stdout}
```text
================================================================
Unit 5 — 2-D diffusion PINN (hand-built), CPU vs GPU
================================================================
GPU available: true  (NVIDIA L4)
u_θ = sin(πx)sin(πy)(1 + t·N);  residual ∂tu − α(∂xxu+∂yyu) by input stencil

CPU  (N=4000)   … 93.4s | loss 9.98e-06 | L2 err vs exact 6.61e-05
GPU  (N=120000) … 105.9s | loss 7.11e-06 | L2 err vs exact 3.20e-05

GPU trains 30× the collocation points (and 1.5× the iterations) in 1.1× the
CPU wall-clock, and reaches the analytic solution to L2 = 3.2e-05.
```
:::
