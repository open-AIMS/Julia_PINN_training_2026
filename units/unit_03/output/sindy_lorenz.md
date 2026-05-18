::: {.cell-output .cell-output-stdout}
```text
Lorenz SINDy demo
samples used      : 9999
observation noise : 1.0 % of per-axis std
STLSQ threshold λ : 0.1

True system:
  dx/dt = -10·x + 10·y
  dy/dt =  28·x -  y - x·z
  dz/dt =  x·y - (8/3)·z

Recovered equations:
  dx/dt = -9.995·x  +9.995·y
  dy/dt = +27.971·x  -0.990·y  -0.999·x*z
  dz/dt = -2.667·z  +1.000·x*y

σ̂ ≈ 9.995   (true 10.000)
ρ̂ ≈ 27.971   (true 28.000)
β̂ ≈ 2.667   (true 2.667)
```
:::
