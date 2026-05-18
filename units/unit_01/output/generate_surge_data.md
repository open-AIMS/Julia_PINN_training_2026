::: {.cell-output .cell-output-stdout}
```text
CFL number = 0.731 (must be < 1)
River-mouth source @ cell (29, 105)  H=8.9 m
Gauge G1 @ cell (29, 108)  H=6.1 m
Gauge G2 @ cell (43, 83)  H=6.0 m
Gauge G3 @ cell (60, 144)  H=18.2 m
Gauge G4 @ cell (54, 43)  H=6.0 m

Integrating 1800 steps × 12.0s = 6.0 h …
FD solve done in 11.48 s

Wrote:
  data/gauges_clean.csv     (1801 rows × 4 gauges)
  data/gauges_observed.csv  (σ = 0.015 m)
  data/psi_truth.csv
  data/snapshots.bin        (190 × 100 × 73 Float32)
  data/snapshots_meta.json

Max |η| at any gauge:
  G1 : clean=0.3585 m   noisy=0.3886 m
  G2 : clean=0.2555 m   noisy=0.2940 m
  G3 : clean=0.2478 m   noisy=0.2770 m
  G4 : clean=0.2212 m   noisy=0.2469 m
Max |ψ_truth|: 0.4500 m
```
:::
