# Task B — GPU 3-site run, and the optional scale-up

The CPU sub-scale prototype (`task_b_subscale_prototype.jl`, two sites, static
weights) proves the recipe and the headline finding: a single shared storm
envelope τ(t) recovered jointly from two moorings beats either mooring's
decoupled inverse.

The shipped **three-site GPU run** (`task_b_joint_inverse.jl`) simply takes that
**same static recipe** to all three sites at a slightly wider network — and it
already clears the bar (6.7% joint peak error; see §10.3 Step 6). It needs **no**
modern-PINN machinery:

- static loss weights λ_d = 6000, λ_b = 10, λ_reg = 1e-5 (same as Task A),
- plain (ζ, τ) input (no Fourier features),
- 64 × 5 T-network per site, 48 × 3 shared τ-network,
- N_r = 4000 collocation points / site, 3000 Adam steps,
- the hard-IC / deep-BC ansatz T̃ = ζ·τ·N and the FD-in-input derivative stencil.

## Running it

1. **Environment.** Launch on the course GPU hub; `@pinn` already carries
   `Lux`, `LuxCUDA`, `CUDA`, `Optimisers`, `Zygote`, `CairoMakie`.
   ```
   julia --project=@pinn units/unit_10/scripts/task_b_joint_inverse.jl
   ```
2. **Confirm CUDA is live** — the script prints `CUDA.functional()` and the
   device name at startup; if it reports CPU, stop (a CPU 3-site run is hours,
   not minutes).
3. **Outputs.** `output/task_b_joint_recovery.png` (recovered τ̂(t) vs truth +
   the three sites' sensor fits) and per-site peak-amplitude error, timing
   error, and forward L² to stdout — the numbers §10.3 Step 6 reports.

## Optional scale-up (only if you push to harder regimes)

The shipped static run above clears the bar, so the modern-PINN toolkit is **not
required**. It becomes worth adding only if you deepen all three columns to the
spec's H = 100 m and want lower error. This table is the delta from the shipped
static run to that heavier config:

| Knob | shipped 3-site run | optional scale-up | Why |
|---|---|---|---|
| Collocation N_r / site | 4 000 | up to 50 000 | resolve diurnal + storm scales at depth |
| T-network | 64 × 5, plain (ζ,τ) input | 128 × 6, Fourier-feature input | spectral bias at H = 100 m |
| Loss weights | static λ_d = 6000, λ_b = 10 | gradient-balanced adaptive | three sites' residual/data scales span ~10² |
| Causal weighting | off | on (per-site ε) | Myrmidon's long characteristic time |
| Adam steps | 3 000 | ~40 000 (then optional L-BFGS polish) | larger net + more collocation |

The physics, the hard-IC/deep-BC ansatz T̃ = ζ·τ·N, and the FD-in-input
derivative stencil are **unchanged** — only scale and the spectral-bias /
weight-balancing machinery are added.

## Watch during a scaled-up run

- **Per-site data losses** should fall together — if one site's L_d stalls 2+
  orders above the others, the adaptive weights have not yet balanced; let it
  run, don't hand-pin.
- **Recovered τ̂(t) peak** should climb past ~80% of truth and the storm-day
  timing lock in within the first few thousand steps; the long tail sharpens the
  rising/falling edges.
- **Causal weights** at Myrmidon: if the residual refuses to fall, the per-site
  ε is too aggressive — relax it (see the open question in §10.3).

## Expected runtime (A10G)

The shipped 3-site static run is ~5 min (joint) on the A10G — the numbers
§10.3 Step 6 reports, measured not extrapolated. A scaled-up 128 × 6 /
50 000-collocation / 40k-step run is order-of-an-hour on the same GPU.
