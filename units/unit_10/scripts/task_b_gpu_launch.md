# Task B — GPU full-scale launch checklist

The CPU sub-scale prototype (`task_b_subscale_prototype.jl`, two sites, static
weights) proves the recipe and the headline finding: a single shared storm
envelope $\tau(t)$ recovered jointly from two moorings beats either mooring's
decoupled inverse. This checklist is the delta from that prototype to the full
three-site GPU run (`task_b_joint_inverse.jl`).

## What changes from the CPU prototype

| Knob | CPU prototype | GPU full-scale | Why |
|---|---|---|---|
| Sites | A + B (2) | A + B + C (3) | Myrmidon (Pe ≫ 1) adds the advection-dominated regime |
| Column depth | A 15 m / B 60 m | all three at $H = 100$ m | the spec's deep column; wider scale separation |
| Collocation $N_r$ / site | 4 000 | 50 000 | resolves the diurnal + storm scales at depth |
| T-network | 32 × 4, plain $(\zeta,\tau)$ input | 128 × 6, Fourier-feature input | spectral bias at $H=100$ m needs the embedding |
| Loss weights | **static** $\lambda_d{=}6000,\lambda_b{=}10$ | gradient-balanced **adaptive** | three sites' residual/data scales span $\sim10^2$ |
| Causal weighting | off | on (per-site $\epsilon$) | Myrmidon's long characteristic time needs it |
| Adam steps | 12 000 | ~40 000 (then optional L-BFGS polish) | larger net + more collocation |
| Device | `identity` (CPU) | `gpu_device()` (CUDA) | ~50× throughput at this size |

The physics, the hard-IC/deep-BC ansatz $\theta=\zeta\,\tau\,N$, and the
FD-in-input derivative stencil are **unchanged** — only scale and the
spectral-bias/weight-balancing machinery are added.

## Pre-flight

1. **Environment.** Launch on the course GPU hub; `@pinn` already carries
   `Lux`, `LuxCUDA`, `CUDA`, `Optimisers`, `Zygote`, `CairoMakie`.
   ```
   julia --project=@pinn units/unit_10/scripts/task_b_joint_inverse.jl
   ```
2. **Confirm CUDA is live** — the script prints `CUDA.functional()` and the
   device name at startup; if it reports CPU, stop (a CPU full-scale run is
   hours, not minutes).
3. **Memory.** Three 128 × 6 networks + 3 × 50 000 collocation points fit
   comfortably in the A10G's 22 GiB; if you widen further, watch
   `CUDA.memory_status()`.

## Watch during the run

- **Per-site data losses** should fall together — if one site's $\mathcal{L}_d$
  stalls 2+ orders above the others, the adaptive weights have not yet
  balanced; let it run, don't hand-pin.
- **Recovered $\hat\tau(t)$ peak** should climb past ~80% of truth and the
  storm-day timing lock in within the first few thousand steps; the long tail
  is sharpening the rising/falling edges.
- **Causal weights** at Myrmidon: if the residual refuses to fall, the per-site
  $\epsilon$ is too aggressive — relax it (see the open question in §10.3).

## Outputs

- `output/task_b_joint_recovery.png` — recovered $\hat\tau(t)$ vs truth, plus
  the three sites' sensor fits.
- stdout — per-site peak-amplitude error, timing error, and forward $L^2$ vs
  the FD reference, in the same format as the CPU prototype.

## Expected runtime (A10G)

Order-of-minutes for the forward sanity solve, tens-of-minutes for the full
joint inverse — compared with hours on a CPU. The exact numbers the run
prints are what §10.3 Step 6 reports; treat the table there as measured, not
extrapolated, once this has run.
