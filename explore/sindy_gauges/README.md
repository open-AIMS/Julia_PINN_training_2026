# SINDy on the four Unit-1 tide gauges (exploratory)

> **Side branch only.** This folder lives on the `sindy-explore` branch and is
> **not** part of the rendered course site. It is a candidate for a future
> Unit-3 (SINDy) example, kept here so the work isn't lost.

## What it is

Fit a sparse dynamical model to the four Moreton-Bay tide gauges of Unit 1.
The bay is a *linearised shallow-water system* **forced** by the river surge
`ψ(t)`, so the honest model class is a forced linear system

```
ẋ = A x + b·ψ(t),     x = (η_G1, η_G2, η_G3, η_G4)
```

Two fits are compared: **(A)** autonomous SINDy `ẋ = Θ(x)Ξ` (no forcing) and
**(B)** SINDy-with-control, which appends `ψ(t)` to the library.

## Key findings

- **Autonomous SINDy fails** — with no forcing term, a model started from rest
  stays at rest (90–100% reconstruction error). Forced systems need a control input.
- **Controlled SINDy** recovers a *stable, oscillatory* model whose eigenvalues are
  the bay's damped resonant modes (**≈ 5.4 h and ≈ 19.3 h**), with the forcing
  vector `b` localised to the river-side gauges G1/G2 — physically sensible.
- **Closure caveat** — four point-gauges are a *partial observation of a PDE field*,
  not a closed system, so no four-state ODE is exact (it misses the secondary peaks /
  late ringing). The natural next step is a **time-delay embedding** (HAVOK).

## Files

| file | what |
|------|------|
| `sindy_gauges.jl`  | the full exploration (reads `units/unit_01/data`, writes the figure) |
| `sindy_gauges.png` | the 2×2 per-gauge result plot |
| `sindy_gauges.tex` / `.pdf` | a 3-page writeup (key idea, discovered system, plots) |

## Run

From the repo root, with the course Julia env (all deps already in `Project.toml`):

```
julia --project=. explore/sindy_gauges/sindy_gauges.jl
```

Rebuild the PDF:

```
cd explore/sindy_gauges && pdflatex sindy_gauges.tex
```
