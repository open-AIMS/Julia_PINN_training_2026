# Course code review — runnability + clarity pass (2026-06-21)

Goal (per user): test/review **all** course code; confirm it runs in the `@pinn`
env on the **CPU hub**, **GPU hub**, and **Docker container**; make only minor
clarity improvements (ideally change nothing); verify Quarto→code references are
correct. "No-surprise" review — record anything found and fix what's off.

## Environments used

- **Docker** = running container `goofy_yonath`, image
  `ghcr.io/accumulationpoint/pinns-course-core` (built 2026-06-19), Julia 1.12.6,
  `@pinn` present, **CPU-only** (no CUDA/LuxCUDA/cuDNN). Repo not mounted →
  `docker cp` scripts in to run. This is the faithful CPU `@pinn`.
- **CPU hub / GPU hub** = AWS via SSM (GPU hub `i-0bae725d293a62ec5`, g5.2xlarge,
  ap-southeast-2). GPU-only scripts tested here.
- Local Mac Julia is 1.11.7 with no committed Manifest → **not** `@pinn`; not used
  as a correctness oracle.

## Method

- `@pinn` import smoke test per script (does every `using/import` load?) — catches
  env drift, the failure mode from the 2026-06-13 notes (#4b/#5.3).
- Run representative/cheap scripts end-to-end in Docker; GPU scripts on the GPU hub.
- Cross-reference audit per unit: every `{{< include >}}` resolves; every script
  path named in prose exists; line-number references accurate; "Supporting files"
  manifest complete (none missing / none extra, excluding `_`-prefixed); flag orphans.

## Legend

✅ runs/verified · 🟢 import-clean in @pinn · 🖥 GPU-only (hub) · 📝 clarity edit made ·
⚠ issue found · — not yet done

---

## Runnability results (the headline)

**Every Julia script's import set loads in `@pinn`** — verified by importing the
full union of packages used across all `units/*/scripts/*.jl` in the Docker
`@pinn`. The only three that don't load are the **documented student `Pkg.add`s**,
absent from `@pinn` by design:

| Package | Used by | Documented add? |
|---|---|---|
| `KolmogorovArnold` | unit_02 KAN (§2.8) | ✅ now (was contradicted — **fixed**, see F1) |
| `MLDatasets` | unit_02 MNIST scripts | ✅ unit_02.qmd:167–170 |
| `SciMLSensitivity` | unit_04 neural_ode_train.jl | ✅ unit_04.qmd:310 |

**Real executions (not just imports):**
- ✅ Docker `@pinn` (CPU): `sindy_lorenz.jl` runs, recovers Lorenz (σ̂ 9.995 / ρ̂ 27.971 / β̂ 2.667).
- ✅ Docker `@pinn` (CPU): `pinn_neuralpde_ode.jl` — full NeuralPDE symbolic PINN
  solve, final loss 1.15e-7, max|PINN−exact| 9.6e-5.
- ✅ GPU hub `@pinn` (A10G, via SSM): `CUDA.functional()=true`, `gpu_device()` +
  `Lux.Dense` forward pass on `CuArray` works. GPU stack (`CUDA`/`LuxCUDA`/`Lux`) loads clean.
- **CPU hub** not separately driven: the `@pinn` appendix guarantees the cloud hub
  and the Docker image are *bit-identical* (same Julia 1.12.6, same Manifest), so the
  Docker CPU results stand in for it. (Could SSM-verify on request.)

No script fails to run in `@pinn` for an undocumented reason. "No surprise" — confirmed.

## Fixes applied this pass (minimal, clearly-correct)

- **F1 — unit_02.qmd:1213** *(genuine bug; runnability)*: the §2.8 setup note claimed
  `KolmogorovArnold.jl` "ships in the course `@pinn` environment … no `Pkg.add`
  needed." It does **not** ship in `@pinn` (Docker import test = FAIL; not in the
  appendix's 28; centralization was held, task #65), and the unit itself says the
  opposite at line 1311. A student would hit a load error. Rewrote the note to instruct
  a one-off `Pkg.add("KolmogorovArnold")`, matching the MLDatasets note and line 1311.
- **F2 — unit_04.qmd:310**: "needs `SciMLSensitivity` + `ComponentArrays` added" —
  `ComponentArrays` is already in `@pinn` (import test = OK; in Project.toml). Reworded
  to only `SciMLSensitivity`.
- **F3 — unit_05 scripts ×3 stale section refs**: the hand-rolled ODE PINN is §5.3, not
  §5.2. Fixed `pinn_neuralpde_ode.jl:3`, `pinn_poisson_disk.jl:10`,
  `pinn_2d_diffusion_gpu.jl:15` (the last said "§5.2/§5.3"; §5.2 advocates AD, only §5.3
  uses the FD-in-input stencil).
- **F4 — pinn_2d_diffusion_gpu.jl:27**: header said the qmd shows it `eval: false`; it's
  actually a `{{< include >}}` listing (never executed). Reworded.
- **F5 — unit_09.qmd:450**: §9.6 table `T_f` row had value "30 days" in a row whose
  Units column is "s". Changed value to `2.592×10⁶ (=30 d)`, units "s".

> ⚠ Re-render note: F3/F4 edit scripts that are `{{< include >}}`'d into unit_05, and
> F1/F2/F5 edit qmd prose. `freeze: auto` re-renders the prose units automatically on
> next `make`; unit_05's included-listing freeze must be recomputed (a unit re-render)
> before deploy. Nothing committed/pushed — left for your call.

## Section-exercise review (59 exercises + 59 solutions, 2026-06-21)

Goal: exercises agree with the recently-changed sections in style; solutions run and
make sense; **more generous hints** (students are time-short); no new content — only
simplify/refine/scaffold. Per-unit analysis agents (10) read every exercise against its
section + solution.

- **Pairing:** exactly 59 `#ex-N-M` ↔ 59 `#sol-N-M`, 1:1, none missing/extra.
- **Solutions run:** all of `exercise_solutions.qmd` re-executed cleanly in the full
  render — every solution cell runs. No solution found that mis-answers its prompt or
  contradicts its section.
- **Hints made more generous (44 total):** u1 ×3, u2 ×7, u3 ×5, u4 ×3, u5 ×5, u6 ×7,
  u7 ×4, u8 ×2, u9 ×6, u10 ×2. Each now names the key formula/function, gives a concrete
  first step, states the expected number/behaviour, and points at the exact section
  example — drawn only from existing section+solution text. ~15 hints already generous
  (conceptual/answer-key style) were left as-is (e.g. ex-3-2, 5-6, 6-7, 7-4/6/7/8, several u8).
- **Staleness/correctness fixes found by the agents and applied:**
  - **Regression I introduced:** the D4 `\rho`→`\rho_0` rename had clobbered `\rho_a`
    (air density) into malformed `\rho_0_a` at unit_08:392–393 — **fixed** to `\rho_a`.
  - ex-2-4 hint pointed at "§5.2 … a unit early" for the nested-`ForwardDiff` pattern →
    fixed to **§2.6** (the autodiff section after the reorg).
  - KolmogorovArnold "ships in @pinn / no install" survived in TWO more spots beyond F1 —
    the pasted listing (unit_02:1185) and the real script `kan_julia.jl:6–7` — both
    **fixed** to "needs `Pkg.add`".
  - sol-1-2 still said "slider"; §1.2 is now "movie" → **fixed** (heading + body).
  - ex-7-2 hint said "nested ForwardDiff" but the SWE residual is first-order (two single
    derivatives) → **fixed** wording.
- **Flagged, not changed** (out of exercise scope or pedagogically intentional):
  - §9.9.1/§9.10.1 "720 samples = 3 600 points" is really 721 rows → 3 605 (off-by-one in
    spec prose, not in any exercise/solution; "720 = 30×24" reads cleaner — left for your call).
  - unit_08 §8.4 keeps k₁=0.5, k₂=1/14 deliberately (load-bearing for the ex-8-4
    two-band-vs-single-band coincidence) — confirmed intact; canonical values noted (D3).
  - sol-8-5 uses `sech(...)`; fine if `SpecialFunctions` is loaded (it executed in the render).

## D1–D4 physics-prose fixes (applied on request, 2026-06-21)

- **D1 — unit_04 §4.1 Lotka–Volterra "limit cycle" → neutral centres.** Reworded the
  limit-cycle bullet (qmd:44) to name Van der Pol as the example and explicitly note LV's
  orbits are *neutral centres* (a continuous family, none attracting); and the worked-
  example line (qmd:62) "structurally robust limit cycle" → "a family of closed orbits —
  neutrally stable centres, not an attracting limit cycle". Now consistent with the unit's
  own first-integral discussion at :88.
- **D2 — unit_08 §8.6 false site attribution → generic reference column.** The numbers
  ($\mathrm{Pe}\approx1$, $T_\kappa\approx116$ d) use $H=100$ m with the §9.6 reference
  $\kappa_m=10^{-3}$, $w_0=10^{-5}$ — *not* Davies (60 m, the named site) and *not*
  Myrmidon (100 m but $\mathrm{Pe}\gg1$, not balanced). Renaming to Myrmidon would have
  introduced a new contradiction (its regime is advective). Fixed by dropping the site
  name: "For a representative GBR shelf column with $H=100$ m …".
- **D3 — two-band constants reconciled by note (numbers preserved).** ⚠ Unit 8's
  $k_2=1/14$ is **load-bearing**: the §8.4 exercise + sol-8-4 are built around a designed
  coincidence where the two-band model and the capstone's single-band $\zeta=10$ m agree
  below 30 m (4.9% vs 5.0%), and the red band "dumps a quarter into the top metre" needs
  $k_1=0.5$. Swapping in the canonical 0.35 m/23 m would break both (→ 11.4% vs 5.0%, and
  ~56% in the top metre). So Unit 8's values are deliberately rounded/tuned. Reconciled the
  *narrative* instead: added a clause at unit_08 §8.4 noting these are rounded illustrative
  values and the canonical Jerlov Type-I constants ($1/k_1\approx0.35$ m, $1/k_2\approx23$ m)
  are what Unit 9 §9.5 uses. Exercise + solution untouched (coincidence intact).
- **D4 — ρ vs ρ₀ standardised to ρ₀ course-wide.** ρ₀ was the established majority
  (unit_09, unit_10, physical_review all use it; only unit_08 + part of exercise_solutions
  used plain ρ) and is the conventional reference-density symbol. Changed unit_08 (9
  occurrences) and the 6 `\rho c_p` in exercise_solutions to `\rho_0`. Verified: no double
  subscripts, no plain ρ left in any column-model unit.

## Other minor warts recorded (cosmetic; left as-is)

- unit_01 `generate_site_map.py:138` hard-codes "1 km grid" in the figure label; grid is
  500 m. Source string is stale **and** baked into the committed `site_map.png`
  (regenerating needs network tiles) — fix source + regen figure when convenient.
- unit_01 `generate_surge_frames.jl:11` docstring says "frame_072"; writes 121 frames.
- unit_01 cadence wording: `train_inverse_pinn.jl:85` prints "every 6 min"; actual thinned
  stride is 3 min (DATA_STRIDE=15×12 s), raw CSV is 12 s; qmd:550 says "one-minute". Three
  mismatched cadence statements — reconcile (touches user-facing text).
- unit_01 dead `FIG_DIR` in `generate_surge_data.jl`/`train_inverse_pinn.jl`; unused
  `Random` import in `train_inverse_adjoint.jl`; redundant `round.(Int, …Int)` /
  `skipmissing(filter(!isnan,…))` in `build_bay.jl`.
- unit_03 `pod_chain.jl` is a sidecar duplicate of the inline §3.5 `{julia}` block; its
  `pod_spectrum.png` / `output/pod_chain.md` are produced but never surfaced. Intentional
  mirror — annotate or drop the unused `savefig`.
- unit_04 `neural_ode_train.jl` writes `figures/neural_ode_fit.png` that doesn't exist and
  is never displayed; `epoch` loop var is really a full-batch step.
- unit_05 `pinn_poisson_disk.jl` is an **orphan**: built by `build.sh execute 5` (writes
  `output/pinn_poisson_disk.md`) but never `{{< include >}}`'d nor named in unit_05.qmd —
  the forward/inverse/data-only Poisson demo is invisible in the rendered unit. Either
  surface it or stop building it. Dead imports: `Random` (ode.jl:11), `LinearAlgebra`
  (heat.jl:17).
- unit_07 caption (qmd:951) splices two rows of the A10G table: "~3.5×10⁶ … (~86×) at
  N=10⁵". 3.5e6 is the N=5×10⁵ throughput; 86× is the N=10⁵ latency speedup; no single N
  has both. Pick one N for the caption. (Numbers themselves are real and self-consistent
  in `output/pinn_throughput_gpu.md`.)
- unit_07 `generate_laplace_disk_figure.py` saves `.png` but the unit embeds
  `laplace_disk.svg` (no in-repo generator of the displayed SVG); `naive_swe_failure.svg`
  also has no generator script. Orphan figures (exist on disk; not render-breaking).
- unit_09 `generate_site_map.jl` (CairoMakie schematic) is legacy — superseded by the
  `.py` (real OSM tiles), which is what `site_map.png` matches. Neither is flagged as
  canonical. Prose "720 samples = 3 600 points" is 721 rows on disk → 3 605 (off-by-one).
- unit_10 §10.3 Task B measured numbers (CPU 6.8%, A10G 6.7% + partitions) are
  author-transcribed from GPU runs; the scripts *do* compute them, but unlike
  `column_fd.md` there's no committed stdout artifact pinning the exact digits. Traceability
  only. `column_pinn_gpu.jl`/`task_a_inverse_pinn.jl` default figure dir to `../figures`
  (others use `../output`) — harmless, unsurfaced.

---

## Per-unit status

- **Unit 01** — 13 scripts (11 manifest + 2 `_`-prefixed). All CPU except `surge_gpu.jl`
  (GPU-aware, CPU fallback). Includes resolve; manifest exact; no orphans. Warts: cadence
  wording, stale frame count, "1 km grid" label, dead `FIG_DIR`. 🟢 imports clean.
- **Unit 02** — 13 scripts (8 jl / 5 py). GPU-only: `conv_mnist_lux.jl`, `mnist_gpu_lux.jl`.
  **F1 fixed** (KolmogorovArnold note). Manifest exact. Double-`./255` on already-[0,1]
  MLDatasets in `mnist_linear_lux.jl:16` & `mnist_rf_julia.jl:14` (harmless; mirrored in
  pasted listings) — recorded. 🟢 imports clean (MLDatasets/KAN = documented add).
- **Unit 03** — 7 scripts, all CPU. Includes resolve; manifest exact. ✅ `sindy_lorenz.jl`
  executed in Docker. `pod_chain.jl` sidecar-dup recorded.
- **Unit 04** — 3 scripts, all CPU. **F2 fixed.** D1 (limit cycle) recorded. No GPU.
  `neural_ode_fit.png` orphan recorded.
- **Unit 05** — 4 scripts; GPU-only: `pinn_2d_diffusion_gpu.jl`. **F3/F4 fixed.** ✅
  `pinn_neuralpde_ode.jl` executed in Docker (loss 1.15e-7). §5.3 central-difference
  prose/code consistent. `pinn_poisson_disk.jl` orphan recorded.
- **Unit 06** — 1 script `heat2d_gpu.jl` (GPU-aware, graceful CPU fallback via
  `CUDA.functional()`). Includes + output + figure all resolve. Clean — no issues.
- **Unit 07** — 2 scripts; GPU-only: `pinn_throughput_gpu.jl`. Caption row-splice + orphan
  SVGs recorded. Manifest exact.
- **Unit 08** — science chapter, no scripts. `column_diagram.svg` well-formed (xmllint OK),
  red-flux/axis labels correctly carry `stroke="none"` (the legibility fix held). D2/D3/D4
  recorded. Correctly absent from Supporting-files manifest.
- **Unit 09** — 2 figure-gen scripts (jl legacy, py canonical). **F5 fixed** (T_f units).
  Coordinate style + A10G runtime tables already corrected. Site-table values consistent
  with unit_10 scripts. Legacy-`.jl` + off-by-one recorded.
- **Unit 10** — 8 scripts. GPU-only: `column_pinn_gpu.jl`, `task_a_inverse_pinn.jl`,
  `task_b_joint_inverse.jl`. CPU: `task_b_subscale_prototype.jl`, `column_fd.jl`,
  `generate_mooring_csvs.jl`, `task_a_forward_deepxde.py`. Static λ recipe confirmed (no
  adaptive runaway). scenario_4 Tf=30 d + plot_scn2 panel confirmed. 721-row CSVs confirmed.
  Manifest exact; no dangling refs to deleted scripts. §10.3 provenance recorded.
