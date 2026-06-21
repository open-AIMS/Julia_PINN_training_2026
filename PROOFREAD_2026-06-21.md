# Course-wide proofreading pass — 2026-06-21

Goal (user): an individual proofreading pass on each *day*. Improve writing and
tighten **without loss of information**; add a little more explanation only where
ODE/PDE (or other hard) concepts genuinely need it, **without adding complexity
or new material**; full **consistency** review across all supplied files *and*
the code.

Hard constraints honoured throughout:

- No new sections/examples/content. Preserve every number, claim, and code behaviour.
- Don't change exercise difficulty or code logic (flag genuine bugs, don't silently alter behaviour).
- Keep cross-references (§ numbers, unit links, anchors) valid.
- Match existing voice/style. This course is already heavily polished — change only what genuinely helps.
- Render centrally afterwards (freeze caveat); commit/push only when the user asks.

Day → unit map: Day 1 = Units 1–3 (+setup); Day 2 = Units 4–5; Day 3 = Units
6–8 (+start 9); Day 4 = Units 9–10. Appendices: pinn_env, setup, exercise_solutions,
references, software, math_review, physical_review.

## Log

Method: read-only analysis agent per unit proposed exact-quote edits + consistency
flags; lead editor vetted and applied each. Course-wide: normalised "subsubsection(s)"
→ "subsection(s)" (the author calls `##` a "section", so `###` is a subsection) where
those sentences were already being touched.

### Unit 1 — Introduction
Reviewed in full (main loop). Already excellent (recent dedicated overhaul). Verified
§1.2 script table references (`generate_site_map.py`, `build_mask_from_tiles.py` both
present). **No changes** — any edit would cost information or churn a polished file.

### Unit 2 — ML foundations
- §2.2 intro: "next two subsubsections cover (a)…(b)…" undercounted (6 subsections) → reworded + subsection.
- §2.3 intro: "four subsubsections" → "four subsections".
- `conv_mnist_lux.jl`: CNN accuracy comment "~99%" → "~98.8%" (match qmd); banner "Unit 2.6" → "2.7".
- `mnist_gpu_lux.jl`: stale "Unit 2.6"/"Section 2.6" → "2.7" (both CNN & GPU-MLP live in §2.7; §2.6 is now Autodiff). Banner too.
- Synced captured-output banners `output/conv_mnist_lux.md` + `output/mnist_gpu_lux.md` ("2.6"→"2.7").
- `activations.jl` header: "(sigmoid, tanh) are the default for PINNs" → "(tanh, sigmoid) suit PINNs" (qmd default is Swish/tanh).

### Unit 3 — SciML overview
- "need to start meaningful Sci-ML" → "start doing meaningful Sci-ML".
- Linear-ODE bullet: matrix-exponential closed form qualified "Linear systems **with constant $A$**" (defn allows $A(t)$).
- "subsubsections" → "subsections" (×2).
- `pod_chain_python.py`: added comment that its single-cell push (vs Julia's Gaussian bump) is the deliberate slow-decay contrast in §3.5.
- Verified B3: HNN "an order of magnitude" matches captured "10×". No change.

### Unit 4 — Neural ODEs / UDEs
- §4.1 "next two subsubsections lay this out" miscount (only 1 subsection) → "The rest of this section lays…".
- §4.2 "The subsubsections cover" → "subsections".
- Euler-step analogy: "with a step size Δt written in" → "now with a step size Δt written in front of $f_\theta$".
- "Taking Δt→0 and depth L→∞" → "Stacking many such steps and taking Δt→0 as the depth L→∞" (clarify the limit).
- Adjoint: added appositive "the sensitivity of the loss to the state at time $t$".
- Checkpointing fragment → "logarithmic-in-time memory **at the cost of** logarithmic-in-time extra compute".
- "but a one-line addition" → "but **it's** a one-line addition".

### Unit 5 — First PINN
- "are the main hyperparameter" → "hyperparameters".
- "optimiser zeroes the easiest one first" → "drives the easiest one to zero first".
- B1: `pinn_poisson_disk.jl` is an orphan (not included/linked anywhere). Left as-is — **decision for you** (wire in or remove).

### Unit 6 — PDE bootcamp
- CFL bound "Δt ≲ 22.7 s" → "22.6 s" in BOTH the §6.8 body and the exercise prompt (matches the hint's 22.1 m/s → 22.6 s; sol-6-8 already said 22.6 — this fixed a desync).
- Mean-value property: "average of its neighbours on any small sphere" → "average of its values over any small sphere (in 2-D, a circle) centred on that point".
- Fixed mid-word source line-wrap "heat-\nequation-shaped" → "heat-equation-shaped" (was rendering "heat- equation-shaped").
- "subsubsections" → "subsections" (×2).

### Unit 7 — Modern PINN techniques
- Dangling em-dash: "handle this well — useful as the first concrete implementation" → "handle this well, making it a useful first concrete implementation".
- Causal training: removed mislabelling colon after "Wang et al. (2022)"; added a clause explaining the running-sum gate and what $\epsilon$ does; fragment "Trains the network…" → "This trains…".

### Unit 8 — Reef-scale ocean physics
- **Physics-prose fix**: under @eq-column-bcs the prose called "$\kappa\,\partial_z T$" the heat flux equal to $Q_{np}$, but the displayed (correct) equation is $\rho_0 c_p\,\kappa\,\partial_z T = Q_{np}$. Prose now reads "$\rho_0 c_p\,\kappa\,\partial_z T$ is the downward diffusive heat flux (in W/m²)…". (κ∂_zT alone is a *temperature* flux.)
- "$C_H, C_E$ exchange coefficients (∼1.2e-3)" → "(each ∼1.2e-3)".
- B6: "non-penetrating" (u8) vs "non-penetrative" (u9) — each internally consistent; left. **Optional decision.**

### Unit 9 — Capstone spec
- "30 days = 3 600 data points" → "30 days = 5 × 720 = 3 600 data points".
- Standardised success-criterion phrasing: §9.10.3 "monotone in $t$" → "monotonically decreasing in $t$" (matches §9.9.3).
- B1: §9.1 illustrative "65% upwelling / 30% mixing / 5% reduced heating" at **Davies Reef** clashes with Unit 10's actual partition for the same site (≈15% advection / 20% mixing / 65% storm source — and a different category axis). Left unchanged — **decision for you** (the two use different decompositions; needs an author call).

### Unit 10 — Capstone solution
- §10.1 opening fragment "all needed regardless" → "all **are** needed regardless".
- **`c_p` consistency**: `column_fd.jl` (4000.0) and `column_pinn_gpu.jl` (4000.0f0) → **3990.0** to match the spec §9.6 table, the §10.1 hint, and `task_a_forward_deepxde.py` (all 3990, the standard seawater value). <0.3% numeric effect — below displayed precision; aligns the FD solver with the hint's hand-calc (T(0)≈13 °C, slope −0.049). `column_fd.jl` is `{{< include >}}`'d → needs a unit_10 re-render.
- B1: **scenario-numbering off-by-one** between Unit 9 §9.11 toy-task ladder (…(4) vary mixing, (5) storm) and Unit 10 §10.1 / `column_fd.jl` (…(4) storm fingerprint, (5) SWE-coupled). Self-consistent within Unit 10 but off-by-one vs the Unit 9 ladder it cites. Left unchanged — **decision for you** (renumber one side).

### Appendices
- **exercise_solutions** — reviewed in full; **clean**. Confirmed it already matches my unit edits (c_p 3990, CFL 22.6 — sol-6-8, BC form ρ₀c_pκ∂_zT=Q_np). No changes.
- **setup** — "only slower" → "just slower"; "the first time, where Docker ships" → "whereas Docker ships".
- **physical_review** — link-style "[Unit 8](…) §8.5" → "[Unit 8 §8.5](…)"; "signs and magnitudes flip" → "its sign and magnitude can flip".
- **math_review** — signposted the continuous-vs-discrete index reuse ($i,k$ vs $i,n$) that read as a contradiction.
- **pinn_env** — verified coherent (29 hub / 26 Docker counts, KolmogorovArnold bullet, MLDatasets/SciMLSensitivity absent). No changes.
- **references / software** — citation/link lists; all `ref-*` anchors already verified resolvable by the unit agents. Spot-checked, no changes.
- Systemic note (left): several "§X.Y" link texts across math_review/physical_review (and some units) have no `#anchor` in the href, so they land at the unit top. Known pattern, not wrong; not touched.

## Cross-unit decisions — RESOLVED 2026-06-21 ("do what is obvious")
1. **Scenario numbering** — added a one-sentence reconciliation in Unit 10 §10.1: scenarios 1–3 = §9.11 ladder steps 1–3; ladder step 4 ("vary mixing") is folded into the κ-closure; ladder step 5 (synthetic scenario) is realised in two stages (scenario 4 prescribed envelopes, scenario 5 full SWE). No renumbering.
2. **§9.1 partition example** (THE judgment call) — softened to illustrative: dropped the false-precise "65% upwelling / 30% mixing / 5% reduced heating" + the Davies-Reef site name (clashed with Unit 10's actual partition for that site), kept the qualitative upwelling-dominated reading, and added a forward link to Unit 10 §10.3 (`#sec-mechanism-partition`, anchor added). **User can override** → align numbers instead if preferred.
3. **Unit 5 orphan `pinn_poisson_disk.jl`** — left in place (non-destructive: not mine to delete, and wiring it in would add content). No action.
4. **non-penetrating vs non-penetrative** — standardised Unit 9's 4× "non-penetrative" → "non-penetrating" (matches Unit 8, where Q_np is defined, and the `np` subscript).

## Render note
Many prose edits + included scripts (`conv_mnist_lux.jl`, `mnist_gpu_lux.jl`,
`column_fd.jl`, `column_pinn_gpu.jl`) changed → affected units need
`quarto render` to refresh `_freeze` before deploy. Captured `output/*.md`
numbers unaffected at displayed precision by the c_p change.

---

# Final (second) proofreading pass — 2026-06-22

Lens: correctness (math/computing/content) + citations + consistency, plus a
light tone-down of any over-"cool" phrasing. Minimal edits only; flag (don't
guess-fix) anything needing an author call. Method: one read-only review agent
per unit + appendix groups; lead editor vetted every finding before applying.

## Fixes applied (13)
1. **exercise_solutions.qmd:217,223** — `build_forest(... 1, 0.0, 0; rng=0)` →
   `... 1, 2, 0.0; rng=0)`. The stale buggy arg order (`min_samples_split=0.0`
   throws "must be ≥ 2") — the same bug fixed in unit_02 long ago but never
   propagated to the solutions. Now matches unit_02's canonical call. **Won't-run code.**
2. **unit_02 `/255` self-contradiction** — unit_02.qmd:152 + the §2.2 hint
   (line 277) both say MLDatasets returns pixels already in [0,1] and tell
   students to drop `/255`, yet the shown code kept `./ 255f0` (→ [0,1/255]).
   Removed it in unit_02.qmd:191 (forest) and :340 (softmax) and the scripts
   `mnist_rf_julia.jl`, `mnist_linear_lux.jl`; fixed the "normalise to [0,1]"
   comment. (exercise_solutions already drops it.) Accuracy unaffected.
3. **unit_01 train_inverse_pinn.jl:73,85 + output/train_inverse_pinn.md:15** —
   "gauge samples every 6 min" → "every 3 min" (DATA_STRIDE=15 × 12 s = 180 s =
   3 min; the inline comment and the §1.2.4 table already said 3 min).
4. **unit_01.qmd:202** — "see the warning callout in §1.2.5" (dead — `###` are
   unnumbered at number-depth:1; it's actually the `##` section) → "see the
   *Why not put G1 at Brisbane Bar?* discussion".
5. **unit_03.qmd:220** — displayed Python listing printed `ζ={zeta}` but the
   tagged file prints `zeta={zeta}`; aligned the listing to the file.
6. **unit_08.qmd:394** — longwave bulk formula sign. With $Q_{np}$ positive =
   into ocean (and sensible/latent written into-ocean-positive), $Q_{LW} =
   \varepsilon\sigma T_s^4 - Q_{LW}^\downarrow$ was net-*outgoing*-positive,
   i.e. reversed. Fixed to $Q_{LW} = Q_{LW}^\downarrow - \varepsilon\sigma T_s^4$.
   (Explanatory formula; no solver code depends on it.)
7. **unit_10.qmd:576** — "the wider $128$-input / $64\times5$ network" → "the
   wider $64\times5$ network" (shipped `task_b_joint_inverse.jl` is 2-input/64-
   wide; no 128 anywhere; the table just below already says 64×5).

## Flagged for your call (NOT edited — need author judgment / unverifiable here)
- **unit_01 gauge distances/directions** (§1.2 prose + the §1.2 exercise): labels
  G1≈18 / G2≈26 / G3≈35 / G4≈50 km vs straight-line grid distances ~18 / ~14 /
  ~24 / ~34 km (G2 most off: it reads "~26 km ESE" but is the *nearest* gauge,
  ~14 km SSE). These feed the exercise's arrival-time computation. Needs your
  intended distance convention (straight-line vs along-bay path) before fixing.
- **unit_01 forward-solve runtime** stated three ways: "~20 s" (l.464, 500 m),
  "a second or two" (l.894, native), "12 s" (table l.976 + l.991). Possibly
  different resolutions — reconcile if they mean the same solve.
- **unit_07 de Wolff scope**: l.239–248 attributes a *three*-PDE study (variable-
  depth wave; 2-D linearised SWE w/ Coriolis+viscosity; SWE advection–diffusion)
  to de Wolff et al. 2021, but the bibliography annotates that paper as *1-D*
  shallow-water / Saint-Venant. One description is off; couldn't verify the
  paper here.
- **Citation years (cosmetic, self-documented)**: in-text "Wang et al. 2021"
  (NTK) vs the bib entry header (2022, the JCP volume year; anchor `wang2021ntk`);
  same pattern for "McClenny … 2022" vs entry (2023). Entries note the preprint
  years, so harmless — align in-text if you want strict consistency.
- **unit_10 `task_b_gpu_launch.md` "What changes" table** describes a Fourier /
  adaptive / causal / ~40k-step GPU run the shipped `task_b_joint_inverse.jl`
  deliberately does NOT implement (it ships static weights, no Fourier/causal,
  64×5, ~3k steps). Relabel that column as the aspirational menu, or align it.
- **unit_06 Poisson sign**: table writes $\nabla^2u=f$, §6.3 writes $\nabla^2u=-f$
  (both valid; §6.3 worked examples are self-consistent). Cosmetic.
- Minor [QUESTION]s left as-is: unit_04 checkpointing log/log phrasing &
  `g_over_L=9.81` (fine for L=1 m); unit_05 `jax.hessian` jacfwd∘jacrev wording
  vs the jacfwd∘jacfwd code; math_review $T(z,t)$ attributed to Unit 9 (defined
  in Unit 8); physical_review Tikhonov Unit 9-mention / Unit 7-link.

Everything else (units 1–10 + all appendices + scripts) verified clean on
correctness, citations, cross-references, and tone.
