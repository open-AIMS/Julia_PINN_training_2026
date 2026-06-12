# Review notes — collected while adding section exercises (2026-06-13)

Typos, inconsistencies, and technical nits noticed during the full
read-through. Not fixed — collected for triage. Delete this file once
processed.

## Likely genuine errors

1. **Unit 4 §4.1 — Lotka–Volterra called a "limit cycle" (twice).**
   "Limit cycles — closed orbits that *attract* nearby trajectories …
   the simplest example is Lotka–Volterra" and later "a structurally
   robust limit cycle". LV's closed orbits are a continuum of neutral
   centres (a conserved quantity exists — the unit itself plots $V$ in
   §4.1/ex-4-1); they attract nothing and are *not* structurally robust
   (any perturbation destroys them). The standard "simplest limit
   cycle" example is Van der Pol. Suggest rewording both spots.

2. **Unit 8 §8.6 vs Unit 9 §9.1 — Davies Reef depth.** Unit 8 §8.6:
   "the canonical mid-shelf column (Davies Reef in §9.1) with
   $H = 100$ m". Unit 9's site table gives Davies Reef $H = 60$ m;
   $H = 100$ m is Myrmidon. Either the name or the number is wrong.
   (Related: with the §8.6 numbers, $\mathrm{Pe} = w_0H/\kappa_m = 1$
   for $H = 100$ m, yet Unit 9 labels Myrmidon "$\mathrm{Pe} \gg 1$" —
   the label only holds using the interior $\kappa_b$; see ex-8-6.)

3. **Unit 8 §8.4 vs Unit 9 §9.5 — two-band Paulson–Simpson constants
   disagree.** Unit 8: $k_1 \approx 0.5\,\mathrm{m^{-1}}$ (1/k = 2 m),
   $k_2 \approx 1/14\,\mathrm{m^{-1}}$ (14 m). Unit 9: "a 0.35 m red
   band and a 23 m blue–green band". Both cite Paulson–Simpson; the
   PS-77 Type-I water constants are ζ₁ = 0.35 m, ζ₂ = 23 m (Unit 9's
   numbers). Unit 8's 2 m / 14 m pair is a different water type (or an
   error) — should be reconciled or the water-type difference stated.

4. **Unit 5 §5.3 — bullet list contradicted the code beneath it**
   (bullets said ForwardDiff-inner / "forward-over-reverse"; code used
   Zygote-inner). **FIXED 2026-06-13, and then some:** on current
   package versions the Zygote-inner (reverse-over-reverse) cell
   *errors outright* ("Mutating arrays is not supported"), and the
   ForwardDiff-inner alternative silently drops the parameter gradient
   of the derivative term (Zygote warns "cannot track gradients with
   respect to f"). The cell now uses a central difference for the
   inner time derivative — Zygote differentiates exactly through two
   plain network evaluations — with a listing comment explaining the
   AD-inside-AD caveats. Bullets, intro, and the dependent exercise
   solutions (sol-5-3, sol-7-5) updated to match.

4b. **`Manifest.toml` is untracked** (root `.gitignore` excludes
   `Manifest*.toml`), so every freeze recompute — local or via the
   Recompute action — resolves package versions afresh. That is
   exactly how the §5.3 cell silently went from working (old committed
   freeze) to erroring (tonight's re-execution): Zygote/Lux moved
   underneath it. The `.gitignore`'s own comment says manifests
   "should be committed for applications that require a static
   environment" — this repo is such an application. Recommend
   committing `Manifest.toml` (and refreshing it deliberately).

5. **Unit 2 §2.2 / §2.3 — possible double normalisation of MNIST.**
   `flatten(x) = reshape(Float32.(x), …) ./ 255f0`: `MLDatasets.MNIST`
   features are already Float32 in [0, 1], so the extra `./ 255`
   shrinks inputs to [0, 0.004]. Harmless for the forest (split-point
   invariant), slightly distorting for the linear/MLP models (works,
   but the stated "normalise to [0, 1]" comment is wrong either way).
   Worth checking against the actual scripts in `units/unit_02/scripts/`.

## Inconsistencies / cross-reference nits

6. **Unit 1 §1.2 figure caption — "§1.2.5".** The course never numbers
   subsubsections; there is no "§1.2.5" anywhere else. Refers to the
   "Why not put G1 at Brisbane Bar?" callout — better to name it.

7. **Unit 10 §10.1 — "the four toy scenarios from Unit 9 §9.11".**
   §9.11's ladder has **five** rungs (scenario 5 is the synthetic
   forward). §10.1 indeed only works through 1–4, but the phrase "the
   four toy scenarios from §9.11" reads as if the ladder had four.
   Also §10.1's later sentence says "the four toy scenarios" while the
   §10 intro says "the toy-task ladder".

8. **Unit 10 title — "Project Solution"** lacks the "Unit 10:" prefix
   that every other unit title carries (sidebar shows it fine via
   _quarto.yml, but the page H1 and toc-title break the pattern).

9. **Unit 9 §9.6 table — $T_f$ row**: value "30 days", units column
   says "s". Either give 2.592×10⁶ s or change the units cell.

10. **Unit 8 §8.2 — ρ vs ρ₀ notation.** @eq-column-bcs uses $\rho$,
    the explanatory text below uses $\rho_0$ ("$\rho_0 ≈ 1025$ …"),
    and Unit 9 §9.4 uses $\rho_0$ in the same BC. Pick one.

11. **Coordinate style "−27.210°S".** Unit 1's gauge table and Unit 9's
    site table both write negative latitude *and* a S suffix
    (redundant/contradictory: −27.210° already means south). Also
    Unit 9's Site B longitude has 3 decimals (147.647°E) vs 2–3
    elsewhere — minor precision inconsistency.

12. **Unit 2 §2.5 — "what makes a PINN frame work"** — presumably
    "framework work" / "framework function"; current phrasing reads
    like a typo (or an unflagged pun).

13. **Unit 3 §3.4 intro — "The next three subsubsections cover (a),
    (b), (c)"** but the section has four subsubsections (the fourth,
    "When discovery beats approximation", is mentioned separately —
    fine, but easy to misread; same pattern is used inconsistently
    across units).

14. **Unit 10 DeepXDE sketch vs Unit 9 reference values.** The script
    uses `T_deep = 22.0`, IC `25 + 3·tanh((z+5)/2)`, `Q_np = −120`,
    ζ-like decay 8 m — none of which match the §9.6 reference table
    (T_deep 18, Q_cool 200, ζ 10) nor the Site A values. If the
    numbers are deliberately "Site-A-flavoured", a one-line note would
    prevent readers diffing them against the spec.

## Bigger finding (deployment, not prose) — FIXED 2026-06-13

15. **`freeze: true` blocked prose-only updates from deploying.** The
    rendered site (and the GitHub Pages deploy, which runs plain
    `quarto render`) served each unit's markdown from the committed
    `_freeze/` cache. Commit 8f74ce6 ("Cosmetic pass…", 2026-06-11)
    states "deploys via the light path (no recompute needed)" — but
    its fixes were **not** in the rendered HTML (e.g. unit_06 still
    showed the pre-fix "coastal coastlines") until the 2026-06-13
    re-render.

    **Fix applied:** `_quarto.yml` now uses `freeze: auto` — a changed
    source re-executes locally (`make course`) and refreshes its
    `_freeze` entry, which is committed alongside the qmd; on CI a
    stale cache fails loudly (no Julia installed) instead of silently
    deploying stale content. Comments in `_quarto.yml`, `Makefile`,
    and `.github/workflows/build_course.yaml` updated to describe the
    real semantics. All units + index re-rendered on 2026-06-13, so
    the stranded June-11 fixes are now baked in.
