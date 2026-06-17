# Julia_PINN_training_2026

Material for the training workshop on Physics-informed Neural Networks with Julialang.

## Course site

**View the course online:** <https://open-aims.github.io/Julia_PINN_training_2026/>

The site is published to GitHub Pages automatically on every push to `main`.

## How publishing works

- **Deploy (automatic, light).** Every push to `main` runs the *Deploy course
  site* workflow, which renders the HTML from the committed Quarto freeze cache
  (`_freeze/`) and publishes it. With `freeze: true` in `_quarto.yml` this never
  re-executes the Julia code, so it finishes in a couple of minutes.

- **Recompute (manual, heavy).** When a unit's code or its results change, the
  freeze cache must be refreshed. Either:
  - Run the **Recompute freeze cache** workflow from the GitHub Actions tab and
    pick the unit — it re-executes that unit, commits the refreshed cache, and
    the deploy then runs automatically; or
  - Run it locally and commit the result:
    ```
    make recompute        # re-execute everything (can be slow)
    git add _freeze && git commit -m "Refresh freeze cache" && git push
    ```

Building the whole course from cold can exceed GitHub's 6-hour job limit, so a
full recompute is best done locally; per-unit recomputes run fine in CI.

- **GPU examples (manual, out-of-band — neither workflow runs them).** A few
  units carry GPU-accelerated examples (Unit 1 shallow-water, Unit 2.6 MNIST,
  Unit 5 2-D diffusion PINN, Unit 6 2-D heat finite-difference, Unit 7
  collocation throughput, Unit 10 capstone column PINN + Task A inverse). Their
  code cells are marked `#| eval: false`, so **neither Deploy nor Recompute ever
  executes them** — the CI machines are CPU-only and the freeze cache does not
  depend on them. Instead, each example's results are generated **by hand on the
  GPU hub** (the `@pinn` environment: Lux + LuxCUDA + CUDA) and committed to the
  repo as static artifacts:
  - the captured run output in `units/<unit>/output/*.md` (the CPU-vs-GPU timing
    tables shown in the notes), and
  - the figures in `units/<unit>/figures/*.png`.

  To refresh one, run its script under `@pinn` on the GPU hub, e.g.
  ```
  julia --project=@pinn units/unit_10/scripts/task_a_inverse_pinn.jl
  ```
  then copy the regenerated `output/*.md` and `figures/*.png` back into the repo
  and commit them. A normal push-to-`main` Deploy then publishes the new
  artifacts without re-executing anything. (The scripts are written to run on the
  CPU too, at a smaller collocation budget — handy for a local smoke test before
  the full GPU run; see each script's header.)
