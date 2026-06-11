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
