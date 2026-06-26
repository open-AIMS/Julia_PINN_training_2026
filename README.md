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

## PDF booklet (offline notes)

The whole course can be rendered to a **single PDF** — all units plus the
appendices, with a title page, unified table of contents and continuous page
numbers. It is a **local, on-demand build**: it is *not* produced in CI and is
*not* published with the site (the output `_booklet/` is git-ignored). Only the
build tooling lives in the repo.

```
./booklet/build.sh           # builds _booklet/…Course-Notes.pdf
./booklet/build.sh --open    # …and opens it
./booklet/build.sh --help    # usage + notes
```

**No Julia recompute.** Quarto's freeze cache is format-specific (`html.json`
for the site vs `tex.json` for PDF), but the freeze *hash* is computed from the
document source, not the output format. So the script copies each committed
`_freeze/**/execute-results/html.json` into the `tex.json` slot (dropping its
HTML-only payload); `freeze: auto` then reuses it and the PDF render executes
**zero** code cells. A full build takes about a minute. (If the freeze is stale
you get stale-but-valid output — never a Julia run. Refresh the freeze the usual
way, see *Recompute* above, to update the booklet's content.)

**Dependencies.**

| Tool | macOS | Linux (Debian/Ubuntu) |
|---|---|---|
| Quarto | <https://quarto.org/docs/get-started/> | same |
| `xelatex` | `brew install --cask mactex-no-gui` *or* `quarto install tinytex` | `apt install texlive-xetex texlive-fonts-recommended` |
| `rsvg-convert` (SVG→PDF figures) | `brew install librsvg` | `apt install librsvg2-bin` |
| JuliaMono font (Unicode glyphs in code: ζ, τ, ∂, λ, T̃ …) | `brew install --cask font-juliamono` | drop the `JuliaMono-*.ttf` into `~/.fonts` and `fc-cache -f` |
| `python3` | preinstalled | `apt install python3` |

`build.sh` checks for each tool and prints the matching install command if one
is missing, finds JuliaMono across the standard macOS/Linux font directories,
and generates `booklet/header.tex` (git-ignored).

**Configuration.** `_quarto-booklet.yml` is a Quarto *profile* — it is inert for
the normal site build and only takes effect under `--profile booklet` (run for
you by `build.sh`). It lists the chapters/appendices and the PDF settings.
`booklet/fix-refs.lua` neutralises the hand-written `::: {#ref-*}` bibliography
divs that LaTeX would otherwise reject.

**Cover.** The title page records the authorship (Yoni Nazarathy, Accumulation
Point Pty Ltd; developed in collaboration with AIMS, contact Dr Takuya Iwanaga),
and the **exact git commit and date the PDF was built from**, with a link back to
this repository — license and attribution are as per this repo.
