# Environment Setup

## Julia

The project's Julia environment is declared in `Project.toml`. To install:

```bash
julia --project=. -e 'using Pkg; Pkg.instantiate(); Pkg.precompile()'
```

First-time install is slow (10–30 minutes; NeuralPDE / Lux / ModelingToolkit
have many dependencies). Subsequent activations are fast.

To run the project Julia REPL:

```bash
julia --project=.
```

## Python

A virtualenv lives at `.venv/`, with dependencies pinned in `requirements.txt`.
To create from scratch:

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
```

To run a Python script in the env:

```bash
.venv/bin/python <script.py>
```

Or activate the venv:

```bash
source .venv/bin/activate
python <script.py>
deactivate
```

## Building the course site

```bash
./build.sh           # full render (uses both Julia and Python where present)
./build.sh 2         # render unit 2 only
./build.sh serve     # serve _site/ at http://localhost:8080
./build.sh clean     # remove rendered output
```

## Code execution in Quarto

There are **two execution paths**, separated so render time stays cheap.

### 1. Inline `{julia}` cells

Julia code tagged `{julia}` executes during `quarto render`. Outputs are
cached via Quarto's `freeze: auto` mechanism in `_freeze/` — a cell only
re-runs when its source changes. Force re-execution by deleting `_freeze/`.

Use for: short Julia snippets (a few seconds at most).

### 2. Sidecar scripts (`./build.sh execute`)

For Python and slow Julia, put scripts in `units/unit_NN/scripts/*.{py,jl}`.
Run them with:

```bash
./build.sh execute       # all sidecar scripts
./build.sh execute 2     # only unit 2's scripts
```

Each script's stdout is wrapped in a Quarto cell-output block and saved to
`units/unit_NN/output/<scriptname>.md`. The qmd embeds it with

```
{{< include output/<scriptname>.md >}}
```

Plots: have your script save them to `units/unit_NN/output/<name>.png`
(or `.svg`); embed in qmd as `![](output/<name>.png)`.

Use for: anything Python; Julia code that takes minutes (PINN training).
Outputs are committed alongside the qmd so renders never need to re-run them.

### Worked example

`units/unit_02/scripts/iris_rf.py` runs the random-forest example via the
venv. `./build.sh execute 2` populates `units/unit_02/output/iris_rf.md`.
`units/unit_02/unit_02.qmd` includes that file directly below the static
Python code block — so the rendered page shows the code *and* its captured
output.
