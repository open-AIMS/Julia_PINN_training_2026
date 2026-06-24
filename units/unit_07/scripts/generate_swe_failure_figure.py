"""Render the naïve-SWE-PINN failure figure from REAL run data (§7.2).

Reads the two diagnostics written by `naive_swe_failure_solve.jl`:
  output/naive_swe_snapshot.csv   x, eta_pinn, eta_exact   (η at t = T_f/2)
  output/naive_swe_residual.csv   t_center, mean_residual  (binned |r|²)
and draws a two-panel figure. Both panels are the trained model's own
output; the η reference is the exact d'Alembert solution. Nothing here is
hand-drawn — that is the whole point of regenerating it.

Run (after the Julia solve):
  python units/unit_07/scripts/generate_swe_failure_figure.py
Output: units/unit_07/figures/naive_swe_failure.png
"""

import csv
from pathlib import Path
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parents[1]
OUT  = HERE / "figures" / "naive_swe_failure.png"


def read_csv(name):
    rows = list(csv.reader((HERE / "output" / name).open()))
    header, data = rows[0], rows[1:]
    cols = {h: [float(r[i]) for r in data] for i, h in enumerate(header)}
    return cols


snap = read_csv("naive_swe_snapshot.csv")
resid = read_csv("naive_swe_residual.csv")

fig, (axA, axB) = plt.subplots(1, 2, figsize=(11, 4.2))

# ---------- Panel A: η(x, t = T_f/2) ----------
axA.plot(snap["x"], snap["eta_exact"], color="#222", lw=2.0,
         label="d'Alembert reference (exact)")
axA.plot(snap["x"], snap["eta_pinn"], color="#c0392b", lw=2.0, ls="--",
         label="naïve PINN (trained)")
axA.axhline(0.0, color="#bbb", lw=0.6, ls=":")
axA.set_xlabel("x along channel")
axA.set_ylabel("η")
axA.set_title("η(x, t = T_f/2) — surface elevation", fontsize=12, fontweight="bold")
axA.legend(frameon=False, fontsize=9, loc="upper right")

# ---------- Panel B: residual |r|² vs t (measured, told honestly) ----------
tc = resid["t_center"]
mr = resid["mean_residual"]
width = (tc[1] - tc[0]) * 0.8 if len(tc) > 1 else 0.004
axB.bar(tc, mr, width=width, color="#c0392b", alpha=0.8)
axB.set_xlabel("t")
axB.set_ylabel("mean PDE residual  |r|²")
axB.set_title("PDE residual vs t — small, yet the solution is wrong",
              fontsize=12, fontweight="bold")
axB.annotate(
    "residual is largest at the sharp t = 0 pulse and small later —\n"
    "but Panel A shows η is ~75% wrong: a small PDE residual\n"
    "does not certify a correct solution.",
    xy=(0.97, 0.80), xycoords="axes fraction", ha="right", va="top",
    fontsize=8.5, color="#444")

fig.suptitle("Naïve PINN on the 1-D linearised SWE — what failure looks like",
             fontsize=13, fontweight="bold", y=1.02)
fig.text(0.5, -0.04,
         "Both panels from an actual run (scripts/naive_swe_failure_solve.jl): "
         "the η reference is the exact d'Alembert solution; the PINN curve and "
         "the binned residuals are the trained model's own output.",
         ha="center", va="top", fontsize=8.5, color="#555")

fig.tight_layout()
OUT.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(OUT, dpi=170, bbox_inches="tight", facecolor="white")
print(f"wrote {OUT}")
