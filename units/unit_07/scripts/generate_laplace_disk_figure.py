"""Generate a clean two-panel figure for Unit 7 §7.1 Laplace-on-a-disk.

Left  — polar heatmap of u(r, θ) = r³ sin(3θ) on the unit disk with an
        inner cutoff r_min.  Diverging colour scale, smooth in (r, θ),
        so the radial fade (r³ → 0 at the centre) is visually correct.
Right — typical PINN collocation cloud: ~400 uniform-on-disk interior
        residual points plus ~36 boundary points on r = R.

Run:  python units/unit_07/scripts/generate_laplace_disk_figure.py
Output: units/unit_07/figures/laplace_disk.png
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

OUT = Path(__file__).resolve().parents[1] / "figures" / "laplace_disk.png"

rng = np.random.default_rng(0)

R = 1.0
r_min = 0.05

fig, axes = plt.subplots(
    1, 2,
    figsize=(11, 5),
    subplot_kw={"projection": "polar"},
)

# ---------- Left panel: closed-form heatmap ----------
ax_l = axes[0]
nr, ntheta = 240, 360
r_grid = np.linspace(0.0, R, nr)
theta_grid = np.linspace(0.0, 2 * np.pi, ntheta)
RR, TT = np.meshgrid(r_grid, theta_grid)
U = (RR ** 3) * np.sin(3 * TT)
vmax = 1.0

mesh = ax_l.pcolormesh(
    TT, RR, U,
    cmap="RdBu_r", vmin=-vmax, vmax=vmax,
    shading="auto",
)
# Inner cutoff annulus
ax_l.fill_between(theta_grid, 0.0, r_min, color="white", linewidth=0)
ax_l.plot(theta_grid, np.full_like(theta_grid, r_min),
          color="black", linestyle=(0, (3, 2)), linewidth=0.8)
# Outer disk boundary
ax_l.plot(theta_grid, np.full_like(theta_grid, R),
          color="black", linewidth=1.2)

ax_l.set_yticklabels([])
ax_l.set_xticks(np.deg2rad([0, 90, 180, 270]))
ax_l.set_xticklabels(["θ = 0", "θ = π/2", "θ = π", "θ = 3π/2"], fontsize=9)
ax_l.set_rlim(0, R * 1.02)
ax_l.set_title(r"Exact solution  $u(r, \theta) = r^{3}\,\sin(3\theta)$",
               fontsize=12, fontweight="bold", pad=14)

cbar = fig.colorbar(mesh, ax=ax_l, fraction=0.046, pad=0.10, shrink=0.85)
cbar.set_label("u", rotation=0, labelpad=10, fontsize=10)
cbar.set_ticks([-1, -0.5, 0, 0.5, 1])

ax_l.text(0, R + 0.30,
          "Dirichlet BC:  u(R, θ) = sin(3θ)",
          ha="center", va="center", fontsize=9, color="#444",
          transform=ax_l.transData)

# ---------- Right panel: collocation cloud ----------
ax_r = axes[1]

# Uniform-on-disk interior samples (~400 points).
n_int = 400
u_unit = rng.random(n_int)
r_int = R * np.sqrt(u_unit)            # uniform-on-disk distribution
theta_int = 2 * np.pi * rng.random(n_int)
mask = r_int > r_min                   # respect the cutoff
ax_r.scatter(theta_int[mask], r_int[mask],
             s=8, c="#27ae60", alpha=0.85, edgecolor="none",
             label=f"{mask.sum()} interior residual points")

# Boundary samples on r = R (uniform in θ).
n_bdry = 36
theta_bdry = np.linspace(0, 2 * np.pi, n_bdry, endpoint=False)
ax_r.scatter(theta_bdry, np.full_like(theta_bdry, R),
             s=22, c="#e67e22", alpha=0.95, edgecolor="none",
             label=f"{n_bdry} boundary points (BC loss)")

# Inner cutoff and outer outline
ax_r.plot(theta_grid, np.full_like(theta_grid, r_min),
          color="black", linestyle=(0, (3, 2)), linewidth=0.8)
ax_r.plot(theta_grid, np.full_like(theta_grid, R),
          color="black", linewidth=1.2)

ax_r.set_yticklabels([])
ax_r.set_xticks(np.deg2rad([0, 90, 180, 270]))
ax_r.set_xticklabels(["", "", "", ""])
ax_r.set_rlim(0, R * 1.02)
ax_r.set_title("Mesh-free collocation cloud", fontsize=12, fontweight="bold", pad=14)

ax_r.legend(loc="lower center",
            bbox_to_anchor=(0.5, -0.13),
            ncol=1, fontsize=9, frameon=False)

ax_r.text(0, R + 0.30,
          "No mesh — random samples on a disk are handled natively",
          ha="center", va="center", fontsize=9, color="#444",
          transform=ax_r.transData)

fig.suptitle(
    "Laplace on a disk — three-fold harmonic ground truth and a PINN collocation cloud",
    fontsize=11, fontweight="bold", y=1.02,
)

fig.tight_layout()
OUT.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(OUT, dpi=170, bbox_inches="tight", facecolor="white")
print(f"wrote {OUT}")
