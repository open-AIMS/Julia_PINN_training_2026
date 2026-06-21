"""Proper Orthogonal Decomposition on a 50-mass damped chain — Python.

Pairs with pod_chain.jl. Same dynamics, same snapshot SVD pipeline.
"""

import numpy as np
from scipy.integrate import solve_ivp
from scipy.linalg import svd

M, omega, zeta = 50, 2 * np.pi, 0.02

def chain(t, u):
    x, v = u[:M], u[M:]
    du = np.zeros_like(u)
    du[:M] = v
    for i in range(M):
        l = 0.0 if i == 0     else x[i - 1]
        r = 0.0 if i == M - 1 else x[i + 1]
        du[M + i] = omega**2 * (l - 2 * x[i] + r) - 2 * zeta * omega * v[i]
    return du

# Localised single-cell push (vs pod_chain.jl's smooth Gaussian bump): it excites
# *all* normal modes, so the POD spectrum decays slowly and compresses far less —
# the contrasting case discussed in unit_03.qmd §3.5.
u0 = np.zeros(2 * M)
u0[M // 2] = 1.0
sol = solve_ivp(chain, (0.0, 8.0), u0, t_eval=np.linspace(0, 8, 401))

X = sol.y[:M, :]

U, Sigma, Vt = svd(X, full_matrices=False)
print("first 6 singular values (relative):",
      np.round(Sigma[:6] / Sigma[0], 4))
print(f"variance captured by k=6 modes: "
      f"{(Sigma[:6]**2).sum() / (Sigma**2).sum():.4f}")
