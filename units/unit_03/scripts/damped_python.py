"""Damped harmonic oscillator: under-, critically-, over-damped.

ẍ + 2ζω ẋ + ω²x = 0, with ω = 2π and ζ ∈ {0.1, 1.0, 2.0}.
Pairs with the Julia OrdinaryDiffEq demo in Unit 3 §3.1.
"""

import numpy as np
from scipy.integrate import solve_ivp

omega = 2 * np.pi
def damped(t, u, zeta):
    return [u[1], -omega**2 * u[0] - 2 * zeta * omega * u[1]]

for zeta in (0.1, 1.0, 2.0):
    sol = solve_ivp(damped, (0.0, 3.0), [1.0, 0.0],
                    args=(zeta,), t_eval=np.linspace(0, 3, 301))
    print(f"zeta={zeta}: x(3.0) = {sol.y[0, -1]:+.4f}")
