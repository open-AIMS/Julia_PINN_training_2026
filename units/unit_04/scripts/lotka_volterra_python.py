"""Lotka–Volterra in Python — three initial conditions, scipy.integrate.

Pairs with the inline Julia OrdinaryDiffEq example in Unit 4 §4.1.
Each initial condition lands on a different closed orbit; this script
just prints the final state of each trajectory.
"""

import numpy as np
from scipy.integrate import solve_ivp

alpha, beta, gamma, delta = 1.5, 1.0, 3.0, 1.0
def lv(t, u):
    x, y = u
    return [alpha * x - beta * x * y,
            delta * x * y - gamma * y]

t_eval = np.linspace(0, 12, 1201)
for u0 in ([1.0, 1.0], [1.5, 1.0], [2.0, 1.0]):
    sol = solve_ivp(lv, (0, 12), u0, t_eval=t_eval, rtol=1e-8)
    print(f"x(0)={u0[0]:.1f}  x(12)={sol.y[0, -1]:.3f}  "
          f"y(12)={sol.y[1, -1]:.3f}")
