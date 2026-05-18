"""Simplest ODE: exponential decay, in scipy.integrate.solve_ivp.

Pairs with the Julia OrdinaryDiffEq Tsit5 demo in Unit 3 §3.1.
"""

import numpy as np
from scipy.integrate import solve_ivp

lam, x0 = 0.5, 1.0
sol = solve_ivp(lambda t, x: -lam * x,
                t_span=(0.0, 6.0), y0=[x0],
                method="RK45", dense_output=True)

t = np.linspace(0, 6, 121)
print("max |numerical - exact| =",
      float(np.max(np.abs(sol.sol(t)[0] - x0 * np.exp(-lam * t)))))
