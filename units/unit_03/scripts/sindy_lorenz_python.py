"""SINDy on the Lorenz system, in PySINDy.

Generates a long clean Lorenz trajectory, then asks PySINDy to
recover the equations from data alone.  Pairs with the
hand-rolled STLSQ implementation in sindy_lorenz.jl.

Requires:  pip install pysindy
"""

import numpy as np
from scipy.integrate import solve_ivp
import pysindy as ps

sigma, rho, beta = 10.0, 28.0, 8.0 / 3.0
def lorenz(t, u): return [sigma * (u[1] - u[0]),
                          u[0] * (rho - u[2]) - u[1],
                          u[0] * u[1] - beta * u[2]]

t = np.linspace(0, 10, 5001)
sol = solve_ivp(lorenz, (t[0], t[-1]), [-8.0, 7.0, 27.0],
                t_eval=t, rtol=1e-9, atol=1e-9)
X = sol.y.T

model = ps.SINDy(
    feature_library=ps.PolynomialLibrary(degree=2),
    optimizer=ps.STLSQ(threshold=0.1),
)
model.fit(X, t=t)
model.print()
