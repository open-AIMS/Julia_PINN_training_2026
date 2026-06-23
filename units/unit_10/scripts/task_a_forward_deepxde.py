# ===========================================================================
# Task A forward column problem (Cleveland Bay, H = 15 m, fixed tau) as a
# DeepXDE sketch — the Python parallel to the Julia NeuralPDE.jl version.
# Same architecture (4 x 32 tanh MLP), same hard-BC ansatz at the deep
# reservoir, same Adam -> L-BFGS schedule.  See unit_10 §10.4.
#
# This is a *forward* solve (tau known); it is the Python-ecosystem analogue,
# not part of the Julia inverse pipeline.  Runtime ~5 min on a laptop CPU with
# the PyTorch backend.
#
#   pip install deepxde[torch]
#   python task_a_forward_deepxde.py
# ===========================================================================
import numpy as np
import deepxde as dde

H = 15.0;  Tf = 30.0 * 86400.0
T_deep = 22.0
def kappa_z(z):  return 1e-3 * np.exp(z / 5.0) + 1e-5     # mixed-layer profile
def w_z(z, t):   return 0.0                              # forward problem: no wind
def Q_np(t):     return -120.0                           # net cooling (W/m^2)
def S_z(z, t):   return 400.0 / 4.0e6 / 8.0 * np.exp(z / 8.0)  # body source S = Q_SW/(rho0*cp*zeta)*e^(z/zeta), zeta=8 m

# Geometry: z in [-H, 0], t in [0, Tf]
geom     = dde.geometry.Interval(-H, 0.0)
timedom  = dde.geometry.TimeDomain(0.0, Tf)
geomtime = dde.geometry.GeometryXTime(geom, timedom)

def pde(zt, T):
    z, t = zt[:, 0:1], zt[:, 1:2]
    dT_dt = dde.grad.jacobian(T, zt, j=1)
    dT_dz = dde.grad.jacobian(T, zt, j=0)
    d2T_dz2 = dde.grad.hessian(T, zt, component=0, i=0, j=0)
    kappa = kappa_z(z)
    return dT_dt + w_z(z, t) * dT_dz - kappa * d2T_dz2 - S_z(z, t)

# Hard BC at the deep reservoir via output transform.
def output_transform(zt, T):
    z = zt[:, 0:1]
    return T_deep + (z + H) * T          # T(-H, t) = T_deep automatically

# Soft surface flux at z = 0.
def surface_flux(zt, T, _):
    dT_dz_top = dde.grad.jacobian(T, zt, j=0)
    rho, cp = 1025.0, 3990.0
    return rho * cp * kappa_z(zt[:, 0:1]) * dT_dz_top - Q_np(zt[:, 1:2])
bc_surface = dde.icbc.OperatorBC(
    geomtime, surface_flux,
    lambda zt, on_boundary: on_boundary and np.isclose(zt[0], 0.0),
)

# IC: initial thermocline tanh profile
def T0(z): return 25.0 + 3.0 * np.tanh((z + 5.0) / 2.0)
ic = dde.icbc.IC(geomtime, lambda zt: T0(zt[:, 0:1]),
                 lambda _, on_initial: on_initial)

data = dde.data.TimePDE(
    geomtime, pde, [bc_surface, ic],
    num_domain=2000, num_boundary=200, num_initial=200,
)
net = dde.nn.FNN([2] + [32] * 4 + [1], "tanh", "Glorot uniform")
net.apply_output_transform(output_transform)
model = dde.Model(data, net)
model.compile("adam", lr=1e-3); model.train(iterations=2000)
model.compile("L-BFGS"); model.train()
