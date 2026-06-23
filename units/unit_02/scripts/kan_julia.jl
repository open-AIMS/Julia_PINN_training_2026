#!/usr/bin/env julia
# A two-input KAN. Approximate the spatiotemporal field
#
#   u(x, t) = e^{-t} sin(pi x)
#
# — a single decaying eigenmode of the heat equation on (x, t) in [0, 1]^2 —
# with a small Kolmogorov-Arnold network, then plot the learned field next to
# the exact one. Two inputs (x, t) in, one scalar out: the same shape a PINN's
# solution network has.

using KolmogorovArnold, Lux, Random, Zygote, Optimisers, Statistics, Plots

rng = Random.MersenneTwister(0)
u(x, t) = exp(-t) * sinpi(x)                  # the field we want the KAN to learn

# training data: random (x, t) samples in the unit square
n = 2000
X = rand(rng, Float32, 2, n)                  # row 1 = x, row 2 = t
y = reshape(u.(X[1, :], X[2, :]), 1, n)

# KAN 2 → 5 → 1. KDense(in, out, grid_len): grid_len is the number of basis
# centres on each edge — the per-edge spline/RBF resolution. More centres = a
# more flexible learnable φ, and more parameters.
grid_len = 6
model = Lux.Chain(
    KDense(2, 5, grid_len; basis_func = rbf, normalizer = softsign),
    KDense(5, 1, grid_len; basis_func = rbf, normalizer = softsign),
)
ps, st = Lux.setup(rng, model)

loss(ps) = mean(abs2, first(model(X, ps, st)) .- y)
opt_state = Optimisers.setup(Optimisers.Adam(1f-2), ps)
for k in 1:3000
    global ps, opt_state            # reassign the script-level vars (soft scope)
    gs = first(Zygote.gradient(loss, ps))
    opt_state, ps = Optimisers.update(opt_state, ps, gs)
end
@info "final training MSE = $(loss(ps))"

# evaluate the trained KAN on a regular grid and compare to the exact field
Ng = 100
xs = range(0f0, 1f0; length = Ng)
ts = range(0f0, 1f0; length = Ng)
grid  = reduce(hcat, [[x, t] for t in ts for x in xs])
û     = reshape(first(model(grid, ps, st)), Ng, Ng)   # KAN prediction, indexed [x, t]
exact = [u(x, t) for x in xs, t in ts]

base = (xlabel = "x", ylabel = "t", aspect_ratio = :equal,
        xlims = (0, 1), ylims = (0, 1), framestyle = :box)
p1 = heatmap(xs, ts, exact'; title = "exact  u(x,t) = e⁻ᵗ sin(πx)", c = :viridis, base...)
p2 = heatmap(xs, ts, û';     title = "KAN approximation",           c = :viridis, base...)
p3 = heatmap(xs, ts, abs.(û .- exact)'; title = "|KAN − exact|",    c = :magma,   base...)
plt = plot(p1, p2, p3; layout = (1, 3), size = (1150, 360),
           bottom_margin = 5Plots.mm, left_margin = 5Plots.mm)
savefig(plt, joinpath(@__DIR__, "..", "figures", "kan_xt.png"))
