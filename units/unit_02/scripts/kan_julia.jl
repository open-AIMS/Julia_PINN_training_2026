#!/usr/bin/env julia
# Toy KAN example in KolmogorovArnold.jl.
#
# Fit f(x1, x2) = exp(sin(pi x1) + x2^2) with a small 2-layer KAN.
#
# Uses KolmogorovArnold.jl (pinned to 0.0.1). It is NOT in the course @pinn
# environment — add it once with Pkg.add("KolmogorovArnold").

using KolmogorovArnold, Lux, Random, Zygote, Optimisers, Statistics

rng = Random.MersenneTwister(0)

n = 1000
X = 2f0 .* rand(rng, Float32, 2, n) .- 1f0
y = reshape(exp.(sinpi.(X[1, :]) .+ X[2, :] .^ 2), 1, n)

# KAN 2 → 5 → 1. KDense(in, out, grid_len): grid_len is the number of basis
# centres on each edge — the spline/RBF resolution. More grid points = a more
# flexible per-edge function = more parameters.
grid_len = 6
model = Lux.Chain(
    KDense(2, 5, grid_len; basis_func = rbf, normalizer = softsign),
    KDense(5, 1, grid_len; basis_func = rbf, normalizer = softsign),
)
ps, st = Lux.setup(rng, model)

loss(ps, st) = mean(abs2, first(model(X, ps, st)) .- y)
opt_state = Optimisers.setup(Optimisers.Adam(1f-2), ps)
for k in 1:2000
    gs = first(Zygote.gradient(p -> loss(p, st), ps))
    opt_state, ps = Optimisers.update(opt_state, ps, gs)
end
@info "test MSE = $(loss(ps, st))"
