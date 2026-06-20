#!/usr/bin/env julia
# Illustrate what lives on a single KAN edge: a learnable univariate function φ
# built as a weighted sum of `grid_len` fixed basis bumps (radial basis
# functions, as in KolmogorovArnold.jl).  Left panel: the basis bumps. Right
# panel: a particular φ = Σ wₖ·bumpₖ — the bumps are fixed, the heights wₖ are
# the trainable parameters.  Output: figures/kan_splines.png  (used in §2.8).

using Plots, Printf

const FIG_DIR = normpath(joinpath(@__DIR__, "..", "figures"))
isdir(FIG_DIR) || mkpath(FIG_DIR)

grid_len = 6
centres  = range(-1, 1; length = grid_len)      # bump centres on the squashed input range
h        = 0.5 * step(centres)                   # bump width (overlap so the sum is smooth)
bump(x, c) = exp(-((x - c) / h)^2)               # one radial basis function

xs = range(-1, 1; length = 400)

# Left: the grid_len fixed basis bumps.
p_basis = plot(; title = "$(grid_len) fixed basis bumps on one edge",
               titlefontsize = 10, xlabel = "input (after normalizer)",
               ylabel = "bump value", legend = false, ylims = (-0.05, 1.05))
for (k, c) in enumerate(centres)
    plot!(p_basis, xs, bump.(xs, c); lw = 2, c = k)
    scatter!(p_basis, [c], [1.0]; ms = 3, c = k, msw = 0)
end

# Right: one learned φ = Σ wₖ·bumpₖ. Pick illustrative (fixed) heights so the
# sum has a clear peak and dip — these wₖ are what training would learn.
w = [-0.6, 0.9, 0.2, -0.8, 1.1, 0.1]
φ(x) = sum(w[k] * bump(x, centres[k]) for k in 1:grid_len)

p_phi = plot(; title = "one learnable edge function  φ = Σ wₖ·bumpₖ",
             titlefontsize = 10, xlabel = "input (after normalizer)",
             ylabel = "φ(input)", legend = false)
for (k, c) in enumerate(centres)             # faint scaled bumps that sum to φ
    plot!(p_phi, xs, w[k] .* bump.(xs, c); lw = 1, c = k, alpha = 0.35, ls = :dash)
end
plot!(p_phi, xs, φ.(xs); lw = 3.5, c = :black)

p = plot(p_basis, p_phi; layout = (1, 2), size = (900, 360),
         left_margin = 5Plots.mm, bottom_margin = 5Plots.mm)
savefig(p, joinpath(FIG_DIR, "kan_splines.png"))
println("wrote figures/kan_splines.png  (grid_len = $grid_len bumps, $(grid_len) trainable heights per edge)")
