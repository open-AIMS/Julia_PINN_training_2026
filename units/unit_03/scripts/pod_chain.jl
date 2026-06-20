#!/usr/bin/env julia
# POD / reduced-order modelling on a 50-mass damped spring chain.
# A smooth initial bump excites mainly the low-frequency normal modes, so the
# snapshot SVD compresses the 100-dimensional state into a few POD modes.
# Produces a 3-panel figure: singular-value spectrum, reconstruction error vs
# number of modes kept, and the leading POD mode shapes.

using OrdinaryDiffEq, LinearAlgebra, Plots

# 50-mass chain: x¨ᵢ = ω²(xᵢ₋₁ - 2xᵢ + xᵢ₊₁) - 2ζω x˙ᵢ  (fixed at endpoints)
M, ω, ζ = 50, 2π, 0.02
function chain!(du, u, p, t)
    x, v = @views u[1:M], u[M+1:end]
    du[1:M] .= v
    @inbounds for i in 1:M
        l = i == 1 ? 0.0 : x[i-1]
        r = i == M ? 0.0 : x[i+1]
        du[M+i] = ω^2 * (l - 2x[i] + r) - 2ζ * ω * v[i]
    end
end

u0 = zeros(2M)
for i in 1:M; u0[i] = exp(-((i - M/2) / 5.0)^2); end   # smooth Gaussian bump
sol = solve(ODEProblem(chain!, u0, (0.0, 8.0)), Tsit5(); saveat = 0.02)

# Snapshot matrix (rows = positions, columns = time samples), then SVD
X = reduce(hcat, [u[1:M] for u in sol.u])
U, Σ, V = svd(X)

println("first 10 singular values (relative):")
foreach(i -> println("  σ_$i / σ_1 = ", round(Σ[i] / Σ[1]; digits = 4)), 1:10)

p1 = plot(1:15, Σ[1:15] ./ Σ[1]; lw = 2.4, marker = :circle, ms = 4, c = :steelblue,
          yaxis = :log, xlabel = "mode i", ylabel = "σᵢ / σ₁", legend = false,
          title = "singular-value spectrum", framestyle = :box, gridalpha = 0.25)

errs = [norm(X - U[:, 1:k] * Diagonal(Σ[1:k]) * V[:, 1:k]') / norm(X) for k in 1:15]
p2 = plot(1:15, errs; lw = 2.4, marker = :circle, ms = 4, c = :firebrick,
          yaxis = :log, xlabel = "modes kept k", ylabel = "rel. reconstruction error",
          legend = false, title = "reconstruction error", framestyle = :box, gridalpha = 0.25)

p3 = plot(xlabel = "mass index", ylabel = "mode amplitude", title = "first 3 POD modes",
          framestyle = :box, gridalpha = 0.25, legend = :topright)
for k in 1:3
    plot!(p3, 1:M, U[:, k]; lw = 2.2, label = "mode $k")
end

plt = plot(p1, p2, p3; layout = @layout([a b; c _]), size = (900, 660),
           bottom_margin = 5Plots.mm, left_margin = 5Plots.mm)
savefig(plt, joinpath(@__DIR__, "..", "figures", "pod_spectrum.png"))
println("wrote figures/pod_spectrum.png")
