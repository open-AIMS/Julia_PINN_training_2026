#!/usr/bin/env julia
# Proper Orthogonal Decomposition on a 50-mass damped chain.
#
# Builds a snapshot matrix of the chain's positions over time, takes
# the SVD, and reports the singular-value spectrum + reconstruction
# error vs. number of POD modes kept.

using OrdinaryDiffEq, LinearAlgebra, Plots

M, ω, ζ = 50, 2π, 0.02

function chain!(du, u, p, t)
    x, v = @views u[1:M], u[M+1:end]
    du[1:M] .= v
    @inbounds for i in 1:M
        l = i == 1     ? 0.0 : x[i - 1]
        r = i == M     ? 0.0 : x[i + 1]
        du[M + i] = ω^2 * (l - 2x[i] + r) - 2ζ * ω * v[i]
    end
end

u0 = zeros(2M); u0[M ÷ 2] = 1.0
sol = solve(ODEProblem(chain!, u0, (0.0, 8.0)), Tsit5(); saveat = 0.02)

X = reduce(hcat, [u[1:M] for u in sol.u])
U, Σ, V = svd(X)

println("first 10 singular values (relative):")
for i in 1:10
    println("  σ_$i / σ_1 = ", round(Σ[i] / Σ[1]; digits = 4))
end

k_range = 1:15
errs = [norm(X - U[:, 1:k] * Diagonal(Σ[1:k]) * V[:, 1:k]') / norm(X)
        for k in k_range]

println("\nrelative reconstruction error vs k:")
for (k, e) in zip(k_range, errs)
    println("  k = $k  err = $(round(e; sigdigits = 3))")
end
