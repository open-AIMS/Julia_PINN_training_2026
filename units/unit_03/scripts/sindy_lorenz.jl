# SINDy on the Lorenz system.
#
# Simulate a Lorenz trajectory, add observation noise, and recover the
# symbolic equations by sparse regression against a degree-2 polynomial
# library — the worked example in unit_03.qmd §3.4.
#
# Algorithm: sequentially thresholded least squares (STLSQ), the
# canonical SINDy update from Brunton, Proctor & Kutz (2016).
# Derivatives are obtained by second-order central differences; the
# qmd notes that derivative estimation is a separate art and we keep
# this script simple by relying on a dense save interval.
#
# Run via ./build.sh execute 3 (writes output to ../output/sindy_lorenz.md).

using OrdinaryDiffEq, Random, LinearAlgebra, Printf, Statistics

# ── 1. simulate Lorenz ─────────────────────────────────────────────────
σ_true, ρ_true, β_true = 10.0, 28.0, 8/3
function lorenz!(du, u, p, t)
    x, y, z = u
    du[1] = σ_true * (y - x)
    du[2] = x * (ρ_true - z) - y
    du[3] = x * y - β_true * z
end

u0    = [-8.0, 7.0, 27.0]
tspan = (0.0, 20.0)
dt    = 0.002
sol   = solve(ODEProblem(lorenz!, u0, tspan), Tsit5();
              saveat = dt, abstol = 1e-10, reltol = 1e-10)

X_clean = permutedims(reduce(hcat, sol.u))            # N × 3

# ── 2. add observation noise ───────────────────────────────────────────
rng = Random.MersenneTwister(0)
noise_level = 0.01                                    # 1 % of per-axis std
σ_obs = noise_level .* vec(std(X_clean; dims = 1))'
X = X_clean .+ σ_obs .* randn(rng, size(X_clean))

# ── 3. central-difference derivatives, trim end points ─────────────────
Ẋ = (X[3:end, :] .- X[1:end-2, :]) ./ (2 * dt)
Xc = X[2:end-1, :]

# ── 4. degree-2 polynomial feature library ─────────────────────────────
# columns: 1, x, y, z, x², xy, xz, y², yz, z²
function poly_library(X)
    x, y, z = X[:, 1], X[:, 2], X[:, 3]
    hcat(ones(length(x)), x, y, z,
         x.^2, x.*y, x.*z, y.^2, y.*z, z.^2)
end
labels = ["1", "x", "y", "z", "x^2", "x*y", "x*z", "y^2", "y*z", "z^2"]
Θ = poly_library(Xc)

# ── 5. STLSQ — sequentially thresholded least squares ──────────────────
function stlsq(Θ, Ẋ, λ; n_iters = 15)
    Ξ = Θ \ Ẋ
    for _ in 1:n_iters
        small = abs.(Ξ) .< λ
        Ξ[small] .= 0
        for j in axes(Ξ, 2)
            keep = .!small[:, j]
            sum(keep) == 0 && continue
            Ξ[keep, j] = Θ[:, keep] \ Ẋ[:, j]
        end
    end
    Ξ
end

λ = 0.1
Ξ = stlsq(Θ, Ẋ, λ)

# ── 6. report ──────────────────────────────────────────────────────────
state_names = ["dx/dt", "dy/dt", "dz/dt"]

println("Lorenz SINDy demo")
println("samples used      : ", size(Xc, 1))
println("observation noise : ", noise_level * 100, " % of per-axis std")
println("STLSQ threshold λ : ", λ)
println()
println("True system:")
println("  dx/dt = -10·x + 10·y")
println("  dy/dt =  28·x -  y - x·z")
println("  dz/dt =  x·y - (8/3)·z")
println()
println("Recovered equations:")
for j in 1:3
    terms = String[]
    for i in eachindex(labels)
        c = Ξ[i, j]
        c == 0 && continue
        push!(terms, @sprintf("%+0.3f·%s", c, labels[i]))
    end
    println("  $(state_names[j]) = ", isempty(terms) ? "0" : join(terms, "  "))
end

println()
@printf("σ̂ ≈ %0.3f   (true %0.3f)\n", -Ξ[2, 1], σ_true)
@printf("ρ̂ ≈ %0.3f   (true %0.3f)\n",  Ξ[2, 2], ρ_true)
@printf("β̂ ≈ %0.3f   (true %0.3f)\n", -Ξ[4, 3], β_true)
