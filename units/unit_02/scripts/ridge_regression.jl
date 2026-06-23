#!/usr/bin/env julia
# Ridge regression (L2-regularized least squares) on a small synthetic problem.
#
# Only the first 3 of 12 features actually drive y; with few training points,
# ordinary least squares (OLS) chases the noise in the 9 irrelevant features
# and generalizes poorly. Adding an L2 penalty shrinks the coefficients and
# lowers the test error — the bias-variance trade-off, dialled by one knob λ.
#
# Uses only stdlib (LinearAlgebra, Random, Statistics) — nothing to install.

using LinearAlgebra, Random, Statistics

Random.seed!(1)

# ── synthetic linear data: y = Xβ + noise, only 3 features matter ─────────
p      = 12
β_true = vcat([3.0, -2.0, 1.5], zeros(p - 3))
gen(m) = (X = randn(m, p); (X, X * β_true .+ 0.5 .* randn(m)))
Xtr, ytr = gen(15)     # small training set (only 3 d.o.f. to spare) → OLS overfits
Xte, yte = gen(2000)   # large test set → clean risk estimate

rmse(β, X, y) = sqrt(mean(abs2, y .- X * β))

# ── ordinary least squares:  minimize ‖y − Xβ‖²   (β̂ = X \ y) ─────────────
β_ols = Xtr \ ytr

# ── ridge (L2):  minimize ‖y − Xβ‖² + λ‖β‖²  ──────────────────────────────
#    closed form  β̂(λ) = (XᵀX + λI)⁻¹ Xᵀy   — OLS normal equations + λI.
#    λ = 0 recovers OLS; the +λI also makes the system well-conditioned.
ridge(X, y, λ) = (X'X + λ * I) \ (X'y)

println("λ        ‖β‖      test RMSE")
for λ in (0.0, 1.0, 10.0, 100.0)
    β = ridge(Xtr, ytr, λ)
    println(rpad(λ, 8), rpad(round(norm(β), digits = 2), 9),
            round(rmse(β, Xte, yte), digits = 2))
end
# (λ = 0 row equals plain OLS: ‖β_ols‖ = $(round(norm(β_ols), digits=2)))
