#!/usr/bin/env julia
# Ordinary least squares (OLS) line fit, the Julia way: the `\` operator.
#
# For a square invertible A, `A \ b` solves A x = b. For a TALL matrix X
# (more data rows than parameters), `X \ y` instead returns the LEAST-SQUARES
# solution — the β minimizing ‖y - Xβ‖² — computed stably by QR. Stacking a
# column of ones for the intercept turns line-fitting into a single call.
#
# Uses only stdlib — nothing to install.

# 5 noisy points, roughly on the line  y = 2 + 3x
x = [0.0, 1.0, 2.0, 3.0, 4.0]
y = [2.1, 5.2, 7.8, 11.1, 13.9]

# design matrix: column of 1s (intercept) next to x (slope)
X = [ones(length(x)) x]

# least-squares fit — `\` minimizes ‖y - Xβ‖² for this tall X
β = X \ y

println("intercept = ", round(β[1], digits = 3))
println("slope     = ", round(β[2], digits = 3))
