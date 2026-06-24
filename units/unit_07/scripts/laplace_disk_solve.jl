# Laplace on a disk — train the §7.1 PINN and grade it against the exact
# harmonic u(r,θ) = rⁿ sin(nθ). Produces:
#   output/laplace_disk_error.md      — relative L² and worst-case error (n = 3)
#   figures/laplace_disk_error.png    — |u_PINN − u_exact| over (θ, r), n = 3
# and prints the n = 3 vs n = 6 comparison used by Solution 7.1(b).
# Run from the repo root with the course project active:
#   julia --project=. units/unit_07/scripts/laplace_disk_solve.jl

using NeuralPDE, Lux, ModelingToolkit, Optimization, OptimizationOptimJL
using DomainSets, Statistics, Plots, Printf, Random

R = 1.0;  r_min = 0.05

# Train the disk PINN for boundary data sin(nθ) (exact solution rⁿ sin(nθ))
# and grade it on a 50×100 polar grid. Returns the error field and metrics.
function solve_disk(n::Int)
    Random.seed!(0)
    @parameters r θ
    @variables u(..)
    Dᵣ  = Differential(r);  Dᵣᵣ = Differential(r) ∘ Differential(r)
    Dθθ = Differential(θ) ∘ Differential(θ)
    eq = r^2 * Dᵣᵣ(u(r, θ)) + r * Dᵣ(u(r, θ)) + Dθθ(u(r, θ)) ~ 0
    domains = [r ∈ ClosedInterval(r_min, R), θ ∈ ClosedInterval(0.0, 2π)]
    bcs = [
        u(R, θ)     ~ sin(n * θ),
        u(r_min, θ) ~ r_min^n * sin(n * θ),
        u(r, 0.0)   ~ 0.0,
        u(r, 2π)    ~ 0.0,
    ]
    @named pde_system = PDESystem(eq, bcs, domains, [r, θ], [u(r, θ)])
    chain = Lux.Chain(
        Lux.Dense(2, 32, tanh), Lux.Dense(32, 32, tanh),
        Lux.Dense(32, 32, tanh), Lux.Dense(32, 1),
    )
    discretization = PhysicsInformedNN(chain, GridTraining([0.05, 0.1]))
    prob = discretize(pde_system, discretization)
    res  = Optimization.solve(prob, LBFGS(); maxiters = 3000)

    phi = discretization.phi;  ps = res.u
    rs  = range(r_min, R; length = 50)
    θs  = range(0.0, 2π;  length = 100)
    u_pinn  = [first(phi([rr, tt], ps)) for rr in rs, tt in θs]
    u_exact = [rr^n * sin(n * tt)        for rr in rs, tt in θs]
    rel_L2  = sqrt(sum(abs2, u_pinn .- u_exact) / sum(abs2, u_exact))
    worst   = findmax(abs.(u_pinn .- u_exact))
    rw, θw  = rs[worst[2][1]], θs[worst[2][2]]
    return (; rel_L2, wmax = worst[1], rw, θw, err = abs.(u_pinn .- u_exact), rs, θs)
end

# --- (a) the headline case: sin(3θ) ---
s3 = solve_disk(3)
@printf("n=3: relative L2 = %.3e | worst = %.3e at (r,θ)=(%.3f, %.3f)\n",
        s3.rel_L2, s3.wmax, s3.rw, s3.θw)

open(joinpath(@__DIR__, "..", "output", "laplace_disk_error.md"), "w") do io
    println(io, "| metric | value |")
    println(io, "|---|---|")
    println(io, @sprintf("| relative \$L^2\$ error | %.2e |", s3.rel_L2))
    println(io, @sprintf("| worst pointwise error | %.2e |", s3.wmax))
    println(io, @sprintf("| worst-error location \$(r,\\theta)\$ | (%.2f, %.2f) |", s3.rw, s3.θw))
end

# error field: θ on x, r on y — the error rides the rim (large r) near the 2π seam
heatmap(s3.θs, s3.rs, s3.err;
        xlabel = "θ", ylabel = "r", title = "Pointwise error |u_PINN − u_exact|  (sin 3θ)",
        c = :viridis, size = (700, 360), right_margin = 12Plots.mm,
        colorbar_title = "  |error|")
savefig(joinpath(@__DIR__, "..", "figures", "laplace_disk_error.png"))

# --- (b) harder spectrum: sin(6θ) at the same budget ---
s6 = solve_disk(6)
@printf("n=6: relative L2 = %.3e | worst = %.3e at (r,θ)=(%.3f, %.3f)\n",
        s6.rel_L2, s6.wmax, s6.rw, s6.θw)
@printf("(b) error ratio sin6θ / sin3θ = %.2fx\n", s6.rel_L2 / s3.rel_L2)
println("wrote figures/laplace_disk_error.png and output/laplace_disk_error.md")
