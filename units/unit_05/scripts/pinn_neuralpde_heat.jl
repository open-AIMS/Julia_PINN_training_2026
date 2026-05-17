# 1D heat equation PINN with NeuralPDE.jl.
#
#   ∂t u = α ∂xx u       on (x,t) ∈ [0,1] × [0,1]
#   u(x, 0) = exp(-200 (x - 0.5)^2)
#   u(0, t) = u(1, t) = 0
#
# Compares the trained network to a quick second-order finite-
# difference reference solve.
#
# Run via ./build.sh execute 5; writes output/pinn_neuralpde_heat.md
# plus output/pinn_neuralpde_heat.png.

using NeuralPDE, Lux, ModelingToolkit
using Optimization, OptimizationOptimJL
using OrdinaryDiffEq, OrdinaryDiffEqBDF
using DomainSets: ClosedInterval
using LinearAlgebra, Plots

@parameters x t
@variables u(..)
Dt  = Differential(t)
Dxx = Differential(x)^2

α   = 0.01
eq  = Dt(u(x, t)) ~ α * Dxx(u(x, t))

ic_bump(x_) = exp(-200 * (x_ - 0.5)^2)
bcs = [
    u(x, 0.0) ~ ic_bump(x),
    u(0.0, t) ~ 0.0,
    u(1.0, t) ~ 0.0,
]
domains = [x ∈ ClosedInterval(0.0, 1.0), t ∈ ClosedInterval(0.0, 1.0)]

@named pde_system = PDESystem(eq, bcs, domains, [x, t], [u(x, t)])

chain = Lux.Chain(
    Lux.Dense(2 => 32, tanh),
    Lux.Dense(32 => 32, tanh),
    Lux.Dense(32 => 1),
)
disc = PhysicsInformedNN(chain, GridTraining([0.02, 0.02]))
prob = discretize(pde_system, disc)

@info "solving heat equation PINN..."
sol = solve(prob, LBFGS(); maxiters = 1500)
println("retcode    : $(sol.retcode)")
println("final loss : $(round(sol.objective; sigdigits = 4))")

# ── finite-difference reference (centred diffs, BDF in time) ───────────
Nx = 101
xg = range(0.0, 1.0; length = Nx)
Δx = step(xg)
u0 = ic_bump.(xg)
u0[1] = 0.0; u0[end] = 0.0          # enforce BCs at t=0

function rhs!(du, u, p, t)
    du[1] = 0.0
    du[end] = 0.0
    @inbounds for i in 2:length(u)-1
        du[i] = α * (u[i+1] - 2u[i] + u[i-1]) / Δx^2
    end
end
ref_prob = ODEProblem(rhs!, u0, (0.0, 1.0))
ref_sol  = solve(ref_prob, FBDF(); saveat = [0.0, 0.3, 0.7, 1.0])

# ── evaluate PINN at the same grid+times ───────────────────────────────
phi    = disc.phi
params = sol.u
pinn_at(t) = [phi([xi, t], params)[1] for xi in xg]

max_err = let m = 0.0
    for (i, t) in enumerate(ref_sol.t)
        ref = ref_sol.u[i]
        pin = pinn_at(t)
        m = max(m, maximum(abs, pin .- ref))
    end
    m
end
println("max |PINN-FD|: $(round(max_err; sigdigits = 3))")

# ── plot ───────────────────────────────────────────────────────────────
plt = plot(xlabel = "x", ylabel = "u(x, t)",
           title  = "1D diffusion: PINN vs. FD reference", legend = :topright)
colors = [:black, :red, :blue, :green]
for (i, t) in enumerate(ref_sol.t)
    plot!(plt, xg, ref_sol.u[i]; lw = 2,  ls = :solid, color = colors[i],
          label = "FD t=$(round(t; digits=2))")
    plot!(plt, xg, pinn_at(t);    lw = 2,  ls = :dash,  color = colors[i],
          label = "PINN t=$(round(t; digits=2))")
end
outpath = joinpath(@__DIR__, "..", "output", "pinn_neuralpde_heat.png")
savefig(plt, outpath)
println("saved $outpath")
