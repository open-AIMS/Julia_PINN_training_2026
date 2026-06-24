# Naïve PINN on the 1-D linearised SWE (§7.2) — train the failing model and
# extract the two diagnostics shown in @fig-naive-swe, from the ACTUAL run:
#   1. η(x, t = T_f/2) from the trained PINN vs the exact d'Alembert reference
#   2. PDE residual |r|² binned in t (the causality-violation signature)
# Writes plain-CSV data the figure script consumes, plus a summary line.
#   julia --project=. units/unit_07/scripts/naive_swe_failure_solve.jl

using NeuralPDE, Lux, ModelingToolkit, Optimization, OptimizationOptimJL
using DomainSets, ForwardDiff, Statistics, Printf, Random, DelimitedFiles

Random.seed!(0)

@parameters x t
@variables η(..) u_vel(..)

Dₜ = Differential(t);  Dₓ = Differential(x)
g = 9.81;  H = 10.0                       # wave speed c = √(gH) ≈ 9.9 m/s
c = sqrt(g * H)
Tf = 0.05

eqs = [
    Dₜ(η(x, t))     + H * Dₓ(u_vel(x, t)) ~ 0,
    Dₜ(u_vel(x, t)) + g * Dₓ(η(x, t))     ~ 0,
]
domains = [x ∈ ClosedInterval(0.0, 2.0), t ∈ ClosedInterval(0.0, Tf)]
η0(x) = exp(-100 * (x - 1.0)^2)
bcs = [
    η(x, 0)       ~ exp(-100 * (x - 1.0)^2),
    u_vel(x, 0)   ~ 0.0,
    u_vel(0.0, t) ~ 0.0,
    u_vel(2.0, t) ~ 0.0,
]
@named pde_system = PDESystem(eqs, bcs, domains, [x, t], [η(x, t), u_vel(x, t)])

chain_η = Lux.Chain(Lux.Dense(2, 32, tanh), Lux.Dense(32, 32, tanh),
                    Lux.Dense(32, 32, tanh), Lux.Dense(32, 1))
chain_u = Lux.Chain(Lux.Dense(2, 32, tanh), Lux.Dense(32, 32, tanh),
                    Lux.Dense(32, 32, tanh), Lux.Dense(32, 1))

discretization = PhysicsInformedNN([chain_η, chain_u], GridTraining([0.02, 0.001]))
prob = discretize(pde_system, discretization)
println("training naïve SWE PINN (LBFGS, 3000 iters)…"); flush(stdout)
res = Optimization.solve(prob, LBFGS(); maxiters = 3000)
@printf("final objective = %.4e\n", res.objective)

# Save the trained params so eval/plotting can be re-run without retraining.
using Serialization
serialize(joinpath(@__DIR__, "..", "output", "swe_res_u.jls"), res.u)
println("param container keys: ", keys(res.u))

phi = discretization.phi
# This NeuralPDE version does NOT auto-slice a multi-network param vector, so
# pass each network its own slice (res.u.depvar.<name>) rather than the full res.u.
pη = res.u.depvar.η
pu = res.u.depvar.u_vel
ηf(x, t) = first(phi[1]([x, t], pη))
uf(x, t) = first(phi[2]([x, t], pu))

# ---- diagnostic 1: half-time snapshot vs d'Alembert ----
tsnap = Tf / 2                                   # 0.025
xs = range(0.0, 2.0; length = 400)
η_pinn  = [ηf(xx, tsnap) for xx in xs]
η_exact = [0.5 * η0(xx - c * tsnap) + 0.5 * η0(xx + c * tsnap) for xx in xs]
open(joinpath(@__DIR__, "..", "output", "naive_swe_snapshot.csv"), "w") do io
    writedlm(io, [["x" "eta_pinn" "eta_exact"]; hcat(collect(xs), η_pinn, η_exact)], ',')
end

# ---- diagnostic 2: residual |r|² binned in t ----
r1(x, t) = ForwardDiff.derivative(τ -> ηf(x, τ), t) +
           H * ForwardDiff.derivative(ξ -> uf(ξ, t), x)
r2(x, t) = ForwardDiff.derivative(τ -> uf(x, τ), t) +
           g * ForwardDiff.derivative(ξ -> ηf(ξ, t), x)
xg = range(0.0, 2.0; length = 120)
tbins = range(0.0, Tf; length = 11)
tc   = [(tbins[i] + tbins[i+1]) / 2 for i in 1:10]
means = [mean(abs2(r1(xx, tt)) + abs2(r2(xx, tt))
              for xx in xg, tt in range(tbins[i], tbins[i+1]; length = 12)) for i in 1:10]
open(joinpath(@__DIR__, "..", "output", "naive_swe_residual.csv"), "w") do io
    writedlm(io, [["t_center" "mean_residual"]; hcat(collect(tc), means)], ',')
end

# ---- summary for sanity / prose ----
relL2 = sqrt(sum(abs2, η_pinn .- η_exact) / sum(abs2, η_exact))
@printf("snapshot relative L² error (η vs d'Alembert) = %.3e\n", relL2)
@printf("PINN η peak = %.3f at x = %.3f  (d'Alembert peaks = 0.5 at x = %.3f, %.3f)\n",
        maximum(η_pinn), xs[argmax(η_pinn)], 1 - c*tsnap, 1 + c*tsnap)
@printf("residual bins (early→late): %s\n", join((@sprintf("%.2e", m) for m in means), "  "))
@printf("late/early residual ratio = %.2f  (>1 ⇒ causality violation)\n", means[end] / means[1])
println("wrote output/naive_swe_snapshot.csv and output/naive_swe_residual.csv")
