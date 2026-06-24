# Eval-only companion to naive_swe_failure_solve.jl — loads the SERIALIZED
# trained params (output/swe_res_u.jls), rebuilds the discretization WITHOUT
# retraining, and computes time-resolved diagnostics:
#   - residual |r|² vs t (already in the solve script)
#   - L² error vs d'Alembert vs t   (does the solution degrade at late times?)
# Lets us choose an honest Panel B without paying for another 3000-iter train.
#   julia --project=. units/unit_07/scripts/swe_eval_only.jl

using NeuralPDE, Lux, ModelingToolkit, Optimization, OptimizationOptimJL
using DomainSets, ForwardDiff, Statistics, Printf, Serialization, DelimitedFiles

@parameters x t
@variables η(..) u_vel(..)
Dₜ = Differential(t);  Dₓ = Differential(x)
g = 9.81;  H = 10.0;  c = sqrt(g * H);  Tf = 0.05
eqs = [Dₜ(η(x, t)) + H * Dₓ(u_vel(x, t)) ~ 0,
       Dₜ(u_vel(x, t)) + g * Dₓ(η(x, t)) ~ 0]
domains = [x ∈ ClosedInterval(0.0, 2.0), t ∈ ClosedInterval(0.0, Tf)]
η0(x) = exp(-100 * (x - 1.0)^2)
bcs = [η(x, 0) ~ exp(-100 * (x - 1.0)^2), u_vel(x, 0) ~ 0.0,
       u_vel(0.0, t) ~ 0.0, u_vel(2.0, t) ~ 0.0]
@named pde_system = PDESystem(eqs, bcs, domains, [x, t], [η(x, t), u_vel(x, t)])
chain_η = Lux.Chain(Lux.Dense(2,32,tanh), Lux.Dense(32,32,tanh), Lux.Dense(32,32,tanh), Lux.Dense(32,1))
chain_u = Lux.Chain(Lux.Dense(2,32,tanh), Lux.Dense(32,32,tanh), Lux.Dense(32,32,tanh), Lux.Dense(32,1))
discretization = PhysicsInformedNN([chain_η, chain_u], GridTraining([0.02, 0.001]))
discretize(pde_system, discretization)             # build phi (no training)

resu = deserialize(joinpath(@__DIR__, "..", "output", "swe_res_u.jls"))
phi = discretization.phi
ηf(x, t) = first(phi[1]([x, t], resu.depvar.η))
uf(x, t) = first(phi[2]([x, t], resu.depvar.u_vel))
ηexact(x, t) = 0.5 * η0(x - c * t) + 0.5 * η0(x + c * t)

xg = range(0.0, 2.0; length = 200)
tcs = range(0.0, Tf; length = 11)
tc  = [(tcs[i] + tcs[i+1]) / 2 for i in 1:10]

# L² error of η vs d'Alembert, per time bin (normalised by the exact field there)
errt = map(1:10) do i
    ts = range(tcs[i], tcs[i+1]; length = 6)
    num = sum(abs2(ηf(xx, tt) - ηexact(xx, tt)) for xx in xg, tt in ts)
    den = sum(abs2(ηexact(xx, tt)) for xx in xg, tt in ts)
    sqrt(num / den)
end

open(joinpath(@__DIR__, "..", "output", "naive_swe_error_vs_t.csv"), "w") do io
    writedlm(io, [["t_center" "rel_L2_error"]; hcat(collect(tc), errt)], ',')
end
@printf("rel L² error vs t (early→late): %s\n", join((@sprintf("%.2f", e) for e in errt), "  "))
@printf("late/early error ratio = %.2f  (>1 ⇒ solution degrades with time)\n", errt[end] / errt[1])
println("wrote output/naive_swe_error_vs_t.csv")
