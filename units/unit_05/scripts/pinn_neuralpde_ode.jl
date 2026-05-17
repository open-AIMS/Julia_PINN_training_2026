# A PINN for ẋ = -x, x(0) = 1 using NeuralPDE.jl.
#
# Same problem as the hand-rolled version in unit_05.qmd §5.2, but
# written declaratively against a ModelingToolkit PDESystem.
#
# Run via ./build.sh execute 5 (writes output to ../output/pinn_neuralpde_ode.md).

using NeuralPDE, Lux, ModelingToolkit
using Optimization, OptimizationOptimJL
using DomainSets: ClosedInterval
using Random

@parameters t
@variables x(..)
Dt = Differential(t)

eq      = Dt(x(t)) ~ -x(t)
bcs     = [x(0.0) ~ 1.0]
domains = [t ∈ ClosedInterval(0.0, 4.0)]

@named pde_system = PDESystem(eq, bcs, domains, [t], [x(t)])

chain = Lux.Chain(Lux.Dense(1 => 16, tanh), Lux.Dense(16 => 1))
disc  = PhysicsInformedNN(chain, GridTraining(0.05))

prob = discretize(pde_system, disc)
@info "solving with LBFGS..."
sol  = solve(prob, LBFGS(); maxiters = 200)

# Pull the trained function out of the solution and score it against
# the exact e^{-t} at a held-out grid.
phi    = disc.phi
params = sol.u
tgrid  = collect(range(0.0, 4.0; length = 200))
learnt = [phi([t], params)[1] for t in tgrid]
exact  = exp.(-tgrid)
err    = maximum(abs, learnt .- exact)

println("retcode         : $(sol.retcode)")
println("final loss      : $(round(sol.objective; sigdigits = 4))")
println("max |PINN-exact|: $(round(err; sigdigits = 3))")
