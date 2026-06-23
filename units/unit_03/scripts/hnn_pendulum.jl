#!/usr/bin/env julia
# Hamiltonian Neural Network (HNN) on the simple pendulum — Greydanus et al. (2019),
# the worked example for unit_03.qmd §3.3.
#
# We parametrise the Hamiltonian H_θ(q,p) with a small Lux MLP and DERIVE the
# dynamics by autodiff — q̇ = ∂H/∂p, ṗ = -∂H/∂q — so the flow is symplectic and
# energy is conserved by construction. For contrast we train a vanilla Lux MLP
# that predicts the field (q̇, ṗ) directly from the SAME noisy data. On a long
# rollout the HNN's energy barely drifts while the vanilla MLP's wanders off.
#
# Autodiff split — the §2.7 forward-over-reverse pattern AND its one caveat:
#   • the HNN FIELD is the input-gradient of H_θ → ForwardDiff (cheap for 2 inputs);
#   • the VANILLA model trains with Zygote (reverse mode over the parameters);
#   • the HNN PARAMETER gradient, however, must ALSO use ForwardDiff: Zygote
#     silently returns `nothing` when asked to differentiate THROUGH the inner
#     ForwardDiff input-gradient (the exact nesting caveat called out in §5.3 /
#     Solution 2.7 — the central reason real PINNs lean on NeuralPDE.jl). The net
#     is tiny, so the forward-over-forward cost is negligible.
#
# Run via ./build.sh execute 3 (writes output to ../output/hnn_pendulum.md).

using Lux, Random, Zygote, ForwardDiff, ComponentArrays, Optimisers,
      Statistics, OrdinaryDiffEq, Printf

# ── true pendulum ───────────────────────────────────────────────────────
Htrue(q, p)     = 0.5 * p^2 + (1 - cos(q))
truefield(q, p) = (p, -sin(q))                  # (q̇, ṗ) = (∂_p H, -∂_q H)

# ── data: noisy field samples over the region the test orbit visits ─────────
rng = Random.MersenneTwister(0)
N   = 120
U   = vcat((5.0 .* rand(rng, N) .- 2.5)',        # q ∈ [-2.5, 2.5]
           (4.0 .* rand(rng, N) .- 2.0)')        # p ∈ [-2.0, 2.0]
F   = reduce(hcat, [collect(truefield(U[1, i], U[2, i])) for i in 1:N])
F .+= 0.05 .* randn(rng, size(F))                # 5% observation noise

# ── two small Lux MLPs, with Float64 parameter vectors ──────────────────────
hnet = Lux.Chain(Lux.Dense(2 => 16, tanh), Lux.Dense(16 => 16, tanh), Lux.Dense(16 => 1))
vnet = Lux.Chain(Lux.Dense(2 => 16, tanh), Lux.Dense(16 => 16, tanh), Lux.Dense(16 => 2))
psh, sth = Lux.setup(rng, hnet); psh = ComponentArray{Float64}(psh)
psv, stv = Lux.setup(rng, vnet); psv = ComponentArray{Float64}(psv)

# HNN: H_θ(q,p) is a scalar; the field is its INPUT-gradient (ForwardDiff).
Hθ(u, ps)        = first(hnet(u, ps, sth))[1]
hnn_field(u, ps) = (g = ForwardDiff.gradient(z -> Hθ(z, ps), u); (g[2], -g[1]))
# Vanilla baseline: (q,p) ↦ (q̇, ṗ) directly — no structure imposed.
van_field(u, ps) = vnet(u, ps, stv)[1]

# ── losses (mean squared field error) ───────────────────────────────────
function loss_hnn(ps)
    s = zero(eltype(ps))
    for i in 1:N
        q̇, ṗ = hnn_field(U[:, i], ps)
        s += (q̇ - F[1, i])^2 + (ṗ - F[2, i])^2
    end
    s / N
end
loss_van(ps) = mean(sum(abs2, van_field(U[:, i], ps) .- F[:, i]) for i in 1:N)

# ── train: Optimisers.Adam; the HNN gradient via ForwardDiff (forward-over-
#    forward), the vanilla gradient via Zygote (plain reverse mode) ──────────
function train(loss, ps, grad; iters = 1500, lr = 5e-3)
    opt = Optimisers.setup(Optimisers.Adam(lr), ps)
    for _ in 1:iters
        opt, ps = Optimisers.update(opt, ps, grad(loss, ps))
    end
    ps
end
psh = train(loss_hnn, psh, (l, p) -> ForwardDiff.gradient(l, p))
psv = train(loss_van, psv, (l, p) -> Zygote.gradient(l, p)[1])

# ── long rollout from a held-out initial condition; measure energy drift ────
u0, tspan = [2.0, 0.0], (0.0, 50.0)
H0 = Htrue(u0...)
rollout(rhs) = solve(ODEProblem((u, _, _) -> rhs(u), u0, tspan), Tsit5();
                     reltol = 1e-8, abstol = 1e-8, saveat = 0.5)
solh = rollout(u -> collect(hnn_field(u, psh)))
solv = rollout(u -> van_field(u, psv))
drifth = [Htrue(u...) - H0 for u in solh.u]
driftv = [Htrue(u...) - H0 for u in solv.u]

@printf("HNN pendulum demo\n")
@printf("training points     : %d  (5%% field noise)\n", N)
@printf("rollout horizon     : t ∈ [0, %d]\n", Int(tspan[2]))
@printf("HNN     max |ΔH|    : %.4f\n", maximum(abs, drifth))
@printf("vanilla max |ΔH|    : %.4f\n", maximum(abs, driftv))
@printf("vanilla drifts %.0f× more than the energy-conserving HNN\n",
        maximum(abs, driftv) / maximum(abs, drifth))

# ── figure: energy drift vs time ────────────────────────────────────────
using Plots
plot(solh.t, drifth; lw = 2, label = "HNN (∂H/∂p, -∂H/∂q)",
     xlabel = "time", ylabel = "H(t) - H(0)", title = "Pendulum energy drift",
     legend = :topleft)
plot!(solv.t, driftv; lw = 2, ls = :dash, label = "vanilla MLP field")
savefig(joinpath(@__DIR__, "..", "figures", "hnn_energy_drift.png"))
