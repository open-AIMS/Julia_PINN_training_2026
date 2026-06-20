# Hamiltonian Neural Network (HNN) on the simple pendulum — Greydanus et al. (2019),
# the worked example for unit_03.qmd §3.3.
#
# We parametrise the Hamiltonian H_θ(q, p) with a small MLP and DERIVE the dynamics
# by autodiff — q̇ = ∂H/∂p, ṗ = -∂H/∂q — so the flow is symplectic and energy is
# conserved by construction. For contrast we also train a vanilla MLP that predicts
# the field (q̇, ṗ) directly from the SAME noisy data. On a long rollout the HNN's
# energy barely drifts while the vanilla MLP's wanders off.
#
# This is the §2.6 nested-autodiff pattern in the wild: the field uses ForwardDiff
# for the INPUT gradient of H_θ, and training differentiates the loss w.r.t. the
# PARAMETERS. We keep the net tiny and use ForwardDiff for the parameter gradient
# too, so the nesting is transparent and needs no extra packages.
#
# Run via ./build.sh execute 3 (writes output to ../output/hnn_pendulum.md).

using ForwardDiff, Random, Statistics, OrdinaryDiffEq, Printf

# ── true pendulum ───────────────────────────────────────────────────────
Htrue(q, p) = 0.5 * p^2 + (1 - cos(q))
truefield(q, p) = (p, -sin(q))                 # (q̇, ṗ) = (∂_p H, -∂_q H)

# ── tiny MLP with parameters as a flat vector (so ForwardDiff nests cleanly) ──
struct MLP; sizes; end
nparams(m::MLP) = sum(m.sizes[i] * m.sizes[i+1] + m.sizes[i+1] for i in 1:length(m.sizes)-1)
function (m::MLP)(x, θ)
    k = 1; nl = length(m.sizes) - 1
    for l in 1:nl
        din, dout = m.sizes[l], m.sizes[l+1]
        W = reshape(θ[k:k+din*dout-1], dout, din); k += din * dout
        b = θ[k:k+dout-1];                          k += dout
        x = W * x .+ b
        l < nl && (x = tanh.(x))                    # tanh hidden, linear output
    end
    return x
end

# ── the two models ──────────────────────────────────────────────────────
# HNN: H_θ(q,p) is a scalar; the field comes from its input-gradient via ForwardDiff.
hnet = MLP((2, 16, 16, 1))
hnn_field(u, θ) = (g = ForwardDiff.gradient(z -> hnet(z, θ)[1], u); (g[2], -g[1]))

# Vanilla baseline: an MLP that maps (q,p) straight to (q̇,ṗ) — no structure imposed.
vnet = MLP((2, 16, 16, 2))

# ── data: noisy field samples over the region the test orbit visits ─────────
rng = MersenneTwister(0)
N   = 120
U   = vcat((5.0 .* rand(rng, N) .- 2.5)',        # q ∈ [-2.5, 2.5]
           (4.0 .* rand(rng, N) .- 2.0)')        # p ∈ [-2.0, 2.0]
F   = reduce(hcat, [collect(truefield(U[1, i], U[2, i])) for i in 1:N])
F .+= 0.05 .* randn(rng, size(F))                # 5% observation noise

# ── losses (mean squared field error) ───────────────────────────────────
function loss_hnn(θ)
    s = zero(eltype(θ))
    for i in 1:N
        q̇, ṗ = hnn_field(U[:, i], θ)
        s += (q̇ - F[1, i])^2 + (ṗ - F[2, i])^2
    end
    s / N
end
function loss_vanilla(θ)
    s = zero(eltype(θ))
    for i in 1:N
        pred = vnet(U[:, i], θ)
        s += (pred[1] - F[1, i])^2 + (pred[2] - F[2, i])^2
    end
    s / N
end

# ── hand-rolled Adam, gradient by ForwardDiff (forward-over-forward) ────────
function train(loss, θ; iters = 1500, lr = 5e-3)
    m = zero(θ); v = zero(θ); β1 = 0.9; β2 = 0.999; ϵ = 1e-8
    for t in 1:iters
        g = ForwardDiff.gradient(loss, θ)
        @. m = β1 * m + (1 - β1) * g
        @. v = β2 * v + (1 - β2) * g^2
        @. θ -= lr * (m / (1 - β1^t)) / (sqrt(v / (1 - β2^t)) + ϵ)
    end
    return θ
end

θh = train(loss_hnn,     0.1 .* randn(rng, nparams(hnet)))
θv = train(loss_vanilla, 0.1 .* randn(rng, nparams(vnet)))

# ── long rollout from a held-out initial condition; measure energy drift ────
u0    = [2.0, 0.0]
tspan = (0.0, 50.0)
H0    = Htrue(u0...)
rollout(rhs) = solve(ODEProblem((u, _, _) -> rhs(u), u0, tspan), Tsit5();
                     reltol = 1e-8, abstol = 1e-8, saveat = 0.5)

solh = rollout(u -> collect(hnn_field(u, θh)))
solv = rollout(u -> vnet(u, θv))
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
try
    using Plots
    plot(solh.t, drifth; lw = 2, label = "HNN (∂H/∂p, -∂H/∂q)",
         xlabel = "time", ylabel = "H(t) - H(0)", title = "Pendulum energy drift",
         legend = :topleft)
    plot!(solv.t, driftv; lw = 2, ls = :dash, label = "vanilla MLP field")
    mkpath(joinpath(@__DIR__, "..", "figures"))
    savefig(joinpath(@__DIR__, "..", "figures", "hnn_energy_drift.png"))
catch e
    @warn "plot step skipped (non-fatal)" exception = e
end
