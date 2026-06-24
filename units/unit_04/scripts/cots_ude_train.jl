#!/usr/bin/env julia
# A COMPLETE (runnable) Universal-Differential-Equation example for Unit 4 §4.3.
#
# Goal: recover a HIDDEN piece of physics from data. We keep the COTS–coral
# skeleton (logistic coral growth + Holling-II grazing) FIXED and known, and let
# a small neural network stand in for the one term the ecology literature is
# honest about not knowing — the density-dependent COTS mortality m(S). We then
# show the network recovers the true m(S) over the range the data actually visits.
#
# Backprop runs straight THROUGH an explicit RK4 integrator with plain Zygote
# (Zygote.Buffer makes the in-place rollout differentiable) — the trajectory-loss
# idea of §4.2 with the solver fully in view, and no package beyond what `@pinn`
# already ships. Output: figures/cots_ude_fit.png
#
#   Pkg deps (all already in the course environment):
#     Lux, Zygote, Optimisers, Optimization, OptimizationOptimJL,
#     ComponentArrays, Plots, Random, Statistics
using Lux, Zygote, Optimisers, Optimization, OptimizationOptimJL,
      ComponentArrays, Random, Statistics, Printf, Plots

rng = Random.MersenneTwister(1)

# ── known physics — held FIXED in both the ground truth and the UDE ─────────
const r_C, K = 0.30, 80.0      # coral growth rate (1/yr), carrying capacity (% cover)
const a, h   = 0.005, 0.10     # COTS attack rate, handling time
const e      = 0.40            # conversion efficiency
graze(C, S) = a*S*C / (1 + a*h*C)

# ── the HIDDEN mortality law the UDE must recover from data alone ────────────
m_true(S) = 0.035 + 0.045*(S/35)^2      # baseline + density-dependent crowding/disease

# explicit RK4 rollout of a 2-state field → 2×(n+1) trajectory.
# Zygote.Buffer makes the in-place writes differentiable.
function rollout(field, u0, dt, n)
    buf = Zygote.Buffer(zeros(eltype(u0), length(u0), n + 1))
    buf[:, 1] = u0
    u = u0
    for k in 1:n
        k1 = field(u);                k2 = field(u .+ 0.5dt .* k1)
        k3 = field(u .+ 0.5dt .* k2); k4 = field(u .+ dt .* k3)
        u  = u .+ (dt/6) .* (k1 .+ 2 .* k2 .+ 2 .* k3 .+ k4)
        buf[:, k+1] = u
    end
    copy(buf)
end

# ── ground-truth data: integrate the true model, sample, add noise ──────────
true_field(u) = (C = u[1]; S = u[2]; g = graze(C, S);
                 [r_C*C*(1 - C/K) - g,  e*g - m_true(S)*S])
u0    = [60.0, 2.0]
T, dt = 60.0, 0.1
n     = Int(T / dt)
idx   = 1:10:(n + 1)                      # observe once a year
full  = rollout(true_field, u0, dt, n)
tfull = collect((0:n) .* dt)
tobs  = tfull[idx]
clean = full[:, idx]
Cv, Sv = clean[1, :], clean[2, :]
data   = clean .* (1 .+ 0.03 .* randn(rng, size(clean)))   # 3% observation noise

# ── the UDE: mortality is a small NN of S; every other term stays known ──────
net = Lux.Chain(Lux.Dense(1 => 16, tanh), Lux.Dense(16 => 16, tanh), Lux.Dense(16 => 1))
p0, stt = Lux.setup(rng, net)
ps = ComponentArray{Float64}(p0)
ps.layer_3.bias .= -2.95                  # softplus(-2.95)≈0.05 → sane initial mortality
mθ(S, p) = log1p(exp(first(net([S], p, stt))[1]))   # softplus keeps mortality ≥ 0

ude_field(u, p) = (C = u[1]; S = u[2]; g = graze(C, S);
                   [r_C*C*(1 - C/K) - g,  e*g - mθ(S, p)*S])
predict(p) = rollout(u -> ude_field(u, p), u0, dt, n)

wC, wS = 1/std(Cv), 1/std(Sv)             # put coral and COTS misfits on equal footing
function loss(p)
    sol = predict(p)[:, idx]
    mean(abs2, (sol[1, :] .- data[1, :]) .* wC) + mean(abs2, (sol[2, :] .- data[2, :]) .* wS)
end

# ── train in two stages — the standard SciML recipe ─────────────────────────
#   1) Adam: robust, cheap first-order steps that get us into the right basin;
#   2) L-BFGS: a quasi-Newton polish that converges sharply once we are close.
# Both differentiate the same trajectory loss with Zygote (through the RK4 solver).
function adam!(ps, lr, iters)
    os = Optimisers.setup(Optimisers.Adam(lr), ps)
    for _ in 1:iters
        os, ps = Optimisers.update(os, ps, Zygote.gradient(loss, ps)[1])
    end
    ps
end
@printf("COTS-coral UDE — recovering the hidden mortality m(S)\n")
@printf("observations        : %d  (3%% noise);  S visits [%.1f, %.1f] per ha\n",
        length(tobs), minimum(Sv), maximum(Sv))
@printf("initial loss        : %.4e\n", loss(ps))

ps = adam!(ps, 0.01, 500)                       # stage 1: Adam
@printf("after 500 Adam steps: %.4e\n", loss(ps))

# stage 2: L-BFGS polish, via the Optimization.jl wrapper (Zygote gradients)
optf = OptimizationFunction((p, _) -> loss(p), Optimization.AutoZygote())
res  = solve(OptimizationProblem(optf, ps), OptimizationOptimJL.LBFGS(); maxiters = 300)
ps   = res.u
@printf("after L-BFGS polish : %.4e\n", loss(ps))

# recovery quality over the range the data actually explored
Smin, Smax = minimum(Sv), maximum(Sv)
Sg  = range(0, Smax * 1.18; length = 160)
ml  = [mθ(s, ps) for s in Sg]
mt  = m_true.(Sg)
vis = (Sg .>= Smin) .& (Sg .<= Smax)
@printf("m(S) recovery error : %.1f%% (mean rel.) over visited range\n",
        100 * mean(abs.(ml[vis] .- mt[vis]) ./ mt[vis]))

# ── figure: (1) trajectory fit, (2) learned vs true mortality law ───────────
fit = predict(ps)
p1 = plot(tfull, fit[1, :]; lw = 2.4, c = :seagreen, label = "UDE coral C",
          xlabel = "time (years)", ylabel = "% cover  /  COTS per ha",
          title = "trajectory fit", framestyle = :box, legend = :topright, gridalpha = 0.25)
plot!(p1, tfull, fit[2, :]; lw = 2.4, c = :firebrick, label = "UDE COTS S")
scatter!(p1, tobs, data[1, :]; ms = 3.2, c = :seagreen,  msw = 0, label = "coral data")
scatter!(p1, tobs, data[2, :]; ms = 3.2, c = :firebrick, msw = 0, label = "COTS data")

p2 = plot(Sg, mt; lw = 3, c = :black, label = "true m(S)  (hidden)",
          xlabel = "COTS density S (per ha)", ylabel = "mortality rate m",
          title = "the UDE recovers the hidden law", framestyle = :box,
          legend = :topleft, gridalpha = 0.25)
plot!(p2, Sg, ml; lw = 2.6, c = :dodgerblue, ls = :dash, label = "learned m(S) — NN closure")
vspan!(p2, [Smin, Smax]; c = :gray, alpha = 0.12, label = "range seen in data")

plt = plot(p1, p2; layout = (1, 2), size = (940, 360),
           bottom_margin = 5Plots.mm, left_margin = 5Plots.mm)
savefig(plt, joinpath(@__DIR__, "..", "figures", "cots_ude_fit.png"))
println("wrote figures/cots_ude_fit.png")
