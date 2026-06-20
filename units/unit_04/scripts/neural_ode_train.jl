#!/usr/bin/env julia
# A COMPLETE (runnable) Neural-ODE training example for Unit 4 §4.2.
# Fit a neural vector field f_θ(u) to data sampled from a damped spiral, training
# through the ODE solver with adjoint sensitivities. Output: figures/neural_ode_fit.png
#
#   Pkg.add(["Lux","OrdinaryDiffEq","SciMLSensitivity","Zygote","Optimisers",
#            "ComponentArrays","Plots"])
using Lux, OrdinaryDiffEq, SciMLSensitivity, Zygote, Optimisers,
      ComponentArrays, Random, Plots

rng = Random.MersenneTwister(0)

# --- ground-truth system: a damped linear spiral ---------------------------
A_true = Float32[-0.1 2.0; -2.0 -0.1]
truth(u, p, t) = A_true * u
u0    = Float32[2.0, 0.0]
tspan = (0.0f0, 6.0f0)
ts    = range(tspan[1], tspan[2]; length = 40)
data  = Array(solve(ODEProblem(truth, u0, tspan), Tsit5(); saveat = ts))  # 2×40

# --- the Neural ODE: a small MLP IS the vector field -----------------------
fθ      = Lux.Chain(Lux.Dense(2 => 16, tanh), Lux.Dense(16 => 2))
ps0, st = Lux.setup(rng, fθ)
ps      = ComponentArray(ps0)                 # flat params the solver can carry

rhs(u, p, t) = first(fθ(u, p, st))
predict(p) = Array(solve(ODEProblem(rhs, u0, tspan, p), Tsit5();
                         saveat = ts, sensealg = InterpolatingAdjoint(autojacvec = ZygoteVJP())))
loss(p) = sum(abs2, predict(p) .- data)

# --- train: Adam on θ, gradients flow through the solver -------------------
opt_state = Optimisers.setup(Optimisers.Adam(0.05f0), ps)
losses = Float32[]
for epoch in 1:300
    l, back = Zygote.pullback(loss, ps)
    gs = back(one(l))[1]
    opt_state, ps = Optimisers.update(opt_state, ps, gs)
    push!(losses, l)
    epoch % 50 == 0 && println("epoch $epoch   loss = $(round(l; digits = 4))")
end

# --- figure: data vs learned trajectory + loss curve -----------------------
pred = predict(ps)
p1 = plot(data[1, :], data[2, :]; lw = 3, c = :black, label = "true trajectory",
          xlabel = "u₁", ylabel = "u₂", title = "Neural ODE fit (phase plane)",
          framestyle = :box, legend = :topright, gridalpha = 0.25)
plot!(p1, pred[1, :], pred[2, :]; lw = 2.4, c = :crimson, ls = :dash, label = "learned f_θ")
scatter!(p1, [u0[1]], [u0[2]], c = :black, ms = 5, msw = 0, label = "")

p2 = plot(losses; lw = 2.2, c = :steelblue, yaxis = :log, legend = false,
          xlabel = "epoch", ylabel = "loss", title = "training loss",
          framestyle = :box, gridalpha = 0.25)

plt = plot(p1, p2; layout = (1, 2), size = (940, 380),
           bottom_margin = 5Plots.mm, left_margin = 5Plots.mm)
savefig(plt, joinpath(@__DIR__, "..", "figures", "neural_ode_fit.png"))
println("final loss = $(round(losses[end]; digits = 4));  wrote figures/neural_ode_fit.png")
