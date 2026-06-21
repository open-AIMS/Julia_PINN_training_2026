# ===========================================================================
# Unit 7 — why a GPU changes what PINNs you can afford: collocation throughput.
#
# Section 7.2/7.3 make the case that serious PINNs need *many* collocation points
# — causal training, adaptive resampling, and high-frequency problems all pile on
# residual points, and each point is a forward+backward pass through the network.
# The practical question is then: how many collocation points can you evaluate per
# second? That is precisely a batched-matmul throughput question, and it is where
# the GPU rewrites the budget.
#
# This measures the cost of ONE training step of a Poisson PINN
#   residual:  Δu_θ(x,y) − f(x,y),    f chosen so u* = sin(πx)sin(πy)
# where the Laplacian is taken by a cheap finite-difference stencil in the input
# (five batched forward passes), and the parameter gradient by reverse-mode AD —
# the same Lux + Zygote path used throughout the course. We sweep the collocation
# batch size N and report points/second on CPU vs GPU.
#
# Run on the GPU hub (the @pinn env has Lux + LuxCUDA + CUDA):
#   julia --project=@pinn units/unit_07/scripts/pinn_throughput_gpu.jl
# Nothing here runs during `quarto render` — the .qmd shows it `eval: false`.
# ===========================================================================

using Lux, LuxCUDA, CUDA, Optimisers, Zygote, Random, Printf, Statistics

make_model() = Chain(Dense(2 => 128, tanh), Dense(128 => 128, tanh),
                     Dense(128 => 128, tanh), Dense(128 => 1))

const H = 1f-3                          # FD step for the input-space Laplacian
fsource(x, y) = -2f0 * (π^2) .* sinpi.(x) .* sinpi.(y)   # so u* = sin(πx)sin(πy)

# One full training step over a batch of N collocation points; returns seconds.
function step_time(N, dev; nrep = 3)
    model = make_model()
    ps, st = Lux.setup(Xoshiro(0), model)
    ps = ps |> dev; st = st |> dev
    opt = Optimisers.setup(Adam(1f-3), ps)
    P = (rand(Float32, 2, N)) |> dev          # collocation points in (0,1)²
    ex = reshape(P[1, :], 1, N); ey = reshape(P[2, :], 1, N)
    f  = fsource(ex, ey)

    u(p, X) = first(model(X, p, st))
    function lap_residual(p)
        c = u(p, P)
        xp = u(p, vcat(ex .+ H, ey)); xm = u(p, vcat(ex .- H, ey))
        yp = u(p, vcat(ex, ey .+ H)); ym = u(p, vcat(ex, ey .- H))
        Δu = (xp .+ xm .+ yp .+ ym .- 4f0 .* c) ./ H^2
        return mean(abs2, Δu .- f)
    end

    g = Zygote.gradient(lap_residual, ps)[1]; opt, ps = Optimisers.update(opt, ps, g)  # warmup
    dev === identity || CUDA.synchronize()
    t0 = time()
    for _ in 1:nrep
        g = Zygote.gradient(lap_residual, ps)[1]
        opt, ps = Optimisers.update(opt, ps, g)
    end
    dev === identity || CUDA.synchronize()
    return (time() - t0) / nrep
end

println("="^66)
println("Unit 7 — Poisson-PINN training-step throughput: CPU vs GPU")
println("="^66)
have_gpu = CUDA.functional()
@printf("GPU available: %s%s\n", have_gpu, have_gpu ? "  ($(CUDA.name(CUDA.device())))" : "")
println("net 2-128-128-128-1; one step = 5 batched forward passes + reverse-mode AD\n")

@printf("%-12s %12s %12s %10s %14s %14s\n",
        "N (colloc)", "CPU (ms)", "GPU (ms)", "speedup", "CPU pts/s", "GPU pts/s")
println("-"^78)
Ns = [1_000, 10_000, 100_000, 500_000]
cpu_tp = Float64[]; gpu_tp = Float64[]
for N in Ns
    tc = step_time(N, identity)
    tg = have_gpu ? step_time(N, gpu_device()) : NaN
    spd = have_gpu ? tc / tg : NaN
    push!(cpu_tp, N / tc); have_gpu && push!(gpu_tp, N / tg)
    @printf("%-12d %12.1f %12.1f %9.1fx %14.2e %14.2e\n",
            N, 1e3*tc, 1e3*tg, spd, N/tc, have_gpu ? N/tg : NaN)
end

if have_gpu
    @printf("\nAt N=%d the GPU evaluates %.1fx more collocation points per second;\n", Ns[end], gpu_tp[end] / cpu_tp[end])
    @printf("the GPU throughput peaks at N=%d (%.1fx the CPU there).\n", Ns[argmax(gpu_tp)], maximum(gpu_tp) / cpu_tp[argmax(gpu_tp)])
    try
        using CairoMakie
        f = Figure(size = (560, 400))
        ax = Axis(f[1,1], title = "PINN collocation throughput (higher is better)",
                  xlabel = "collocation points N", ylabel = "points / second",
                  xscale = log10, yscale = log10)
        lines!(ax, Ns, cpu_tp, label = "CPU", linewidth = 3)
        scatter!(ax, Ns, cpu_tp, markersize = 9)
        lines!(ax, Ns, gpu_tp, label = "GPU", linewidth = 3)
        scatter!(ax, Ns, gpu_tp, markersize = 9)
        axislegend(ax, position = :rb)
        figdir = get(ENV, "GPU_FIG_DIR", joinpath(@__DIR__, "..", "figures"))
        isdir(figdir) || mkpath(figdir)
        save(joinpath(figdir, "pinn_throughput_gpu.png"), f)
        println("wrote figures/pinn_throughput_gpu.png")
    catch e
        println("(figure skipped: ", e, ")")
    end
end
