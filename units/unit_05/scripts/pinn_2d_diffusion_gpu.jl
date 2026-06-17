# ===========================================================================
# Unit 5 — a 2-D diffusion PINN, trained on the GPU.
#
# Section 5.3 builds a PINN "from first principles" — a Lux network, a residual
# loss assembled by hand, gradient descent — and §5.5 solves the 1-D heat
# equation. This extends that hand-built recipe to TWO space dimensions,
#   ∂t u = α (∂xx u + ∂yy u)   on (x,y) ∈ [0,1]², t ∈ [0, 0.2],
#   u(x,y,0) = sin(πx) sin(πy),   u = 0 on the spatial boundary,
# and trains it on the GPU. Going from u(x,t) to u(x,y,t) multiplies the
# collocation grid, which is exactly when a PINN starts to want a GPU: every
# collocation point is an independent network evaluation, so a large batch fills
# the card.
#
# We keep the §5.3 spirit — no NeuralPDE, a residual built by hand — and use the
# finite-difference-in-input trick from §5.2/§5.3 for the derivatives, so the
# parameter gradient is a single reverse-mode pass that runs cleanly on the GPU.
# The IC and all four boundaries are enforced *exactly* by construction:
#   u_θ(x,y,t) = sin(πx) sin(πy) · (1 + t · N_θ(x,y,t)),
# which is sin·sin (so u=0 on the boundary) and equals sin·sin at t=0 (the IC).
# This IC has the exact analytic solution
#   u*(x,y,t) = exp(-2 α π² t) · sin(πx) sin(πy),
# so we can report the PINN's true L2 error, and we train the SAME model on the
# CPU and the GPU to show the device move and the speed-up.
#
# Run on the GPU hub (the @pinn env has Lux + LuxCUDA + CUDA):
#   julia --project=@pinn units/unit_05/scripts/pinn_2d_diffusion_gpu.jl
# Nothing here runs during `quarto render` — the .qmd shows it `eval: false`.
# ===========================================================================

using Lux, LuxCUDA, CUDA, Optimisers, Zygote, Random, Printf, Statistics

const α = 0.1f0
const Tmax = 0.2f0
const HX = 2f-3; const HT = 2f-3

make_model() = Chain(Dense(3 => 48, tanh), Dense(48 => 48, tanh),
                     Dense(48 => 48, tanh), Dense(48 => 1))

u_exact(x, y, t) = exp(-2 * α * π^2 * t) * sinpi(x) * sinpi(y)

function solve_2d(Ncol, dev; iters = 3000, seed = 1)
    model = make_model()
    ps, st = Lux.setup(Xoshiro(seed), model)
    ps = ps |> dev; st = st |> dev
    opt = Optimisers.setup(Adam(2f-3), ps)

    rng = Xoshiro(seed + 7)
    X = rand(rng, Float32, 1, Ncol); Y = rand(rng, Float32, 1, Ncol); T = Tmax .* rand(rng, Float32, 1, Ncol)
    X, Y, T = dev(X), dev(Y), dev(T)

    N(p, x, y, t) = first(model(vcat(x, y, t), p, st))
    # hard IC (t=0 ⇒ sin·sin) and hard Dirichlet boundary (sin factors vanish):
    uθ(p, x, y, t) = sinpi.(x) .* sinpi.(y) .* (1f0 .+ t .* N(p, x, y, t))

    function loss(p)
        ut  = (uθ(p, X, Y, T .+ HT) .- uθ(p, X, Y, T .- HT)) ./ (2HT)
        uxx = (uθ(p, X .+ HX, Y, T) .- 2f0 .* uθ(p, X, Y, T) .+ uθ(p, X .- HX, Y, T)) ./ HX^2
        uyy = (uθ(p, X, Y .+ HX, T) .- 2f0 .* uθ(p, X, Y, T) .+ uθ(p, X, Y .- HX, T)) ./ HX^2
        return mean(abs2, ut .- α .* (uxx .+ uyy))
    end

    Zygote.gradient(loss, ps)                            # warm up / compile
    dev === identity || CUDA.synchronize()
    t0 = time()
    for _ in 1:iters
        g = Zygote.gradient(loss, ps)[1]
        opt, ps = Optimisers.update(opt, ps, g)
    end
    dev === identity || CUDA.synchronize()
    elapsed = time() - t0

    # L2 error vs analytic at t = Tmax on a 41×41 grid
    gr = range(0f0, 1f0; length = 41)
    xs = reshape(repeat(gr, inner = 41), 1, :); ys = reshape(repeat(gr, outer = 41), 1, :)
    ts = fill(Tmax, 1, length(xs))
    up = Array(uθ(ps, dev(xs), dev(ys), dev(ts)))
    ex = [u_exact(x, y, Tmax) for (x, y) in zip(vec(Array(xs)), vec(Array(ys)))]
    err = sqrt(mean((vec(up) .- ex).^2))
    return (; ps, model, st, uθ, elapsed, err, finalloss = loss(ps))
end

println("="^64)
println("Unit 5 — 2-D diffusion PINN (hand-built), CPU vs GPU")
println("="^64)
have_gpu = CUDA.functional()
@printf("GPU available: %s%s\n", have_gpu, have_gpu ? "  ($(CUDA.name(CUDA.device())))" : "")
println("u_θ = sin(πx)sin(πy)(1 + t·N);  residual ∂tu − α(∂xxu+∂yyu) by input stencil\n")

print("CPU  (N=4000)   … "); flush(stdout)
cpu = solve_2d(4_000, identity; iters = 2000)
@printf("%.1fs | loss %.2e | L2 err vs exact %.2e\n", cpu.elapsed, cpu.finalloss, cpu.err)

if have_gpu
    print("GPU  (N=120000) … "); flush(stdout)
    gpu = solve_2d(120_000, gpu_device(); iters = 3000)
    @printf("%.1fs | loss %.2e | L2 err vs exact %.2e\n", gpu.elapsed, gpu.finalloss, gpu.err)
    @printf("\nGPU trains 30× the collocation points (and 1.5× the iterations) in %.1f× the\n", gpu.elapsed/cpu.elapsed)
    @printf("CPU wall-clock, and reaches the analytic solution to L2 = %.1e.\n", gpu.err)

    try
        using CairoMakie
        ng = 81
        gr = range(0f0,1f0,length=ng)
        xs = reshape(repeat(gr, inner=ng),1,:); ys = reshape(repeat(gr, outer=ng),1,:); ts = fill(Tmax,1,ng*ng)
        up = reshape(Array(gpu.uθ(gpu.ps, CuArray(xs), CuArray(ys), CuArray(ts))), ng, ng)
        ex = [u_exact(x,y,Tmax) for x in gr, y in gr]
        f = Figure(size=(820,330))
        a1=Axis(f[1,1],title="PINN u(x,y,0.2) (GPU)",xlabel="x",ylabel="y",aspect=1)
        h1=heatmap!(a1,gr,gr,up;colormap=:viridis); Colorbar(f[1,2],h1)
        a2=Axis(f[1,3],title="exact u(x,y,0.2)",xlabel="x",ylabel="y",aspect=1)
        h2=heatmap!(a2,gr,gr,ex;colormap=:viridis); Colorbar(f[1,4],h2)
        figdir = get(ENV,"GPU_FIG_DIR",joinpath(@__DIR__,"..","figures")); isdir(figdir)||mkpath(figdir)
        save(joinpath(figdir,"pinn_2d_diffusion_gpu.png"), f)
        println("wrote figures/pinn_2d_diffusion_gpu.png")
    catch e
        println("(figure skipped: ", e, ")")
    end
end
