# ===========================================================================
# Unit 10 — the capstone column model as a forward PINN, CPU vs GPU.
#
# This is the Step-1 *forward sanity check* the capstone calls for
# (§10.2.1 / §10.3.1) before any inversion: solve the column's pure-diffusion
# relaxation toward steady state with a hand-built PINN, on the CPU at a small
# collocation budget and on the GPU at a large one, and report the measured
# wall-clock and the error against a CLOSED-FORM analytic solution.
#
# Physics (scenario 1 of column_fd.jl, non-dimensionalised). With
#   T̃ = (T − T_deep)/(T_surface − T_deep),  ζ = (z + H)/H,  τ = t/T_f,
# pure vertical diffusion under steady surface cooling is
#   ∂τ T̃ = Pe ∂ζζ T̃,   ζ ∈ [0,1], τ ∈ [0,1],
#   T̃(0,τ) = 0   (deep reservoir, Dirichlet),
#   ∂ζ T̃(1,τ) = g (surface cooling flux, Neumann),
# with Pe = κ_m T_f / H² ≈ 3.0 and g = −Q_cool H /(κ_m ρ c_p ΔT) ≈ −0.49 taken
# straight from the §9.6 reference parameters. The steady state is the straight
# line T̃_ss(ζ) = g·ζ (the §10.1 exercise answer).
#
# Why a smooth-anomaly initial condition (not the tanh thermocline). The
# operator ∂ζζ with T̃(0)=0, ∂ζT̃(1)=0 has eigenfunctions sin(λ_n ζ),
# λ_n = (n−½)π, each decaying like exp(−Pe λ_n² τ). Seeding the FIRST mode as
# an anomaly about the steady state,
#   T̃(ζ,0) = g·ζ + A sin(λ₁ ζ),   λ₁ = π/2,
# gives an EXACT solution for all time,
#   T̃*(ζ,τ) = g·ζ + A sin(λ₁ ζ) exp(−Pe λ₁² τ),
# which satisfies the PDE and BOTH boundary conditions identically — so we can
# report the PINN's true L2 error. The capstone's sharp tanh thermocline excites
# many fast-decaying high modes (a stiff time-boundary-layer near τ=0); that
# stiffness is exactly what motivates Task B's Fourier-feature / causal-training
# machinery (§10.3) and is left to the FD reference (column_fd.jl). Here we want
# a clean, well-posed forward solve whose convergence we can certify to a number.
#
# Derivatives use a finite-difference stencil in the input (no nested AD), the
# same trick as Unit 5 §5.3, so the parameter gradient is a single reverse-mode
# pass that runs cleanly on the GPU.
#
# Run on the GPU hub (@pinn has Lux + LuxCUDA + CUDA):
#   julia --project=@pinn units/unit_10/scripts/column_pinn_gpu.jl
# Nothing here runs during `quarto render` — the .qmd shows it `eval: false`.
# ===========================================================================

using Lux, LuxCUDA, CUDA, Optimisers, Zygote, Random, Printf, Statistics

# physical constants (unit_09 §9.6 / column_fd.jl PARAMS), non-dimensionalised
const H = 100.0f0; const Tdeep = 18.0f0; const Tsurf = 28.0f0; const ΔT = Tsurf - Tdeep
const κm = 1.0f-3; const Qcool = 200.0f0; const ρ = 1025.0f0; const cp = 3990.0f0
const Tf = 350.0f0 * 86400.0f0
const Pe = κm * Tf / H^2                                  # ≈ 3.02
const g_surf = -Qcool * H / (κm * ΔT * ρ * cp)            # nondim surface gradient ≈ -0.488

const λ1 = Float32(π) / 2                                 # first eigenmode of ∂ζζ, T̃(0)=0, ∂ζT̃(1)=0
const A1 = 0.5f0                                          # initial anomaly amplitude about steady state

T̃ss(ζ)        = g_surf .* ζ                               # analytic steady state
T̃_ic(ζ)       = g_surf .* ζ .+ A1 .* sin.(λ1 .* ζ)        # initial condition (smooth anomaly)
T̃_exact(ζ, τ) = g_surf .* ζ .+ A1 .* sin.(λ1 .* ζ) .* exp.(-Pe * λ1^2 .* τ)   # closed form

make_model() = Chain(Dense(2 => 64, tanh), Dense(64 => 64, tanh),
                     Dense(64 => 64, tanh), Dense(64 => 1))

const HZ = 2f-3; const HT = 2f-3

function solve_column(Ncol, dev; iters = 3000, seed = 1)
    model = make_model()
    ps, st = Lux.setup(Xoshiro(seed), model)
    ps = ps |> dev; st = st |> dev
    opt = Optimisers.setup(Adam(2f-3), ps)

    rng = Xoshiro(seed + 100)
    ζc = rand(rng, Float32, 1, Ncol); τc = rand(rng, Float32, 1, Ncol)            # interior collocation
    τs = rand(rng, Float32, 1, Ncol ÷ 4)                                          # surface BC points (ζ=1)
    ζc, τc, τs = dev(ζc), dev(τc), dev(τs)
    ζ1 = dev(ones(Float32, 1, size(τs, 2)))

    Nθ(p, ζ, τ) = first(model(vcat(ζ, τ), p, st))
    # Hard IC (τ=0 ⇒ T̃_ic) and hard bottom Dirichlet (ζ=0 ⇒ 0): T̃ = T̃_ic(ζ) + ζ·τ·Nθ.
    T̃hat(p, ζ, τ) = T̃_ic(ζ) .+ ζ .* τ .* Nθ(p, ζ, τ)

    function loss(p)
        # ∂τT̃ and ∂ζζT̃ by central differences in the input
        T̃t  = (T̃hat(p, ζc, τc .+ HT) .- T̃hat(p, ζc, τc .- HT)) ./ (2HT)
        T̃zz = (T̃hat(p, ζc .+ HZ, τc) .- 2f0 .* T̃hat(p, ζc, τc) .+ T̃hat(p, ζc .- HZ, τc)) ./ HZ^2
        Lpde = mean(abs2, T̃t .- Pe .* T̃zz)
        # surface flux ∂ζT̃(1,τ) = g  (one-sided difference at ζ=1)
        T̃zs = (T̃hat(p, ζ1, τs) .- T̃hat(p, ζ1 .- HZ, τs)) ./ HZ
        Lbc = mean(abs2, T̃zs .- g_surf)
        return Lpde + 10f0 * Lbc
    end

    Zygote.gradient(loss, ps)                                 # warm up / compile
    dev === identity || CUDA.synchronize()
    t0 = time()
    for _ in 1:iters
        g = Zygote.gradient(loss, ps)[1]
        opt, ps = Optimisers.update(opt, ps, g)
    end
    dev === identity || CUDA.synchronize()
    elapsed = time() - t0

    # L2 error vs the CLOSED-FORM solution over the whole (ζ,τ) domain
    gr = range(0f0, 1f0; length = 101)
    ζg = reshape(repeat(gr, inner = 101), 1, :); τg = reshape(repeat(gr, outer = 101), 1, :)
    T̃p = Array(T̃hat(ps, dev(ζg), dev(τg)))
    T̃e = Array(T̃_exact(ζg, τg))
    l2 = sqrt(mean((vec(T̃p) .- vec(T̃e)).^2))
    return (; ps, model, st, elapsed, l2, finalloss = loss(ps))
end

println("="^66)
println("Unit 10 — capstone column forward PINN: CPU sub-scale vs GPU full-scale")
println("="^66)
have_gpu = CUDA.functional()
@printf("GPU available: %s%s\n", have_gpu, have_gpu ? "  ($(CUDA.name(CUDA.device())))" : "")
@printf("non-dim PDE: ∂τT̃ = %.2f ∂ζζT̃ ;  steady state T̃_ss(ζ) = %.3f·ζ\n", Pe, g_surf)
@printf("IC = T̃_ss + %.2f·sin(πζ/2);  exact: T̃* = T̃_ss + %.2f·sin(πζ/2)·exp(−%.2f·τ)\n\n",
        A1, A1, Pe * λ1^2)

print("CPU sub-scale  (N=4000)  … "); flush(stdout)
cpu = solve_column(4_000, identity; iters = 2000)
@printf("%.1fs  | final loss %.2e | L2 vs exact = %.3e\n", cpu.elapsed, cpu.finalloss, cpu.l2)

if have_gpu
    print("GPU full-scale (N=120000)… "); flush(stdout)
    gpu = solve_column(120_000, gpu_device(); iters = 5000)
    @printf("%.1fs  | final loss %.2e | L2 vs exact = %.3e\n", gpu.elapsed, gpu.finalloss, gpu.l2)
    @printf("\nThe GPU trains a 30× larger collocation set (and 2.5× the iterations) in\n")
    @printf("%.1f× the CPU sub-scale wall-clock — i.e. ~%.0f× the collocation-iterations\n",
            gpu.elapsed / cpu.elapsed, (120_000*5000)/(4_000*2000) / (gpu.elapsed/cpu.elapsed))
    @printf("per second — and reaches the closed-form solution to L2 = %.1e.\n", gpu.l2)

    # figure: PINN field T̃(ζ,τ) + profiles at several τ vs the closed form
    try
        using CairoMakie
        # re-create the trained ansatz for evaluation on a dense grid
        Nθ(ζ, τ) = first(gpu.model(vcat(ζ, τ), gpu.ps, gpu.st))
        T̃h(ζ, τ) = T̃_ic(ζ) .+ ζ .* τ .* Nθ(ζ, τ)
        ng = 120
        gr = range(0f0, 1f0, length = ng)
        ζg = reshape(repeat(gr, inner = ng), 1, :); τg = reshape(repeat(gr, outer = ng), 1, :)
        field = reshape(Array(T̃h(CuArray(ζg), CuArray(τg))), ng, ng)   # (ζ, τ)
        f = Figure(size = (880, 360))
        a1 = Axis(f[1,1], title = "PINN T̃(ζ,τ) (GPU)", xlabel = "τ (t/T_f)", ylabel = "ζ ((z+H)/H)")
        hm = heatmap!(a1, gr, gr, field; colormap = :thermal); Colorbar(f[1,2], hm, label = "T̃")
        a2 = Axis(f[1,3], title = "profiles vs closed form (dashed)", xlabel = "T̃", ylabel = "ζ")
        ζl = reshape(collect(gr), 1, :)
        for (τ0, col) in zip((0f0, 0.05f0, 0.2f0, 1f0), (:navy, :teal, :darkorange, :firebrick))
            τl = fill(τ0, 1, ng)
            pinn = vec(Array(T̃h(CuArray(ζl), CuArray(τl))))
            exact = vec(T̃_exact(ζl, τl))
            lines!(a2, exact, collect(gr); color = col, linestyle = :dash, linewidth = 2)
            lines!(a2, pinn, collect(gr); color = col, linewidth = 2, label = "τ = $(τ0)")
        end
        axislegend(a2, position = :lt)
        figdir = get(ENV, "GPU_FIG_DIR", joinpath(@__DIR__, "..", "figures")); isdir(figdir) || mkpath(figdir)
        save(joinpath(figdir, "column_pinn_gpu.png"), f)
        println("wrote figures/column_pinn_gpu.png")
    catch e
        println("(figure skipped: ", e, ")")
    end
end
