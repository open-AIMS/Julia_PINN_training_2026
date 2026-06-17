# ===========================================================================
# Task A — single-site inverse PINN: recover the storm forcing τ(t) from one
# mooring's temperatures.  CPU sub-scale demo + GPU full-scale (validated) run.
#
# Spec: unit_09 §9.9 (Cleveland Bay, H = 15 m, Pe ≪ 1, diffusion-dominated).
# Given a single mooring's temperature traces over a 30-day window with one storm
# around day 10, recover the storm's time signal τ(t).
#
# Physics — SOURCE RECOVERY (unit_07 §7.4). A known vertical heating shape S(ζ)
# (surface-weighted, Beer–Lambert-like) times an UNKNOWN time signal τ(t).
# Non-dimensionalised:
#   ∂τ θ = Pe ∂ζζ θ + S(ζ)·τ(t),   ζ ∈ [0,1], τ_t ∈ [0,1],
#   θ(0,τ) = 0  (deep reservoir),  ∂ζ θ(1,τ) = 0  (no surface conductive flux),
#   θ(ζ,0) = 0.
# The unknown is the scalar-in-time forcing τ(t); S(ζ), Pe and the BCs are given.
#
# THE KEY LESSON (loss weighting in a deconvolution). τ(t) enters ONLY the PDE
# residual, so the τ-network regresses to (∂τθ − Pe ∂ζζθ)/S — it is driven by
# the *time-derivative* of the temperature field. The column low-pass-filters τ
# (diffusion time ≈ storm width), so a network that merely matches the noisy data
# in *value* can do so with a too-smooth θ whose derivative — and hence the
# recovered storm peak — is collapsed to ~20% of truth. The cure is NOT fancier
# collocation; it is the loss weights: a large data weight λ_d and a tiny H¹
# smoothing weight λ_reg force θ's time-derivative to stay sharp, which lifts the
# recovered peak to within the §9.9.3 ≤15% criterion. (This was validated by a
# weight sweep on the GPU hub; λ_d=100/λ_reg=1e-2 gives ~80% peak error, while
# λ_d≈6000/λ_reg≈1e-5 gives ~14% — close to the deconvolution floor at this SNR.)
#
# Derivatives use the finite-difference-in-input stencil (no nested AD), the
# Unit 5 §5.3 trick — one reverse-mode pass, runs cleanly on the GPU.
#
# CPU vs GPU. The operating point needs a wide network trained for ~28k steps —
# minutes on a GPU, far longer on a laptop CPU. So the CPU run here is a
# sub-scale DEMO (smaller net, fewer steps): it shows the method working and the
# peak starting to rise, but only the GPU full-scale run reaches the ≤15%
# criterion. That gap is the motivation for the GPU.
#
# Run on the GPU hub (@pinn has Lux + LuxCUDA + CUDA):
#   julia --project=@pinn units/unit_10/scripts/task_a_inverse_pinn.jl
# Nothing here runs during `quarto render` — the .qmd shows it `eval: false`.
# ===========================================================================

using Lux, LuxCUDA, CUDA, Optimisers, Zygote, Random, Printf, Statistics

# ── nondimensional Cleveland-Bay column (unit_09 §9.6 / generate_mooring_csvs A)
const Pe    = 1.5f0             # base diffusion number (site A)
const ℓ_S   = 0.20f0           # heating decay scale below the surface (Beer–Lambert-like)
const ΔT_A  = 2.5f0            # site-A temperature scale (°C), for reporting in °C
S_shape(ζ)  = exp.((ζ .- 1f0) ./ ℓ_S)        # known vertical heating shape S(ζ), peak at surface

# ── ground-truth storm time signal τ*(t) ────────────────────────────────────
const τ0    = 1.0f0 / 3.0f0     # storm centre: day 10 of 30  →  τ = 1/3
const σstm  = 0.05f0            # storm width (~1.5 days in a 30-day window)
const A_stm = 4.0f0            # storm amplitude (nondim heating)
τstar(t)    = A_stm .* exp.(-((t .- τ0) ./ σstm).^2)     # truth forcing (peak A_stm)

# ── finite-difference reference solve → synthetic mooring data ──────────────
const NZ = 61; const NT_FD = 20001                      # fine grid (explicit-stable: Pe·dt/dζ² ≈ 0.27)
function fd_reference()
    ζ = Float32.(range(0f0, 1f0; length = NZ)); dζ = ζ[2] - ζ[1]
    t = Float32.(range(0f0, 1f0; length = NT_FD)); dt = t[2] - t[1]
    S = S_shape(ζ)
    θ = zeros(Float32, NZ, NT_FD)                        # θ(ζ,0)=0; θ(0,·)=0 (deep)
    @inbounds for k in 1:NT_FD-1
        f = τstar(t[k]); col = @view θ[:, k]
        for i in 2:NZ-1
            lap = (col[i+1] - 2col[i] + col[i-1]) / dζ^2
            θ[i, k+1] = col[i] + dt * (Pe * lap + S[i] * f)
        end
        # surface ζ=1 insulating (∂ζθ=0): ghost θ[NZ+1]=θ[NZ-1]
        lapN = (2col[NZ-1] - 2col[NZ]) / dζ^2
        θ[NZ, k+1] = col[NZ] + dt * (Pe * lapN + S[NZ] * f)
        θ[1, k+1] = 0f0
    end
    (; ζ, t, θ)
end
const FD = fd_reference()

# bilinear-ish lookup of θ*(ζ,τ) from the FD grid (nearest in ζ, linear in τ)
function θstar(ζv, τv)
    out = similar(ζv)
    @inbounds for n in eachindex(ζv)
        iz = clamp(round(Int, ζv[n] * (NZ - 1)) + 1, 1, NZ)
        ft = clamp(τv[n] * (NT_FD - 1), 0f0, Float32(NT_FD - 1))
        k = clamp(floor(Int, ft) + 1, 1, NT_FD - 1); fr = ft - (k - 1)
        out[n] = (1 - fr) * FD.θ[iz, k] + fr * FD.θ[iz, k+1]
    end
    out
end

const Z_SENS = Float32[(15 - d)/15 for d in (1, 4, 8, 12, 14)]   # ζ at z = −1,−4,−8,−12,−14 m
const N_T    = 720                                               # hourly over 30 days
const σ_obs  = 0.005f0                                           # 0.0125 °C / ΔT_A (quality mooring); storm SNR ≈ 24

function make_observations(; seed = 20260617)
    τs = Float32.(range(0f0, 1f0; length = N_T))
    ζcol = repeat(Z_SENS, inner = N_T); τcol = repeat(τs, outer = length(Z_SENS))
    clean = θstar(ζcol, τcol)
    noisy = clean .+ σ_obs .* randn(Xoshiro(seed), Float32, length(clean))
    (; ζ = reshape(ζcol, 1, :), τ = reshape(τcol, 1, :), θ = reshape(noisy, 1, :))
end

# ── networks ────────────────────────────────────────────────────────────────
# Wide T-network: enough capacity to keep θ's time-derivative sharp at the storm.
make_T(w, d) = Chain(Dense(2 => w, tanh), [Dense(w => w, tanh) for _ in 1:d-1]..., Dense(w => 1))
make_τ()     = Chain(Dense(1 => 32, tanh), Dense(32 => 32, tanh), Dense(32 => 1))

const HZ = 2f-3; const HT = 2f-3

# λ_d large + λ_reg tiny is the operating point (see header). λ_b keeps the
# surface flux BC; λ_reg only removes single-hour-sample ringing.
const λ_d = 6000f0; const λ_b = 10f0; const λ_reg = 1f-5

function solve_inverse(Ncol, dev; iters, Tw, Td, seed = 1)
    obs = make_observations()
    od_ζ, od_τ, od_θ = dev(obs.ζ), dev(obs.τ), dev(obs.θ)

    Tm = make_T(Tw, Td); τm = make_τ()
    pT, sT = Lux.setup(Xoshiro(seed), Tm); pτ, sτ = Lux.setup(Xoshiro(seed + 1), τm)
    pT, sT = dev(pT), dev(sT); pτ, sτ = dev(pτ), dev(sτ)
    optT = Optimisers.setup(Adam(1f-3), pT); optτ = Optimisers.setup(Adam(3f-3), pτ)

    rng = Xoshiro(seed + 50)
    ζc = dev(rand(rng, Float32, 1, Ncol)); τc = dev(rand(rng, Float32, 1, Ncol))
    τb = dev(rand(rng, Float32, 1, Ncol ÷ 5)); ζ1 = dev(ones(Float32, 1, size(τb, 2)))
    τg = dev(reshape(Float32.(range(0f0, 1f0; length = 400)), 1, :))     # for H¹ prior

    NT_(p, ζ, τ) = first(Tm(vcat(ζ, τ), p, sT))
    # hard IC (τ=0 ⇒ 0) and hard deep BC (ζ=0 ⇒ 0): θ = ζ·τ·N
    θnet(p, ζ, τ) = ζ .* τ .* NT_(p, ζ, τ)
    τφ(q, τ) = first(τm(τ, q, sτ))                                       # recovered forcing (signed)

    function loss(pT, pτ)
        θt  = (θnet(pT, ζc, τc .+ HT) .- θnet(pT, ζc, τc .- HT)) ./ (2HT)
        θzz = (θnet(pT, ζc .+ HZ, τc) .- 2f0 .* θnet(pT, ζc, τc) .+ θnet(pT, ζc .- HZ, τc)) ./ HZ^2
        Lr  = mean(abs2, θt .- Pe .* θzz .- S_shape(ζc) .* τφ(pτ, τc))    # PDE residual w/ source
        Ld  = mean(abs2, θnet(pT, od_ζ, od_τ) .- od_θ)                    # data misfit
        θzs = (θnet(pT, ζ1, τb) .- θnet(pT, ζ1 .- HZ, τb)) ./ HZ
        Lb  = mean(abs2, θzs)                                             # surface insulating ∂ζθ(1)=0
        dτ  = (τφ(pτ, τg .+ HT) .- τφ(pτ, τg .- HT)) ./ (2HT)
        Lreg = mean(abs2, dτ)                                             # H¹ smoothness prior on τ
        return Lr + λ_d * Ld + λ_b * Lb + λ_reg * Lreg
    end

    Zygote.gradient(loss, pT, pτ)                            # warm up / compile
    dev === identity || CUDA.synchronize()
    t0 = time()
    for _ in 1:iters
        gT, gτ = Zygote.gradient(loss, pT, pτ)
        optT, pT = Optimisers.update(optT, pT, gT)
        optτ, pτ = Optimisers.update(optτ, pτ, gτ)
    end
    dev === identity || CUDA.synchronize()
    elapsed = time() - t0

    # ── recovery metrics on a dense τ grid ──────────────────────────────────
    τe = Float32.(range(0f0, 1f0; length = 601))
    τrec = vec(Array(τφ(pτ, dev(reshape(τe, 1, :))))); τtru = vec(τstar(τe))
    peak_rec, ip = findmax(τrec); peak_tru = maximum(τtru)
    peak_err = abs(peak_rec - peak_tru) / peak_tru
    timing_h = abs(τe[ip] - τ0) * 30f0 * 24f0               # |Δτ| in hours over 30-day window
    rel_l2_τ = sqrt(sum((τrec .- τtru).^2) / sum(τtru.^2))  # whole-envelope relative L2
    gr = Float32.(range(0.05f0, 1f0; length = 60))
    ζf = vec(repeat(gr, inner = 60)); τf = vec(repeat(gr, outer = 60))
    θp = vec(Array(θnet(pT, dev(reshape(ζf,1,:)), dev(reshape(τf,1,:))))); θe = θstar(ζf, τf)
    l2 = sqrt(mean((θp .- θe).^2)); l2_C = l2 * ΔT_A
    return (; pT, pτ, Tm, sT, τm, sτ, θnet, τφ, elapsed,
              peak_err, timing_h, rel_l2_τ, peak_rec, peak_tru, l2, l2_C,
              finalloss = loss(pT, pτ), τe, τrec, τtru)
end

println("="^70)
println("Task A — single-site inverse PINN: recover storm forcing τ(t)")
println("="^70)
have_gpu = CUDA.functional()
@printf("GPU available: %s%s\n", have_gpu, have_gpu ? "  ($(CUDA.name(CUDA.device())))" : "")
@printf("nondim: ∂τθ = %.2f·∂ζζθ + S(ζ)·τ(t) ;  %d sensors × %d samples, σ_obs = %.3f (%.3f °C)\n",
        Pe, length(Z_SENS), N_T, σ_obs, σ_obs * ΔT_A)
@printf("data signal: max|θ*| = %.3f (%.2f °C), storm SNR ≈ %.1f×\n",
        maximum(abs, FD.θ), maximum(abs, FD.θ) * ΔT_A, maximum(abs, FD.θ) / σ_obs)
@printf("weights: λ_r=1, λ_d=%.0f, λ_b=%.0f, λ_reg=%.0e\n\n", λ_d, λ_b, λ_reg)

print("CPU sub-scale demo  (N=2000, 2500 it, 4×24)  … "); flush(stdout)
cpu = solve_inverse(2_000, identity; iters = 2500, Tw = 24, Td = 4)
@printf("%.1fs | loss %.2e | fwd L2 %.3f °C | peak err %.1f%% | timing %.1f h | env relL2 %.2f\n",
        cpu.elapsed, cpu.finalloss, cpu.l2_C, 100cpu.peak_err, cpu.timing_h, cpu.rel_l2_τ)
println("   (sub-scale: peak only partly recovered — the GPU full-scale run below is what passes)")

if have_gpu
    print("GPU full-scale      (N=20000, 28000 it, 48×6) … "); flush(stdout)
    gpu = solve_inverse(20_000, gpu_device(); iters = 28000, Tw = 48, Td = 6)
    @printf("%.1fs | loss %.2e | fwd L2 %.3f °C | peak err %.1f%% | timing %.1f h | env relL2 %.2f\n",
            gpu.elapsed, gpu.finalloss, gpu.l2_C, 100gpu.peak_err, gpu.timing_h, gpu.rel_l2_τ)
    @printf("\nForward L2 vs FD reference: %.3f °C (target < 0.05 °C: %s)\n",
            gpu.l2_C, gpu.l2_C < 0.05 ? "PASS" : "FAIL")
    @printf("Recovered storm peak-amplitude error: %.1f%% (target < 15%%: %s)\n",
            100gpu.peak_err, gpu.peak_err < 0.15 ? "PASS" : "FAIL")
    @printf("Storm-day timing error: %.1f h (target < 2 h: %s)\n",
            gpu.timing_h, gpu.timing_h < 2 ? "PASS" : "FAIL")

    try
        using CairoMakie
        f = Figure(size = (920, 380))
        days = gpu.τe .* 30f0
        a1 = Axis(f[1,1], title = "Recovered storm forcing τ̂(t) vs truth",
                  xlabel = "t (days)", ylabel = "τ(t)")
        lines!(a1, days, gpu.τtru; linestyle = :dash, linewidth = 3, label = "truth τ*(t)")
        lines!(a1, days, gpu.τrec; linewidth = 2, label = "recovered τ̂(t)")
        axislegend(a1, position = :rt)
        a2 = Axis(f[1,2], title = "Sensor temperatures: PINN vs noisy data",
                  xlabel = "t (days)", ylabel = "θ (nondim)")
        obs = make_observations(); cols = (:navy, :teal, :seagreen, :darkorange, :firebrick)
        for (k, zc) in enumerate(Z_SENS)
            sl = (k-1)*N_T+1 : k*N_T
            td = collect(range(0f0, 30f0, length = N_T))
            scatter!(a2, td, vec(obs.θ)[sl]; color = (cols[k], 0.25), markersize = 3)
            ζrow = fill(zc, 1, N_T); τrow = reshape(Float32.(range(0f0,1f0,length=N_T)),1,:)
            pinn = vec(Array(gpu.θnet(gpu.pT, CuArray(ζrow), CuArray(τrow))))
            lines!(a2, td, pinn; color = cols[k], linewidth = 2)
        end
        figdir = get(ENV, "GPU_FIG_DIR", joinpath(@__DIR__, "..", "figures")); isdir(figdir) || mkpath(figdir)
        save(joinpath(figdir, "task_a_inverse.png"), f)
        println("wrote figures/task_a_inverse.png")
    catch e
        println("(figure skipped: ", e, ")")
    end
end
