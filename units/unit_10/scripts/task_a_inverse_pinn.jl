# ===========================================================================
# Task A — single-site inverse PINN: recover the storm forcing τ(t) from one
# mooring's temperatures.  Runs to completion on a CPU in minutes.
#
# Spec: unit_09 §9.9 (Site B, Davies Reef, H = 60 m, Pe ~ 1, vertical
# advection + diffusion). Given a single mooring's temperature traces over a
# 30-day window with one storm around day 10, recover the storm's time signal
# τ(t).
#
# Why Site B and not the shallow diffusion-only Cleveland Bay: a 60 m column
# with upwelling carries a sharp, depth-structured storm fingerprint, so a
# single mooring already pins τ(t) to within the §9.9.3 ≤15% bar — on a CPU.
# (The shallower, diffusion-only site washes the storm out and is the harder,
# less informative inverse; it is kept only as the "easy site" contrast in the
# Task B joint study, §10.3.)
#
# Physics — SOURCE RECOVERY (unit_07 §7.4). A known vertical heating shape S(ζ)
# (surface-weighted, Beer–Lambert-like) times an UNKNOWN time signal τ(t), on a
# column with vertical advection (upwelling) and diffusion. Non-dimensionalised:
#   ∂τ T̃ = -Wadv·w(ζ)·∂ζ T̃ + Pe·∂ζζ T̃ + S(ζ)·τ(t),   ζ ∈ [0,1], τ_t ∈ [0,1],
#   T̃(0,τ) = 0  (deep reservoir),  ∂ζ T̃(1,τ) = 0  (no surface conductive flux),
#   T̃(ζ,0) = 0,   w(ζ) = ζ  (upwelling: zero at depth, max at surface).
# The unknown is the scalar-in-time forcing τ(t); S(ζ), Pe, Wadv and the BCs
# are given.
#
# THE KEY LESSON (loss weighting in a deconvolution). τ(t) enters ONLY the PDE
# residual, so the τ-network regresses to a combination of T̃'s time- and
# space-derivatives divided by S — it is driven by *derivatives* of the
# temperature field. The column low-pass-filters τ (response time ≈ storm
# width), so a network that merely matches the noisy data in *value* can do so
# with a too-smooth T̃ whose derivatives — and hence the recovered storm peak —
# collapse well below truth. The cure is NOT fancier collocation; it is the loss
# weights: a large data weight λ_d and a tiny H¹ smoothing weight λ_reg force
# T̃'s derivatives to stay sharp, which lifts the recovered peak to within the
# §9.9.3 ≤15% criterion. (Validated by a weight sweep: λ_d=100/λ_reg=1e-2 gives
# ~92% peak error (only ~8% of truth recovered), while λ_d≈6000/λ_reg≈1e-5 lands near the deconvolution floor
# at this SNR — see §10.2.)
#
# Derivatives use the finite-difference-in-input stencil (no nested AD), the
# Unit 5 §5.3 trick — one reverse-mode pass.
#
# Run (CPU is fine; a GPU only makes it faster):
#   julia --project=@pinn units/unit_10/scripts/task_a_inverse_pinn.jl
# Smoke test:  TASKA_NCOL=300 TASKA_ITERS=300 julia --project=. <this file>
# Nothing here runs during `quarto render` — the .qmd shows it `eval: false`.
# ===========================================================================

using Lux, Optimisers, Zygote, Random, Printf, Statistics

# ── nondimensional Davies-Reef column (unit_09 §9.6 / mooring_B.csv) ──────────
const Pe    = 1.0f0            # diffusion number (site B)
const Wadv  = 1.2f0           # advection number (upwelling) → Péclet ~ 1
const ℓ_S   = 0.25f0          # heating decay scale below the surface (Beer–Lambert-like)
const ΔT_B  = 3.0f0           # site-B temperature scale (°C), for reporting in °C
const H     = 60f0            # column depth (m)
S_shape(ζ)  = exp.((ζ .- 1f0) ./ ℓ_S)        # known vertical heating shape S(ζ), peak at surface
w_shape(ζ)  = ζ                              # upwelling shape: 0 at depth, max at surface

# ===========================================================================
# PART 1 — GENERATE THE TWIN'S SYNTHETIC DATA (this is NOT the solution).
# We PLANT a known storm τ*(t), forward-solve the column for the truth field,
# then sample it at the sensors and add noise. In the wild this would be real
# mooring data we don't get to choose; here we choose it so the recovered τ̂
# can be scored against a KNOWN truth — an identical-twin experiment (§9.7).
# The committed mooring_*.csv are generated separately (column_fd.jl); this
# in-file solve is the scorable twin the inverse in Part 2 actually uses.
# ===========================================================================
# ── ground-truth storm time signal τ*(t) ────────────────────────────────────
const τ0    = 1.0f0 / 3.0f0     # storm centre: day 10 of 30  →  τ = 1/3
const σstm  = 0.05f0            # storm width (~1.5 days in a 30-day window)
const A_stm = 4.0f0            # storm amplitude (nondim heating)
τstar(t)    = A_stm .* exp.(-((t .- τ0) ./ σstm).^2)     # truth forcing (peak A_stm)

# ── finite-difference reference solve → synthetic mooring data ──────────────
const NZ = 61; const NT_FD = 40001                     # fine grid (explicit-stable)
function fd_reference()
    ζ = Float32.(range(0f0, 1f0; length = NZ)); dζ = ζ[2] - ζ[1]
    t = Float32.(range(0f0, 1f0; length = NT_FD)); dt = t[2] - t[1]
    S = S_shape(ζ); w = w_shape(ζ)
    T̃ = zeros(Float32, NZ, NT_FD)                        # T̃(ζ,0)=0; T̃(0,·)=0 (deep)
    @inbounds for k in 1:NT_FD-1
        f = τstar(t[k]); col = @view T̃[:, k]
        for i in 2:NZ-1
            lap = (col[i+1] - 2col[i] + col[i-1]) / dζ^2
            adv = w[i] * (col[i+1] - col[i-1]) / (2dζ)   # central; upwelling carries deep water up
            T̃[i, k+1] = col[i] + dt * (-Wadv * adv + Pe * lap + S[i] * f)
        end
        # surface ζ=1 insulating (∂ζT̃=0): ghost T̃[NZ+1]=T̃[NZ-1]
        lapN = (2col[NZ-1] - 2col[NZ]) / dζ^2
        advN = w[NZ] * (col[NZ] - col[NZ-1]) / dζ
        T̃[NZ, k+1] = col[NZ] + dt * (-Wadv * advN + Pe * lapN + S[NZ] * f)
        T̃[1, k+1] = 0f0
    end
    (; ζ, t, T̃)
end
const FD = fd_reference()

# bilinear-ish lookup of T̃*(ζ,τ) from the FD grid (nearest in ζ, linear in τ)
function T̃star(ζv, τv)
    out = similar(ζv)
    @inbounds for n in eachindex(ζv)
        iz = clamp(round(Int, ζv[n] * (NZ - 1)) + 1, 1, NZ)
        ft = clamp(τv[n] * (NT_FD - 1), 0f0, Float32(NT_FD - 1))
        k = clamp(floor(Int, ft) + 1, 1, NT_FD - 1); fr = ft - (k - 1)
        out[n] = (1 - fr) * FD.T̃[iz, k] + fr * FD.T̃[iz, k+1]
    end
    out
end

const Z_SENS = Float32[(H - d)/H for d in (2, 10, 25, 45, 58)]   # ζ at z = −2,−10,−25,−45,−58 m
const N_T    = 720                                               # hourly over 30 days
const σ_obs  = 0.005f0                                           # 0.015 °C / ΔT_B (quality mooring)

function make_observations(; seed = 20260617)
    τs = Float32.(range(0f0, 1f0; length = N_T))
    ζcol = repeat(Z_SENS, inner = N_T); τcol = repeat(τs, outer = length(Z_SENS))
    clean = T̃star(ζcol, τcol)
    noisy = clean .+ σ_obs .* randn(Xoshiro(seed), Float32, length(clean))
    (; ζ = reshape(ζcol, 1, :), τ = reshape(τcol, 1, :), T̃ = reshape(noisy, 1, :))
end

# ===========================================================================
# PART 2 — THE INVERSE PINN (the actual solution). Recover τ̂(t) from the noisy
# data made in Part 1; the planted τ* is used ONLY to score the result.
# ===========================================================================
# ── networks ────────────────────────────────────────────────────────────────
# T-network: enough capacity to keep T̃'s derivatives sharp at the storm.
make_T(w, d) = Chain(Dense(2 => w, tanh), [Dense(w => w, tanh) for _ in 1:d-1]..., Dense(w => 1))
make_τ()     = Chain(Dense(1 => 32, tanh), Dense(32 => 32, tanh), Dense(32 => 1))

const HZ = 2f-3; const HT = 2f-3

# λ_d large + λ_reg tiny is the operating point (see header). λ_b keeps the
# surface flux BC; λ_reg only removes single-hour-sample ringing.
const λ_d = 6000f0; const λ_b = 10f0; const λ_reg = 1f-5

function solve_inverse(Ncol; iters, Tw, Td, seed = 1)
    obs = make_observations()
    od_ζ, od_τ, od_T̃ = obs.ζ, obs.τ, obs.T̃

    Tm = make_T(Tw, Td); τm = make_τ()
    pT, sT = Lux.setup(Xoshiro(seed), Tm); pτ, sτ = Lux.setup(Xoshiro(seed + 1), τm)
    optT = Optimisers.setup(Adam(1f-3), pT); optτ = Optimisers.setup(Adam(3f-3), pτ)

    rng = Xoshiro(seed + 50)
    ζc = rand(rng, Float32, 1, Ncol); τc = rand(rng, Float32, 1, Ncol)
    τb = rand(rng, Float32, 1, Ncol ÷ 5); ζ1 = ones(Float32, 1, size(τb, 2))
    τg = reshape(Float32.(range(0f0, 1f0; length = 400)), 1, :)          # for H¹ prior

    NT_(p, ζ, τ) = first(Tm(vcat(ζ, τ), p, sT))
    # hard IC (τ=0 ⇒ 0) and hard deep BC (ζ=0 ⇒ 0): T̃ = ζ·τ·N
    T̃net(p, ζ, τ) = ζ .* τ .* NT_(p, ζ, τ)
    τφ(q, τ) = first(τm(τ, q, sτ))                                       # recovered forcing (signed)

    function loss(pT, pτ)
        T̃t  = (T̃net(pT, ζc, τc .+ HT) .- T̃net(pT, ζc, τc .- HT)) ./ (2HT)
        T̃z  = (T̃net(pT, ζc .+ HZ, τc) .- T̃net(pT, ζc .- HZ, τc)) ./ (2HZ)
        T̃zz = (T̃net(pT, ζc .+ HZ, τc) .- 2f0 .* T̃net(pT, ζc, τc) .+ T̃net(pT, ζc .- HZ, τc)) ./ HZ^2
        Lr  = mean(abs2, T̃t .+ Wadv .* w_shape(ζc) .* T̃z .- Pe .* T̃zz .- S_shape(ζc) .* τφ(pτ, τc))
        Ld  = mean(abs2, T̃net(pT, od_ζ, od_τ) .- od_T̃)                    # data misfit
        T̃zs = (T̃net(pT, ζ1, τb) .- T̃net(pT, ζ1 .- HZ, τb)) ./ HZ
        Lb  = mean(abs2, T̃zs)                                             # surface insulating ∂ζT̃(1)=0
        dτ  = (τφ(pτ, τg .+ HT) .- τφ(pτ, τg .- HT)) ./ (2HT)
        Lreg = mean(abs2, dτ)                                             # H¹ smoothness prior on τ
        return Lr + λ_d * Ld + λ_b * Lb + λ_reg * Lreg
    end

    Zygote.gradient(loss, pT, pτ)                            # warm up / compile
    t0 = time()
    for _ in 1:iters
        gT, gτ = Zygote.gradient(loss, pT, pτ)
        optT, pT = Optimisers.update(optT, pT, gT)
        optτ, pτ = Optimisers.update(optτ, pτ, gτ)
    end
    elapsed = time() - t0

    # ── recovery metrics on a dense τ grid ──────────────────────────────────
    τe = Float32.(range(0f0, 1f0; length = 601))
    τrec = vec(τφ(pτ, reshape(τe, 1, :))); τtru = vec(τstar(τe))
    peak_rec, ip = findmax(τrec); peak_tru = maximum(τtru)
    peak_err = abs(peak_rec - peak_tru) / peak_tru
    timing_h = abs(τe[ip] - τ0) * 30f0 * 24f0               # |Δτ| in hours over 30-day window
    rel_l2_τ = sqrt(sum((τrec .- τtru).^2) / sum(τtru.^2))  # whole-envelope relative L2
    gr = Float32.(range(0.05f0, 1f0; length = 60))
    ζf = vec(repeat(gr, inner = 60)); τf = vec(repeat(gr, outer = 60))
    T̃p = vec(T̃net(pT, reshape(ζf,1,:), reshape(τf,1,:))); T̃e = T̃star(ζf, τf)
    l2 = sqrt(mean((T̃p .- T̃e).^2)); l2_C = l2 * ΔT_B
    return (; pT, pτ, Tm, sT, τm, sτ, T̃net, τφ, elapsed,
              peak_err, timing_h, rel_l2_τ, peak_rec, peak_tru, l2, l2_C,
              finalloss = loss(pT, pτ), τe, τrec, τtru)
end

println("="^70)
println("Task A — single-site inverse PINN: recover storm forcing τ(t)")
println("Site B (Davies Reef), H = $(Int(H)) m, Pe = $Pe, Wadv = $Wadv")
println("="^70)
@printf("nondim: ∂τT̃ = -%.1f·w(ζ)·∂ζT̃ + %.2f·∂ζζT̃ + S(ζ)·τ(t) ;  %d sensors × %d samples, σ_obs = %.3f (%.3f °C)\n",
        Wadv, Pe, length(Z_SENS), N_T, σ_obs, σ_obs * ΔT_B)
@printf("data signal: max|T̃*| = %.3f (%.2f °C), storm SNR ≈ %.1f×\n",
        maximum(abs, FD.T̃), maximum(abs, FD.T̃) * ΔT_B, maximum(abs, FD.T̃) / σ_obs)
@printf("weights: λ_r=1, λ_d=%.0f, λ_b=%.0f, λ_reg=%.0e\n\n", λ_d, λ_b, λ_reg)

cfg = (Ncol = parse(Int, get(ENV, "TASKA_NCOL", "4000")),
       iters = parse(Int, get(ENV, "TASKA_ITERS", "12000")),
       Tw = 32, Td = 4)

print("CPU run  (N=$(cfg.Ncol), $(cfg.iters) it, $(cfg.Tw)×$(cfg.Td))  … "); flush(stdout)
res = solve_inverse(cfg.Ncol; iters = cfg.iters, Tw = cfg.Tw, Td = cfg.Td)
@printf("%.0fs | loss %.2e | fwd L2 %.3f °C | peak err %.1f%% | timing %.1f h | env relL2 %.2f\n\n",
        res.elapsed, res.finalloss, res.l2_C, 100res.peak_err, res.timing_h, res.rel_l2_τ)

@printf("Forward L2 vs FD reference: %.3f °C (target < 0.05 °C: %s)\n",
        res.l2_C, res.l2_C < 0.05 ? "PASS" : "FAIL")
@printf("Recovered storm peak-amplitude error: %.1f%% (target < 15%%: %s)\n",
        100res.peak_err, res.peak_err < 0.15 ? "PASS" : "FAIL")
@printf("Storm-day timing error: %.1f h (target < 2 h: %s)\n",
        res.timing_h, res.timing_h < 2 ? "PASS" : "FAIL")

try
    using CairoMakie
    f = Figure(size = (920, 380))
    days = res.τe .* 30f0
    a1 = Axis(f[1,1], title = "Recovered storm forcing τ̂(t) vs truth",
              xlabel = "t (days)", ylabel = "τ(t)")
    lines!(a1, days, res.τtru; linestyle = :dash, linewidth = 3, label = "truth τ*(t)")
    lines!(a1, days, res.τrec; linewidth = 2, label = "recovered τ̂(t)")
    axislegend(a1, position = :rt)
    a2 = Axis(f[1,2], title = "Sensor temperatures: PINN vs noisy data",
              xlabel = "t (days)", ylabel = "T̃ (nondim)")
    obs = make_observations(); cols = (:navy, :teal, :seagreen, :darkorange, :firebrick)
    for (k, zc) in enumerate(Z_SENS)
        sl = (k-1)*N_T+1 : k*N_T
        td = collect(range(0f0, 30f0, length = N_T))
        scatter!(a2, td, vec(obs.T̃)[sl]; color = (cols[k], 0.25), markersize = 3)
        ζrow = fill(zc, 1, N_T); τrow = reshape(Float32.(range(0f0,1f0,length=N_T)),1,:)
        pinn = vec(res.T̃net(res.pT, ζrow, τrow))
        lines!(a2, td, pinn; color = cols[k], linewidth = 2)
    end
    figdir = get(ENV, "GPU_FIG_DIR", joinpath(@__DIR__, "..", "output")); isdir(figdir) || mkpath(figdir)
    save(joinpath(figdir, "task_a_inverse.png"), f)
    println("\nwrote output/task_a_inverse.png")
catch e
    println("(figure skipped: ", e, ")")
end
