# ===========================================================================
# Task B — two-site JOINT inverse PINN, CPU sub-scale prototype.
#
# Spec: unit_09 §9.10 / unit_10 §10.3. Recover ONE shared storm wind-stress
# envelope τ(t) from TWO moorings at once — Site A (Cleveland Bay, H = 15 m,
# diffusion-dominated) and Site B (Davies Reef, H = 60 m, advection + diffusion,
# Pe ~ 1). The headline claim, and the thing this script actually measures:
# the cross-site constraint beats either site's decoupled inverse, because each
# site sees a *different observable consequence* of the same physical stress, so
# a single shared τ(t) must satisfy both at once.
#
# Recipe. This is the SAME proven static-weight recipe as Task A
# (task_a_inverse_pinn.jl) — large data weight λ_d, tiny H¹ smoothing λ_reg,
# hard IC + hard deep BC baked into θ = ζ·τ·N, FD-in-input derivative stencil
# (Unit 5 §5.3) — extended minimally to two sites that share ONE τ-network.
# The "modern toolkit" (Fourier features, adaptive weighting, causal training)
# is for the GPU full-scale 3-site launch; at this CPU sub-scale the simpler
# static recipe is what converges cleanly, so that is what we run and report.
#
# Run:  julia --project=@pinn units/unit_10/scripts/task_b_subscale_prototype.jl
# (CPU-only is fine; ~minutes. eval:false in the .qmd.)
# Smoke test:  TASKB_NCOL=200 TASKB_ITERS=200 julia --project=. <this file>
# ===========================================================================

using Lux, Optimisers, Zygote, Random, Printf, Statistics

# ── per-site nondimensional column models ───────────────────────────────────
# Each site: ∂τθ = -Wadv·w(ζ)·∂ζθ + Pe·∂ζζθ + S(ζ)·τ(t),  ζ∈[0,1], τ∈[0,1]
#   θ(0,τ)=0 (deep reservoir), ∂ζθ(1,τ)=0 (insulating surface), θ(ζ,0)=0.
# Site A is diffusion-dominated (Wadv=0); Site B adds surface-intensified
# upwelling (Wadv>0) so its Péclet is ~1 and the storm leaves a different mark.
struct Site
    name::String
    Pe::Float32       # diffusion number
    Wadv::Float32     # advection number (0 = pure diffusion)
    ℓS::Float32       # heating decay scale below surface
    ΔT::Float32       # °C scale, for reporting
    H::Float32        # column depth (m)
    zsens::Vector{Float32}   # sensor depths (m, positive down)
end
S_shape(s::Site, ζ) = exp.((ζ .- 1f0) ./ s.ℓS)               # surface-weighted heating shape
w_shape(ζ)          = ζ                                       # upwelling: 0 at depth, max at surface
ζsens(s::Site)      = Float32[(s.H - d) / s.H for d in s.zsens]

const SITE_A = Site("A — Cleveland Bay", 1.5f0, 0.0f0, 0.20f0, 2.5f0, 15f0, Float32[1,4,8,12,14])
const SITE_B = Site("B — Davies Reef",   1.0f0, 1.2f0, 0.25f0, 3.0f0, 60f0, Float32[2,10,25,45,58])

# ── shared ground-truth storm signal τ*(t) (both sites feel the SAME stress) ──
const τ0 = 1f0/3f0; const σstm = 0.05f0; const A_stm = 4.0f0
τstar(t) = A_stm .* exp.(-((t .- τ0) ./ σstm) .^ 2)

# ── per-site FD reference solve → synthetic mooring data ─────────────────────
const NZ = 61
function fd_reference(s::Site; NT_FD = 40001)
    ζ = Float32.(range(0f0, 1f0; length = NZ)); dζ = ζ[2] - ζ[1]
    t = Float32.(range(0f0, 1f0; length = NT_FD)); dt = t[2] - t[1]
    S = S_shape(s, ζ); w = w_shape(ζ)
    θ = zeros(Float32, NZ, NT_FD)
    @inbounds for k in 1:NT_FD-1
        f = τstar(t[k]); col = @view θ[:, k]
        for i in 2:NZ-1
            lap = (col[i+1] - 2col[i] + col[i-1]) / dζ^2
            adv = w[i] * (col[i+1] - col[i-1]) / (2dζ)        # central; upwelling carries deep water up
            θ[i, k+1] = col[i] + dt * (-s.Wadv * adv + s.Pe * lap + S[i] * f)
        end
        lapN = (2col[NZ-1] - 2col[NZ]) / dζ^2                  # insulating surface ghost
        advN = w[NZ] * (col[NZ] - col[NZ-1]) / dζ
        θ[NZ, k+1] = col[NZ] + dt * (-s.Wadv * advN + s.Pe * lapN + S[NZ] * f)
        θ[1, k+1] = 0f0
    end
    (; ζ, t, θ, NT_FD)
end

function θstar_of(FD)
    (ζv, τv) -> begin
        out = similar(ζv)
        @inbounds for n in eachindex(ζv)
            iz = clamp(round(Int, ζv[n] * (NZ - 1)) + 1, 1, NZ)
            ft = clamp(τv[n] * (FD.NT_FD - 1), 0f0, Float32(FD.NT_FD - 1))
            k = clamp(floor(Int, ft) + 1, 1, FD.NT_FD - 1); fr = ft - (k - 1)
            out[n] = (1 - fr) * FD.θ[iz, k] + fr * FD.θ[iz, k+1]
        end
        out
    end
end

const N_T = 720; const σ_obs = 0.005f0
function observations(s::Site, θstar; seed = 20260617)
    τs = Float32.(range(0f0, 1f0; length = N_T))
    zc = ζsens(s)
    ζcol = repeat(zc, inner = N_T); τcol = repeat(τs, outer = length(zc))
    noisy = θstar(ζcol, τcol) .+ σ_obs .* randn(Xoshiro(seed), Float32, length(ζcol))
    (; ζ = reshape(ζcol, 1, :), τ = reshape(τcol, 1, :), θ = reshape(noisy, 1, :))
end

# ── networks (plain (ζ,τ) input, as Task A — no Fourier at sub-scale) ────────
make_T(w, d) = Chain(Dense(2 => w, tanh), [Dense(w => w, tanh) for _ in 1:d-1]..., Dense(w => 1))
make_τ()     = Chain(Dense(1 => 32, tanh), Dense(32 => 32, tanh), Dense(32 => 1))

const HZ = 2f-3; const HT = 2f-3
# Same operating point as Task A: large data weight, tiny H¹ smoothing.
const λ_d = 6000f0; const λ_b = 10f0; const λ_reg = 1f-5

# ── joint inverse over a list of sites (1 = single-site, 2 = joint) ──────────
# One T-network per site (separate params); ONE shared τ-network across sites.
function solve_joint(sites, FDs, θstars; Ncol, iters, Tw, Td, seed = 1)
    Tms = [make_T(Tw, Td) for _ in sites]
    τm  = make_τ()
    pTs = Vector{Any}(undef, length(sites)); sTs = Vector{Any}(undef, length(sites))
    for (i, Tm) in enumerate(Tms)
        pTs[i], sTs[i] = Lux.setup(Xoshiro(seed + i), Tm)
    end
    pτ, sτ = Lux.setup(Xoshiro(seed + 100), τm)
    optTs = [Optimisers.setup(Adam(1f-3), p) for p in pTs]
    optτ  = Optimisers.setup(Adam(3f-3), pτ)

    rng = Xoshiro(seed + 500)
    ζc = [rand(rng, Float32, 1, Ncol) for _ in sites]
    τc = [rand(rng, Float32, 1, Ncol) for _ in sites]
    τb = [rand(rng, Float32, 1, Ncol ÷ 5) for _ in sites]
    ζ1 = [ones(Float32, 1, Ncol ÷ 5) for _ in sites]
    obs = [observations(s, θstars[i]) for (i, s) in enumerate(sites)]
    τg = reshape(Float32.(range(0f0, 1f0; length = 400)), 1, :)

    NT_(Tm, p, sT, ζ, τ) = first(Tm(vcat(ζ, τ), p, sT))
    θnet(Tm, p, sT, ζ, τ) = ζ .* τ .* NT_(Tm, p, sT, ζ, τ)    # hard IC + hard deep BC
    τφ(q, τ) = first(τm(τ, q, sτ))                            # shared recovered forcing

    function loss(pTs, pτ)
        L = 0f0
        for i in eachindex(sites)
            s = sites[i]; Tm = Tms[i]; sT = sTs[i]; ζi = ζc[i]; τi = τc[i]; pTi = pTs[i]
            θt  = (θnet(Tm, pTi, sT, ζi, τi .+ HT) .- θnet(Tm, pTi, sT, ζi, τi .- HT)) ./ (2HT)
            θz  = (θnet(Tm, pTi, sT, ζi .+ HZ, τi) .- θnet(Tm, pTi, sT, ζi .- HZ, τi)) ./ (2HZ)
            θzz = (θnet(Tm, pTi, sT, ζi .+ HZ, τi) .- 2f0 .* θnet(Tm, pTi, sT, ζi, τi) .+ θnet(Tm, pTi, sT, ζi .- HZ, τi)) ./ HZ^2
            Lr  = mean(abs2, θt .+ s.Wadv .* w_shape(ζi) .* θz .- s.Pe .* θzz .- S_shape(s, ζi) .* τφ(pτ, τi))
            Ld  = mean(abs2, θnet(Tm, pTi, sT, obs[i].ζ, obs[i].τ) .- obs[i].θ)
            θzs = (θnet(Tm, pTi, sT, ζ1[i], τb[i]) .- θnet(Tm, pTi, sT, ζ1[i] .- HZ, τb[i])) ./ HZ
            Lb  = mean(abs2, θzs)
            L += Lr + λ_d * Ld + λ_b * Lb
        end
        dτ = (τφ(pτ, τg .+ HT) .- τφ(pτ, τg .- HT)) ./ (2HT)
        L += λ_reg * mean(abs2, dτ)
        return L
    end

    Zygote.gradient(loss, pTs, pτ)                # warm up / compile
    t0 = time()
    for _ in 1:iters
        gTs, gτ = Zygote.gradient(loss, pTs, pτ)
        for i in eachindex(sites)
            optTs[i], pTs[i] = Optimisers.update(optTs[i], pTs[i], gTs[i])
        end
        optτ, pτ = Optimisers.update(optτ, pτ, gτ)
    end
    elapsed = time() - t0

    τe = Float32.(range(0f0, 1f0; length = 601))
    τrec = vec(τφ(pτ, reshape(τe, 1, :))); τtru = vec(τstar(τe))
    peak_rec, ip = findmax(τrec); peak_tru = maximum(τtru)
    peak_err = abs(peak_rec - peak_tru) / peak_tru
    timing_h = abs(τe[ip] - τ0) * 30f0 * 24f0
    rel_l2 = sqrt(sum((τrec .- τtru) .^ 2) / sum(τtru .^ 2))
    return (; elapsed, peak_err, timing_h, rel_l2, finalloss = loss(pTs, pτ),
              τe, τrec, τtru)
end

# ── run: Site A alone, Site B alone, then JOINT (A+B) ────────────────────────
println("="^74)
println("Task B sub-scale prototype — two-site JOINT inverse (Sites A + B)")
println("="^74)
FD_A = fd_reference(SITE_A); FD_B = fd_reference(SITE_B)
θs_A = θstar_of(FD_A); θs_B = θstar_of(FD_B)
@printf("Site A: Pe=%.1f, no advection, H=%.0f m, max|θ*|=%.2f (SNR≈%.0f)\n",
        SITE_A.Pe, SITE_A.H, maximum(abs, FD_A.θ), maximum(abs, FD_A.θ)/σ_obs)
@printf("Site B: Pe=%.1f, Wadv=%.1f,    H=%.0f m, max|θ*|=%.2f (SNR≈%.0f)\n",
        SITE_B.Pe, SITE_B.Wadv, SITE_B.H, maximum(abs, FD_B.θ), maximum(abs, FD_B.θ)/σ_obs)
println("shared storm τ*(t): peak ", A_stm, " at day ", round(Int, τ0*30), " of 30")
@printf("weights: λ_r=1, λ_d=%.0f, λ_b=%.0f, λ_reg=%.0e\n\n", λ_d, λ_b, λ_reg)

cfg = (Ncol = parse(Int, get(ENV, "TASKB_NCOL", "4000")),
       iters = parse(Int, get(ENV, "TASKB_ITERS", "12000")),
       Tw = 32, Td = 4)

print("Site A alone … "); flush(stdout)
rA = solve_joint([SITE_A], [FD_A], [θs_A]; cfg...)
@printf("%.0fs | peak err %.1f%% | timing %.1f h | env relL2 %.2f\n", rA.elapsed, 100rA.peak_err, rA.timing_h, rA.rel_l2)

print("Site B alone … "); flush(stdout)
rB = solve_joint([SITE_B], [FD_B], [θs_B]; cfg...)
@printf("%.0fs | peak err %.1f%% | timing %.1f h | env relL2 %.2f\n", rB.elapsed, 100rB.peak_err, rB.timing_h, rB.rel_l2)

print("JOINT A+B   … "); flush(stdout)
rJ = solve_joint([SITE_A, SITE_B], [FD_A, FD_B], [θs_A, θs_B]; cfg...)
@printf("%.0fs | peak err %.1f%% | timing %.1f h | env relL2 %.2f\n", rJ.elapsed, 100rJ.peak_err, rJ.timing_h, rJ.rel_l2)

println("\n", "-"^74)
@printf("RESULT: Site A alone %.1f%%, Site B alone %.1f%%, JOINT %.1f%% peak error.\n",
        100rA.peak_err, 100rB.peak_err, 100rJ.peak_err)
println(rJ.peak_err < min(rA.peak_err, rB.peak_err) ?
        "→ the cross-site constraint helps: joint beats either single site." :
        "→ joint did not beat both singles at this sub-scale budget (increase iters/Ncol).")

# ── recovered-τ overlay figure (the headline finding, made visual) ───────────
try
    using CairoMakie
    f = Figure(size = (820, 420))
    ax = Axis(f[1, 1];
              title  = "Recovered storm envelope τ̂(t): two sites jointly beat either alone",
              xlabel = "t (days)", ylabel = "τ(t)  (nondim heating)")
    days = rJ.τe .* 30f0
    lines!(ax, days, rJ.τtru; color = :black, linewidth = 3, linestyle = :dash, label = "truth τ*(t)")
    lines!(ax, days, rA.τrec; color = (:darkorange, 0.9), linewidth = 2,
           label = @sprintf("Site A alone — %.0f%% peak err", 100rA.peak_err))
    lines!(ax, days, rB.τrec; color = (:seagreen, 0.9), linewidth = 2,
           label = @sprintf("Site B alone — %.0f%% peak err", 100rB.peak_err))
    lines!(ax, days, rJ.τrec; color = (:firebrick, 1.0), linewidth = 3,
           label = @sprintf("JOINT A+B — %.0f%% peak err", 100rJ.peak_err))
    xlims!(ax, 5, 15); axislegend(ax; position = :rt, framevisible = true)
    figdir = get(ENV, "GPU_FIG_DIR", joinpath(@__DIR__, "..", "output"))
    isdir(figdir) || mkpath(figdir)
    save(joinpath(figdir, "task_b_subscale_recovery.png"), f)
    println("\nwrote output/task_b_subscale_recovery.png")
catch e
    println("(figure skipped: ", e, ")")
end
