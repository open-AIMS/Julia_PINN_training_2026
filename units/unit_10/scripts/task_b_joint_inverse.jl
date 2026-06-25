# ===========================================================================
# Task B — three-site JOINT inverse PINN, GPU full-scale.
#
# Spec: unit_09 §9.10 / unit_10 §10.3. Recover ONE shared storm wind-stress
# envelope τ(t) from THREE moorings at once — Site A (Cleveland Bay,
# diffusion-only), Site B (Davies Reef, advection ~ diffusion), Site C
# (Myrmidon Reef, advection-dominated). Each site sees a *different observable
# consequence* of the same physical stress, so a single shared τ(t) that must
# satisfy all three is pinned down far better than any single decoupled inverse.
#
# This is the GPU full-scale sibling of the CPU sub-scale prototype
# (task_b_subscale_prototype.jl): the SAME proven static-weight recipe
# (large λ_d, tiny H¹ λ_reg, hard IC + hard deep BC T̃ = ζ·τ·N, FD-in-input
# derivative stencil) at a wider network, more collocation, more steps, and a
# third advection-dominated site — run on the GPU where it is minutes not hours.
#
# Run on the GPU hub (@pinn carries Lux + LuxCUDA + CUDA + CairoMakie):
#   julia --project=@pinn units/unit_10/scripts/task_b_joint_inverse.jl
# Falls back to CPU automatically if no GPU (slow — for the CPU story use the
# 2-site prototype instead). Nothing here runs during `quarto render`.
# ===========================================================================

using Lux, LuxCUDA, CUDA, Optimisers, Zygote, Random, Printf, Statistics

# ── per-site nondimensional column models ───────────────────────────────────
# ∂τT̃ = -Wadv·w(ζ)·∂ζT̃ + Pe·∂ζζT̃ + S(ζ)·τ(t),  ζ∈[0,1], τ∈[0,1]
#   T̃(0,τ)=0 (deep reservoir), ∂ζT̃(1,τ)=0 (insulating surface), T̃(ζ,0)=0.
# "Pe" here is the (nondim) diffusion coefficient; the classical Péclet number
# advection/diffusion = Wadv/Pe rises A → C, so A is diffusion-only, C is
# advection-dominated — the three regimes the joint inverse must straddle.
struct Site
    name::String
    Pe::Float32; Wadv::Float32; ℓS::Float32; ΔT::Float32; H::Float32
    zsens::Vector{Float32}
end
S_shape(s::Site, ζ) = exp.((ζ .- 1f0) ./ s.ℓS)
w_shape(ζ)          = ζ
ζsens(s::Site)      = Float32[(s.H - d) / s.H for d in s.zsens]

const SITE_A = Site("A — Cleveland Bay", 1.5f0, 0.0f0, 0.20f0, 2.5f0, 100f0, Float32[2,10,25,50,90])
const SITE_B = Site("B — Davies Reef",   1.0f0, 1.2f0, 0.25f0, 3.0f0, 100f0, Float32[2,10,25,50,90])
const SITE_C = Site("C — Myrmidon Reef", 0.6f0, 2.5f0, 0.30f0, 3.5f0, 100f0, Float32[2,10,25,50,90])

# ── shared ground-truth storm τ*(t) ──────────────────────────────────────────
const τ0 = 1f0/3f0; const σstm = 0.05f0; const A_stm = 4.0f0
τstar(t) = A_stm .* exp.(-((t .- τ0) ./ σstm) .^ 2)

# ── per-site FD reference → synthetic mooring data ───────────────────────────
const NZ = 61
function fd_reference(s::Site; NT_FD = 40001)
    ζ = Float32.(range(0f0, 1f0; length = NZ)); dζ = ζ[2] - ζ[1]
    t = Float32.(range(0f0, 1f0; length = NT_FD)); dt = t[2] - t[1]
    S = S_shape(s, ζ); w = w_shape(ζ)
    T̃ = zeros(Float32, NZ, NT_FD)
    @inbounds for k in 1:NT_FD-1
        f = τstar(t[k]); col = @view T̃[:, k]
        for i in 2:NZ-1
            lap = (col[i+1] - 2col[i] + col[i-1]) / dζ^2
            adv = w[i] * (col[i+1] - col[i-1]) / (2dζ)
            T̃[i, k+1] = col[i] + dt * (-s.Wadv * adv + s.Pe * lap + S[i] * f)
        end
        lapN = (2col[NZ-1] - 2col[NZ]) / dζ^2
        advN = w[NZ] * (col[NZ] - col[NZ-1]) / dζ
        T̃[NZ, k+1] = col[NZ] + dt * (-s.Wadv * advN + s.Pe * lapN + S[NZ] * f)
        T̃[1, k+1] = 0f0
    end
    (; ζ, t, T̃, NT_FD)
end

function T̃star_of(FD)
    (ζv, τv) -> begin
        out = similar(ζv)
        @inbounds for n in eachindex(ζv)
            iz = clamp(round(Int, ζv[n] * (NZ - 1)) + 1, 1, NZ)
            ft = clamp(τv[n] * (FD.NT_FD - 1), 0f0, Float32(FD.NT_FD - 1))
            k = clamp(floor(Int, ft) + 1, 1, FD.NT_FD - 1); fr = ft - (k - 1)
            out[n] = (1 - fr) * FD.T̃[iz, k] + fr * FD.T̃[iz, k+1]
        end
        out
    end
end

# Mechanism partition (Step 5): integrate |advection|, |mixing|, |source| over
# the storm window at a diagnostic depth → real toy-model term breakdown.
function mechanism_partition(s::Site, FD, zdiag)
    ζ = FD.ζ; dζ = ζ[2]-ζ[1]; t = FD.t; dt = t[2]-t[1]
    S = S_shape(s, ζ); ζd = (s.H - zdiag)/s.H
    id = clamp(round(Int, ζd*(NZ-1))+1, 2, NZ-1)
    klo = max(2, floor(Int, (τ0-3σstm)*(FD.NT_FD-1))+1)
    khi = min(FD.NT_FD-1, ceil(Int, (τ0+3σstm)*(FD.NT_FD-1))+1)
    Iadv=0.0; Imix=0.0; Isrc=0.0
    @inbounds for k in klo:khi
        col = @view FD.T̃[:,k]
        lap = (col[id+1]-2col[id]+col[id-1])/dζ^2
        adv = ζ[id]*(col[id+1]-col[id-1])/(2dζ)
        Iadv += abs(s.Wadv*adv)*dt; Imix += abs(s.Pe*lap)*dt; Isrc += abs(S[id]*τstar(t[k]))*dt
    end
    tot = Iadv+Imix+Isrc
    (; adv=100Iadv/tot, mix=100Imix/tot, src=100Isrc/tot)
end

const N_T = 720; const σ_obs = 0.005f0
function observations(s::Site, T̃star; seed = 20260617)
    τs = Float32.(range(0f0, 1f0; length = N_T)); zc = ζsens(s)
    ζcol = repeat(zc, inner = N_T); τcol = repeat(τs, outer = length(zc))
    noisy = T̃star(ζcol, τcol) .+ σ_obs .* randn(Xoshiro(seed), Float32, length(ζcol))
    (; ζ = reshape(ζcol, 1, :), τ = reshape(τcol, 1, :), T̃ = reshape(noisy, 1, :))
end

# ── networks (plain (ζ,τ) input, static weights — the proven recipe) ─────────
make_T(w, d) = Chain(Dense(2 => w, tanh), [Dense(w => w, tanh) for _ in 1:d-1]..., Dense(w => 1))
make_τ()     = Chain(Dense(1 => 48, tanh), Dense(48 => 48, tanh), Dense(48 => 1))

const HZ = 2f-3; const HT = 2f-3
const λ_d = 6000f0; const λ_b = 10f0; const λ_reg = 1f-5

function solve_joint(sites, FDs, T̃stars, dev; Ncol, iters, Tw, Td, seed = 1)
    Tms = [make_T(Tw, Td) for _ in sites]; τm = make_τ()
    pTs = Vector{Any}(undef, length(sites)); sTs = Vector{Any}(undef, length(sites))
    for (i, Tm) in enumerate(Tms)
        p, st = Lux.setup(Xoshiro(seed + i), Tm); pTs[i] = dev(p); sTs[i] = dev(st)
    end
    pτ, sτ = Lux.setup(Xoshiro(seed + 100), τm); pτ = dev(pτ); sτ = dev(sτ)
    optTs = [Optimisers.setup(Adam(1f-3), p) for p in pTs]
    optτ  = Optimisers.setup(Adam(3f-3), pτ)

    rng = Xoshiro(seed + 500)
    ζc = [dev(rand(rng, Float32, 1, Ncol)) for _ in sites]
    τc = [dev(rand(rng, Float32, 1, Ncol)) for _ in sites]
    τb = [dev(rand(rng, Float32, 1, Ncol ÷ 5)) for _ in sites]
    ζ1 = [dev(ones(Float32, 1, Ncol ÷ 5)) for _ in sites]
    obs = [observations(s, T̃stars[i]) for (i, s) in enumerate(sites)]
    od = [(ζ = dev(o.ζ), τ = dev(o.τ), T̃ = dev(o.T̃)) for o in obs]
    τg = dev(reshape(Float32.(range(0f0, 1f0; length = 400)), 1, :))

    NT_(Tm, p, sT, ζ, τ) = first(Tm(vcat(ζ, τ), p, sT))
    T̃net(Tm, p, sT, ζ, τ) = ζ .* τ .* NT_(Tm, p, sT, ζ, τ)
    τφ(q, τ) = first(τm(τ, q, sτ))

    function loss(pTs, pτ)
        L = 0f0
        for i in eachindex(sites)
            s = sites[i]; Tm = Tms[i]; sT = sTs[i]; ζi = ζc[i]; τi = τc[i]; pTi = pTs[i]
            T̃t  = (T̃net(Tm, pTi, sT, ζi, τi .+ HT) .- T̃net(Tm, pTi, sT, ζi, τi .- HT)) ./ (2HT)
            T̃z  = (T̃net(Tm, pTi, sT, ζi .+ HZ, τi) .- T̃net(Tm, pTi, sT, ζi .- HZ, τi)) ./ (2HZ)
            T̃zz = (T̃net(Tm, pTi, sT, ζi .+ HZ, τi) .- 2f0 .* T̃net(Tm, pTi, sT, ζi, τi) .+ T̃net(Tm, pTi, sT, ζi .- HZ, τi)) ./ HZ^2
            Lr  = mean(abs2, T̃t .+ s.Wadv .* w_shape(ζi) .* T̃z .- s.Pe .* T̃zz .- S_shape(s, ζi) .* τφ(pτ, τi))
            Ld  = mean(abs2, T̃net(Tm, pTi, sT, od[i].ζ, od[i].τ) .- od[i].T̃)
            T̃zs = (T̃net(Tm, pTi, sT, ζ1[i], τb[i]) .- T̃net(Tm, pTi, sT, ζ1[i] .- HZ, τb[i])) ./ HZ
            Lb  = mean(abs2, T̃zs)
            L += Lr + λ_d * Ld + λ_b * Lb
        end
        dτ = (τφ(pτ, τg .+ HT) .- τφ(pτ, τg .- HT)) ./ (2HT)
        L += λ_reg * mean(abs2, dτ)
        return L
    end

    Zygote.gradient(loss, pTs, pτ); dev === identity || CUDA.synchronize()
    t0 = time()
    for _ in 1:iters
        gTs, gτ = Zygote.gradient(loss, pTs, pτ)
        for i in eachindex(sites)
            optTs[i], pTs[i] = Optimisers.update(optTs[i], pTs[i], gTs[i])
        end
        optτ, pτ = Optimisers.update(optτ, pτ, gτ)
    end
    dev === identity || CUDA.synchronize()
    elapsed = time() - t0

    τe = Float32.(range(0f0, 1f0; length = 601))
    τrec = vec(Array(τφ(pτ, dev(reshape(τe, 1, :))))); τtru = vec(τstar(τe))
    peak_rec, ip = findmax(τrec); peak_tru = maximum(τtru)
    peak_err = abs(peak_rec - peak_tru) / peak_tru
    timing_h = abs(τe[ip] - τ0) * 30f0 * 24f0
    rel_l2 = sqrt(sum((τrec .- τtru) .^ 2) / sum(τtru .^ 2))
    return (; elapsed, peak_err, timing_h, rel_l2, τe, τrec, τtru)
end

println("="^74)
println("Task B — three-site JOINT inverse PINN (Sites A + B + C), GPU full-scale")
println("="^74)
have_gpu = CUDA.functional()
dev = have_gpu ? gpu_device() : identity
@printf("GPU available: %s%s\n", have_gpu, have_gpu ? "  ($(CUDA.name(CUDA.device())))" : " — running on CPU (slow)")

FD_A = fd_reference(SITE_A); FD_B = fd_reference(SITE_B); FD_C = fd_reference(SITE_C)
T̃s_A = T̃star_of(FD_A); T̃s_B = T̃star_of(FD_B); T̃s_C = T̃star_of(FD_C)
for (s, FD) in ((SITE_A, FD_A), (SITE_B, FD_B), (SITE_C, FD_C))
    @printf("Site %-18s Pe=%.1f Wadv=%.1f  max|T̃*|=%.2f (SNR≈%.0f)\n",
            s.name, s.Pe, s.Wadv, maximum(abs, FD.T̃), maximum(abs, FD.T̃)/σ_obs)
end
@printf("weights: λ_r=1, λ_d=%.0f, λ_b=%.0f, λ_reg=%.0e\n", λ_d, λ_b, λ_reg)

println("\nmechanism partition over the storm window (toy-model term magnitudes):")
println("site                | z_diag | adv%  mix%  src%")
for (s, FD, zd) in ((SITE_A, FD_A, 10f0), (SITE_B, FD_B, 30f0), (SITE_C, FD_C, 50f0))
    p = mechanism_partition(s, FD, zd)
    @printf("%-19s |  %3.0fm  | %4.0f  %4.0f  %4.0f\n", s.name, zd, p.adv, p.mix, p.src)
end
println()

cfg = (Ncol = parse(Int, get(ENV, "TASKB_NCOL", "4000")),
       iters = parse(Int, get(ENV, "TASKB_ITERS", "3000")),
       Tw = 64, Td = 5)

print("Site A alone … "); flush(stdout)
rA = solve_joint([SITE_A], [FD_A], [T̃s_A], dev; cfg...)
@printf("%.0fs | peak err %.1f%% | timing %.1f h | env relL2 %.2f\n", rA.elapsed, 100rA.peak_err, rA.timing_h, rA.rel_l2)
print("Site B alone … "); flush(stdout)
rB = solve_joint([SITE_B], [FD_B], [T̃s_B], dev; cfg...)
@printf("%.0fs | peak err %.1f%% | timing %.1f h | env relL2 %.2f\n", rB.elapsed, 100rB.peak_err, rB.timing_h, rB.rel_l2)
print("Site C alone … "); flush(stdout)
rC = solve_joint([SITE_C], [FD_C], [T̃s_C], dev; cfg...)
@printf("%.0fs | peak err %.1f%% | timing %.1f h | env relL2 %.2f\n", rC.elapsed, 100rC.peak_err, rC.timing_h, rC.rel_l2)
print("JOINT A+B+C  … "); flush(stdout)
rJ = solve_joint([SITE_A, SITE_B, SITE_C], [FD_A, FD_B, FD_C], [T̃s_A, T̃s_B, T̃s_C], dev; cfg...)
@printf("%.0fs | peak err %.1f%% | timing %.1f h | env relL2 %.2f\n", rJ.elapsed, 100rJ.peak_err, rJ.timing_h, rJ.rel_l2)

println("\n", "-"^74)
@printf("RESULT: A %.1f%%, B %.1f%%, C %.1f%% alone  vs  JOINT %.1f%% peak error.\n",
        100rA.peak_err, 100rB.peak_err, 100rC.peak_err, 100rJ.peak_err)
println(rJ.peak_err < min(rA.peak_err, rB.peak_err, rC.peak_err) ?
        "→ the cross-site constraint helps: joint beats every single site." :
        "→ joint did not beat all singles at this budget (increase iters/Ncol).")

try
    using CairoMakie
    f = Figure(size = (860, 440))
    ax = Axis(f[1, 1]; title = "Three-site joint recovery of the storm envelope τ̂(t)",
              xlabel = "t (days)", ylabel = "τ(t)  (nondim heating)")
    days = rJ.τe .* 30f0
    lines!(ax, days, rJ.τtru; color = :black, linewidth = 3, linestyle = :dash, label = "truth τ*(t)")
    lines!(ax, days, rA.τrec; color = (:darkorange, 0.8), linewidth = 1.8, label = @sprintf("A alone %.0f%%", 100rA.peak_err))
    lines!(ax, days, rB.τrec; color = (:seagreen, 0.8),  linewidth = 1.8, label = @sprintf("B alone %.0f%%", 100rB.peak_err))
    lines!(ax, days, rC.τrec; color = (:steelblue, 0.8), linewidth = 1.8, label = @sprintf("C alone %.0f%%", 100rC.peak_err))
    lines!(ax, days, rJ.τrec; color = (:firebrick, 1.0), linewidth = 3,   label = @sprintf("JOINT %.0f%%", 100rJ.peak_err))
    xlims!(ax, 5, 15); axislegend(ax; position = :rt)
    figdir = get(ENV, "GPU_FIG_DIR", joinpath(@__DIR__, "..", "output")); isdir(figdir) || mkpath(figdir)
    save(joinpath(figdir, "task_b_joint_recovery.png"), f)
    println("wrote output/task_b_joint_recovery.png")
catch e
    println("(figure skipped: ", e, ")")
end
