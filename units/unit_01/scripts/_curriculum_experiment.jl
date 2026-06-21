#!/usr/bin/env julia
# ===========================================================================
# EXPERIMENT (not part of the course notes) — curriculum-learning variant of the
# Unit 1 inverse PINN. Kept in the repo for the record; deliberately underscore-
# prefixed so it is excluded from the index "Supporting files" manifest and is
# never referenced from unit_01.qmd.
#
# Idea (Yoni, 2026-06-21): a third recovery approach. Instead of optimising the
# data + physics losses jointly from step 0 (the shipped `train_inverse_pinn.jl`,
# "Method A"), train the field on the PDE physics FIRST — data weight pinned low
# while the IC + wave-equation residual settle a valid low-amplitude solution —
# then RAMP the gauge-data weight up so the fit to G1-G4 happens only once the
# network already obeys the SWE. "Learn the model first, ease the data in slowly"
# (curriculum learning; cf. Krishnapriyan et al. 2021).
#
# Implementation: identical architecture / step budget / weights as Method A,
# except LAM_DATA follows `lam_data_sched(step)` — held at LAM_DATA_LO for the
# first WARM_FRAC of steps, then linearly ramped to the full LAM_DATA by
# RAMP_FRAC. (LAM_DATA_LO is kept > 0 so the warm-up does not collapse to the
# trivial eta = 0 / psi = 0 solution that physics + zero-IC admit on their own.)
#
# RESULT — it did NOT help.  Recovered surge peak (truth = 0.450 m):
#     Method A (naive joint)      peak 0.055   (88% peak err,  rel-L2 0.884)
#     curriculum (this script)    peak 0.048   (89% peak err,  rel-L2 0.896)
#     adjoint / Tikhonov (B)      peak 0.530   (18% peak err,  rel-L2 0.428)
# The curriculum reaches the SAME flat-source minimum. At convergence the data
# term is already tiny (the 4 sparse gauges are fit) — through the wave operator
# a sharp source is not *needed* to match them (ill-posedness), and the smooth-
# MLP ansatz + smoothness prior actively prefer a flat psi. Training *order* does
# not change which minimum the same objective has.
#
# Follow-up diagnostic (smoothness sweep, no curriculum, constant LAM_DATA):
#     LAM_SMOOTH = 8e-4 -> peak 0.055 ; 8e-5 -> 0.098 ; 8e-6 -> 0.110 ; 0 -> 0.128
# i.e. even with the source-smoothness prior switched OFF the tanh-MLP only
# recovers ~0.13 m (72% err). The binding constraint is spectral bias + ill-
# posedness, fixed by Fourier features / hard constraints (Units 7-9), NOT by the
# optimisation schedule. Conclusion: curriculum is a sound tool for stiff/multi-
# scale *forward* problems, but the wrong lever for this sparse-data source
# recovery. Hence: not promoted into Unit 1.
#
# Run (CPU, ~2-3 min):  julia --project=. units/unit_01/scripts/_curriculum_experiment.jl
# Writes /tmp/psi_recovered_curriculum.csv (does not touch units/unit_01/data/).
# ===========================================================================

using Lux, ComponentArrays, Optimisers, Zygote, Random, Printf
using DelimitedFiles, CSV, DataFrames, JSON3, Statistics

const HERE     = @__DIR__
const DATA_DIR = joinpath(HERE, "..", "data")
const FIG_DIR  = joinpath(HERE, "..", "figures")

# --- Load problem ---------------------------------------------------------

H_grid = readdlm(joinpath(DATA_DIR, "bay_bathymetry.csv"), ',', Float64)
mask   = readdlm(joinpath(DATA_DIR, "bay_mask.csv"),       ',', Int)
H_grid[isnan.(H_grid)] .= 5.0       # placeholder under land

gauges    = CSV.read(joinpath(DATA_DIR, "bay_gauges.csv"),       DataFrame)
river     = CSV.read(joinpath(DATA_DIR, "river_source.csv"),     DataFrame)
obs       = CSV.read(joinpath(DATA_DIR, "gauges_observed.csv"),  DataFrame)
psi_truth = CSV.read(joinpath(DATA_DIR, "psi_truth.csv"),        DataFrame)
meta      = JSON3.read(read(joinpath(DATA_DIR, "bay_meta.json"), String))

const NX  = Int(meta["nx"])
const NY  = Int(meta["ny"])
const DX  = Float64(meta["dx_m"])
const DY  = Float64(meta["dy_m"])
const LX  = NX * DX
const LY  = NY * DY
const TEND = obs.t[end]

const GVAL = 9.81

# --- Coordinates of source + gauges in metres (cell-centre) ---------------

xc(i) = (i - 0.5) * DX
yc(j) = (j - 0.5) * DY

const X_R = xc(river.ix[1])
const Y_R = yc(river.iy[1])

const GAUGE_XY = [(xc(gauges.ix[k]), yc(gauges.iy[k])) for k in 1:nrow(gauges)]

# --- Sub-sample gauge observations (every 6 min) --------------------------

const DATA_STRIDE = 15                   # 15 × 12s = 180s = 3 min
const T_DATA = obs.t[1:DATA_STRIDE:end]
const N_DATA = length(T_DATA)
const GAUGE_OBS = hcat(
    obs.G1[1:DATA_STRIDE:end],
    obs.G2[1:DATA_STRIDE:end],
    obs.G3[1:DATA_STRIDE:end],
    obs.G4[1:DATA_STRIDE:end],
)   # N_DATA × 4

println("N_DATA = ", N_DATA, " (gauge samples every 6 min × 4 gauges = ",
        N_DATA * 4, " data pts)")

# --- Network --------------------------------------------------------------
#
# Inputs scaled to [-1, 1]. Output is η in metres.

nx_norm(x) = 2 * x / LX  - 1
ny_norm(y) = 2 * y / LY  - 1
nt_norm(t) = 2 * t / TEND - 1

rng = Xoshiro(0xC011_C0FFEE)

chain = Chain(
    Dense(3,  64, tanh),
    Dense(64, 64, tanh),
    Dense(64, 64, tanh),
    Dense(64, 64, tanh),
    Dense(64,  1),
)
ps, st = Lux.setup(rng, chain)
ps     = Lux.fmap(x -> x isa AbstractArray ? Float64.(x) : x, ps)
θ0     = ComponentArray(ps)

function η_net(x::Real, y::Real, t::Real, θ)
    inp = [nx_norm(x); ny_norm(y); nt_norm(t)]
    out, _ = chain(inp, θ, st)
    return out[1]
end

function η_batch(X::AbstractMatrix, θ)
    out, _ = chain(X, θ, st)
    return vec(out)
end

# --- Bathymetry sampler (bilinear) ----------------------------------------

function H_at(x, y)
    fi = clamp(x / DX + 0.5, 1.0, Float64(NX))
    fj = clamp(y / DY + 0.5, 1.0, Float64(NY))
    i0 = clamp(floor(Int, fi), 1, NX - 1); i1 = i0 + 1
    j0 = clamp(floor(Int, fj), 1, NY - 1); j1 = j0 + 1
    fx = fi - i0;  fy = fj - j0
    return (1-fx)*(1-fy)*H_grid[j0, i0] + fx*(1-fy)*H_grid[j0, i1] +
           (1-fx)*fy*H_grid[j1, i0] + fx*fy*H_grid[j1, i1]
end

# --- Water-cell list (for collocation sampling) ----------------------------

const WATER_CELLS = [(i, j) for i in 2:NX-1, j in 2:NY-1 if mask[j, i] == 1]
println("# water cells (interior, excl. boundary): ", length(WATER_CELLS))

# --- Loss -----------------------------------------------------------------
#
# Build the input matrices we need:
#   • 4·N_data  gauge-data points
#   • 7·N_phys  PDE-residual points (centre + ±x ±y ±t shifts)
#   • N_ic      IC points (t=0 and t=t_ic_eps for ∂_t η ≈ 0)
#
# We rebuild the collocation sample each step for stochastic coverage.

const N_PHYS = 384      # collocation points per gradient step
const N_IC   = 96
# FD stencils in *normalized* coords (network inputs already live in [-1,1]).
# Physical shift = h_norm · (axis_length/2):
#   H_NX = 0.04 → 1000 m in x      H_NY = 0.02 → 950 m in y
#   H_NT = 0.006 → ~65 s in time
const H_NX = 0.04
const H_NY = 0.02
const H_NT = 0.006
const T_IC_EPS = 8.0    # small offset (s) for ∂_t η check at t=0

# Loss weights — data dominates, physics + IC + smoothness are soft.
# This vanilla setup is exactly the kind of PINN de Wolff et al. (2021)
# benchmarked: gets the *shape* of ψ(t) but underestimates the amplitude
# because the smooth-MLP ansatz can't represent a sharply-peaked source
# from sparse boundary data.  Units 7-9 fix that with Fourier features /
# causal training / hard BC enforcement.
const LAM_DATA   = 50.0
const LAM_PHYS   = 5.0e-4
const LAM_IC     = 2.0
const LAM_TIC    = 0.5
const LAM_SMOOTH = 8.0e-4

# Exclude collocation points within this radius of the source cell — the
# wave equation has a Dirichlet source there, so enforcing the homogeneous
# residual would fight the surge.
const R_SOURCE_M = 1500.0   # 1.5 km

# Pre-build gauge-data input matrix: 3 × (N_DATA * 4)
function gauge_input_matrix()
    K = nrow(gauges)
    X = Matrix{Float64}(undef, 3, N_DATA * K)
    for k in 1:K, n in 1:N_DATA
        xg, yg = GAUGE_XY[k]
        col = (k - 1) * N_DATA + n
        X[1, col] = nx_norm(xg)
        X[2, col] = ny_norm(yg)
        X[3, col] = nt_norm(T_DATA[n])
    end
    return X
end
const GAUGE_INPUT = gauge_input_matrix()
const GAUGE_TARGET = vec(GAUGE_OBS)             # (N_DATA * 4,)
@assert size(GAUGE_INPUT, 2) == length(GAUGE_TARGET)

# Build a collocation matrix (3, 7·N) for the 7-point FD wave-equation stencil.
# Shifts are applied directly in *normalized* input space, so we don't have to
# worry about chain-rule factors when computing the residual.  Per-point K_x
# and K_y coefficients (which encode local depth via H) are pre-computed.
function colloc_inputs(N)
    X = Matrix{Float64}(undef, 3, 7 * N)
    Kx = Vector{Float64}(undef, N)
    Ky = Vector{Float64}(undef, N)
    n = 0
    @inbounds while n < N
        ci = rand(1:length(WATER_CELLS))
        i, j = WATER_CELLS[ci]
        x = xc(i) + (rand() - 0.5) * DX
        y = yc(j) + (rand() - 0.5) * DY
        # reject points too close to the source cell — wave eq doesn't hold there
        if hypot(x - X_R, y - Y_R) < R_SOURCE_M
            continue
        end
        t = rand() * TEND
        H = H_at(x, y)
        n += 1
        Kx[n] = GVAL * H * TEND^2 / LX^2
        Ky[n] = GVAL * H * TEND^2 / LY^2

        nx0 = nx_norm(x); ny0 = ny_norm(y); nt0 = nt_norm(t)
        shifts = ((0.0,  0.0,  0.0),
                  (+H_NX, 0.0, 0.0), (-H_NX, 0.0, 0.0),
                  (0.0, +H_NY, 0.0), (0.0, -H_NY, 0.0),
                  (0.0,  0.0, +H_NT), (0.0, 0.0, -H_NT))
        for (k, s) in enumerate(shifts)
            col = (n - 1) * 7 + k
            X[1, col] = nx0 + s[1]
            X[2, col] = ny0 + s[2]
            X[3, col] = nt0 + s[3]
        end
    end
    return X, Kx, Ky
end

# Initial-condition input matrix — η and ∂_t η both = 0 at t = 0 at water cells
function ic_inputs(N)
    Xz = Matrix{Float64}(undef, 3, N)
    Xt = Matrix{Float64}(undef, 3, N)
    for n in 1:N
        ci = rand(1:length(WATER_CELLS))
        i, j = WATER_CELLS[ci]
        x = xc(i) + (rand() - 0.5) * DX
        y = yc(j) + (rand() - 0.5) * DY
        Xz[1, n] = nx_norm(x); Xz[2, n] = ny_norm(y); Xz[3, n] = nt_norm(0.0)
        Xt[1, n] = nx_norm(x); Xt[2, n] = ny_norm(y); Xt[3, n] = nt_norm(T_IC_EPS)
    end
    return Xz, Xt
end

# Smoothness regulariser on ψ_recovered(t) = η_θ(x_r, y_r, t).
# 3-point stencil in normalized time at the source location.
function smooth_inputs(N)
    X = Matrix{Float64}(undef, 3, 3 * N)
    nxR = nx_norm(X_R); nyR = ny_norm(Y_R)
    for n in 1:N
        t = clamp(rand() * TEND, 4 * H_NT * TEND/2, TEND - 4 * H_NT * TEND/2)
        nt0 = nt_norm(t)
        for (k, dnt) in enumerate((-H_NT, 0.0, +H_NT))
            col = (n - 1) * 3 + k
            X[1, col] = nxR
            X[2, col] = nyR
            X[3, col] = nt0 + dnt
        end
    end
    return X
end

# Total loss — collocation inputs are pre-generated (non-diff) and passed in.
# Residuals are computed in *normalized* coords so the wave-equation residual
# is O(1) when the equation is unsatisfied:
#
#   r̃ = ∂_ττ η  -  K_x · ∂_ξξ η  -  K_y · ∂_ζζ η
#   with  K_x = g H T² / Lx²,    K_y = g H T² / Ly².
function total_loss(θ, Xphys, Kx, Ky, Xz, Xt, Xsm, lam_data, lam_smooth)
    η_g = η_batch(GAUGE_INPUT, θ)
    L_data = mean((η_g .- GAUGE_TARGET).^2)

    η_p_all = η_batch(Xphys, θ)
    η_p = reshape(η_p_all, 7, :)
    η_ξξ = (η_p[2, :] .- 2 .* η_p[1, :] .+ η_p[3, :]) ./ H_NX^2
    η_ζζ = (η_p[4, :] .- 2 .* η_p[1, :] .+ η_p[5, :]) ./ H_NY^2
    η_ττ = (η_p[6, :] .- 2 .* η_p[1, :] .+ η_p[7, :]) ./ H_NT^2
    res = η_ττ .- Kx .* η_ξξ .- Ky .* η_ζζ
    L_phys = mean(res .^ 2)

    η_z = η_batch(Xz, θ)
    η_e = η_batch(Xt, θ)
    L_ic  = mean(η_z .^ 2)
    L_tic = mean(((η_e .- η_z) ./ T_IC_EPS) .^ 2)

    η_s = reshape(η_batch(Xsm, θ), 3, :)
    ψ_ττ = (η_s[1, :] .- 2 .* η_s[2, :] .+ η_s[3, :]) ./ H_NT^2
    L_smooth = mean(ψ_ττ .^ 2)

    return lam_data   * L_data +
           LAM_PHYS   * L_phys +
           LAM_IC     * L_ic   +
           LAM_TIC    * L_tic  +
           lam_smooth * L_smooth,
           (L_data, L_phys, L_ic, L_tic, L_smooth)
end

function eval_loss(θ)
    Xphys, Kx, Ky = colloc_inputs(N_PHYS)
    Xz, Xt        = ic_inputs(N_IC)
    Xsm           = smooth_inputs(96)
    return total_loss(θ, Xphys, Kx, Ky, Xz, Xt, Xsm, LAM_DATA, LAM_SMOOTH)
end

# --- Training -------------------------------------------------------------

println("\nWarming up loss…")
l0, parts0 = eval_loss(θ0)
@printf("  initial loss = %.4e   (data=%.3e phys=%.3e ic=%.3e tic=%.3e sm=%.3e)\n",
        l0, parts0...)

# --- Adam phase
const N_STEPS = 8000
const LR0     = 4.0e-3

# --- CURRICULUM: physics + IC settle a valid (low-amplitude) wave solution
# first (data weight pinned low), then the gauge-data weight ramps up so the
# fit to G1–G4 happens only once the field already obeys the SWE. "Learn the
# model first, ease the data in slowly."
const LAM_DATA_LO = 1.0      # data weight during the warm-up
const WARM_FRAC   = 0.30     # first 30% of steps: data pinned low
const RAMP_FRAC   = 0.80     # reach full LAM_DATA by 80% of steps
function lam_data_sched(step)
    f = step / N_STEPS
    f <= WARM_FRAC && return LAM_DATA_LO
    f >= RAMP_FRAC && return LAM_DATA
    return LAM_DATA_LO + (LAM_DATA - LAM_DATA_LO) * (f - WARM_FRAC) / (RAMP_FRAC - WARM_FRAC)
end

function train_adam!(θ0)
    opt = Optimisers.Adam(LR0)
    opt_state = Optimisers.setup(opt, θ0)
    θ = θ0
    println("Adam: ", N_STEPS, " steps @ lr=", LR0)
    tic = time()
    log_every = 250
    for step in 1:N_STEPS
        Xphys, Kx, Ky = colloc_inputs(N_PHYS)
        Xz, Xt        = ic_inputs(N_IC)
        Xsm           = smooth_inputs(96)

        lam_data = lam_data_sched(step)
        g = first(Zygote.gradient(
            θ -> first(total_loss(θ, Xphys, Kx, Ky, Xz, Xt, Xsm, lam_data, LAM_SMOOTH)),
            θ))
        opt_state, θ = Optimisers.update(opt_state, θ, g)

        if step == 3000
            Optimisers.adjust!(opt_state, LR0 * 0.4)
        elseif step == 5500
            Optimisers.adjust!(opt_state, LR0 * 0.15)
        end

        if step % log_every == 0 || step == 1
            l, parts = total_loss(θ, Xphys, Kx, Ky, Xz, Xt, Xsm, lam_data, LAM_SMOOTH)
            @printf("  step %4d  λ_d=%5.1f  loss = %.4e   (data=%.3e phys=%.3e ic=%.3e tic=%.3e sm=%.3e)   %.1fs\n",
                    step, lam_data, l, parts..., time() - tic)
        end
    end
    @printf("Adam phase done (%.1f s)\n", time() - tic)
    return θ
end

θ = train_adam!(θ0)

# --- Recover ψ(t) ---------------------------------------------------------
ψ_recovered = [η_net(X_R, Y_R, t, θ) for t in psi_truth.t]

# Recovered gauge predictions for diagnostic
gauge_pred = zeros(Float64, length(psi_truth.t), 4)
for (k, (xg, yg)) in enumerate(GAUGE_XY)
    for (n, t) in enumerate(psi_truth.t)
        gauge_pred[n, k] = η_net(xg, yg, t, θ)
    end
end

# --- Save outputs ---------------------------------------------------------

CSV.write(joinpath("/tmp", "psi_recovered_curriculum.csv"),
          DataFrame(t = psi_truth.t,
                    psi_truth = psi_truth.psi,
                    psi_recovered = ψ_recovered))

let df = DataFrame(t = psi_truth.t)
    for (k, gid) in enumerate(gauges.gauge_id)
        df[!, Symbol("$(gid)_pred")] = gauge_pred[:, k]
    end
    CSV.write(joinpath("/tmp", "gauges_pinn_pred_curriculum.csv"), df)
end

# Diagnostic numbers
err_l2  = sqrt(mean((ψ_recovered .- psi_truth.psi) .^ 2))
err_rel = err_l2 / sqrt(mean(psi_truth.psi .^ 2))
peak_t  = ψ_recovered[argmax(abs.(ψ_recovered))]
peak_o  = psi_truth.psi[argmax(abs.(psi_truth.psi))]
@printf("\nψ recovery:\n")
@printf("  L2 err = %.4f m   (relative: %.1f %%)\n", err_l2, 100 * err_rel)
@printf("  truth peak = %+.3f m,  recovered peak = %+.3f m\n", peak_o, peak_t)
@printf("\nWrote data/psi_recovered.csv and data/gauges_pinn_pred.csv\n")
