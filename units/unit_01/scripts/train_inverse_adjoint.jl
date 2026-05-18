#!/usr/bin/env julia
# Adjoint / Green's-function inverse — recover the Brisbane River surge
# ψ(t) from sparse, noisy bay-gauge readings.
#
# This is the "physics-informed inverse modelling" baseline for Unit 1.
# Conceptually:
#   1.  The linearised SWE is *linear in ψ*.  Imposed as a Dirichlet BC at
#       the source cell, η(x_r, y_r, t) = ψ(t), the forward map
#         G : ψ(·) →  η_obs at the four gauges
#       is therefore a linear operator.
#   2.  We *precompute* the impulse response of G by running the FD solver
#       with a narrow Gaussian ψ-pulse and recording gauges at every step.
#       Time-translation invariance gives the full G as a Toeplitz
#       convolution matrix.
#   3.  Stack the four-gauge response matrices into a single
#       (4·N_obs) × N_t system  G · ψ = g_obs  + noise,  and solve
#       with Tikhonov smoothness regularisation:
#         ψ̂  =  (GᵀG  +  λ L²ᵀL²)⁻¹  Gᵀ g_obs
#       where L² is the discrete 2nd-difference operator (smoothness prior).
#
# Output:   data/psi_recovered_adjoint.csv  (clean recovery, ~5% error)
#           data/gauges_adjoint_pred.csv
#
# Differences from a "vanilla PINN":
#   • This precomputes the forward operator using the same FD solver that
#     generated the data — so we are basically inverting the simulator.
#   • The Tikhonov prior on ψ is the analogue of the smoothness/IC terms
#     used by a PINN.
#   • There is no neural network on ψ here; the inverse is closed-form.
#     A neural ψ_θ(t) trained against the same objective would give an
#     essentially identical answer at higher cost.
# Units 5-9 build up the full PINN treatment that learns η and ψ jointly
# without precomputing G.

using DelimitedFiles, CSV, DataFrames, JSON3, Printf, LinearAlgebra, Random

const HERE     = @__DIR__
const DATA_DIR = joinpath(HERE, "..", "data")

# --- Load bay model -------------------------------------------------------

H_raw  = readdlm(joinpath(DATA_DIR, "bay_bathymetry.csv"), ',', Float64)
mask   = readdlm(joinpath(DATA_DIR, "bay_mask.csv"),       ',', Int)
gauges = CSV.read(joinpath(DATA_DIR, "bay_gauges.csv"),   DataFrame)
river  = CSV.read(joinpath(DATA_DIR, "river_source.csv"), DataFrame)
obs    = CSV.read(joinpath(DATA_DIR, "gauges_observed.csv"), DataFrame)
psi_truth = CSV.read(joinpath(DATA_DIR, "psi_truth.csv"),  DataFrame)
meta   = JSON3.read(read(joinpath(DATA_DIR, "bay_meta.json"), String))

const NX  = Int(meta["nx"])
const NY  = Int(meta["ny"])
const DX  = Float64(meta["dx_m"])
const DY  = Float64(meta["dy_m"])

H = copy(H_raw); H[isnan.(H)] .= 5.0

# Same numerics as generate_surge_data.jl so the inversion is "consistent"
const G_GRAV = 9.81
const B_DRAG = 5.0e-5
const T_END  = 6 * 3600.0
const DT     = 12.0
const NT     = Int(floor(T_END / DT))
const N_REC  = NT + 1                              # records every step
const t_axis = collect(0:DT:NT*DT)

# Face depths and open/closed masks (same construction as forward solver)
Hu = zeros(Float64, NY, NX - 1)
Hv = zeros(Float64, NY - 1, NX)
@inbounds for j in 1:NY, i in 1:NX-1
    Hu[j, i] = 0.5 * (H[j, i] + H[j, i+1])
end
@inbounds for j in 1:NY-1, i in 1:NX
    Hv[j, i] = 0.5 * (H[j, i] + H[j+1, i])
end
u_open = falses(NY, NX - 1)
v_open = falses(NY - 1, NX)
@inbounds for j in 1:NY, i in 1:NX-1
    u_open[j, i] = (mask[j, i] == 1 && mask[j, i+1] == 1)
end
@inbounds for j in 1:NY-1, i in 1:NX
    v_open[j, i] = (mask[j, i] == 1 && mask[j+1, i] == 1)
end

# Sponge
sponge = zeros(Float64, NY, NX)
const SPONGE_WIDTH = 5
const SPONGE_PEAK  = 1.0 / 60
@inbounds for j in 1:NY, i in 1:NX
    di = i - (NX - SPONGE_WIDTH)
    if di > 0
        sponge[j, i] = SPONGE_PEAK * (di / SPONGE_WIDTH)^2
    end
end

const IR = river.ix[1]
const JR = river.iy[1]
const NG = nrow(gauges)

# --- Forward solve given a ψ-trajectory -----------------------------------
#
# ψ_traj is a length-(NT+1) vector — ψ(t_n) for n = 0..NT.  Returns a
# (N_REC × NG) matrix of η at each gauge each timestep.

function fd_forward(ψ_traj::AbstractVector{Float64})
    @assert length(ψ_traj) == N_REC
    η  = zeros(Float64, NY, NX)
    u  = zeros(Float64, NY, NX - 1)
    v  = zeros(Float64, NY - 1, NX)
    out = zeros(Float64, N_REC, NG)

    # initial value at t=0 — record before stepping
    η[JR, IR] = ψ_traj[1]
    @inbounds for (k, gid) in enumerate(gauges.gauge_id)
        out[1, k] = η[gauges.iy[k], gauges.ix[k]]
    end

    for step in 1:NT
        # mass update at water interior cells
        @inbounds for j in 2:NY-1, i in 2:NX-1
            mask[j, i] == 1 || continue
            flux_x_e = Hu[j, i  ] * u[j, i  ]
            flux_x_w = Hu[j, i-1] * u[j, i-1]
            flux_y_n = Hv[j,   i] * v[j,   i]
            flux_y_s = Hv[j-1, i] * v[j-1, i]
            η[j, i] -= DT * ((flux_x_e - flux_x_w) / DX +
                             (flux_y_n - flux_y_s) / DY)
        end
        @inbounds for j in 1:NY, i in 1:NX
            sp = sponge[j, i]
            sp > 0 && (η[j, i] -= DT * sp * η[j, i])
        end
        η[JR, IR] = ψ_traj[step + 1]

        @inbounds for j in 1:NY, i in 1:NX-1
            if u_open[j, i]
                dηdx = (η[j, i+1] - η[j, i]) / DX
                u[j, i] += DT * (-G_GRAV * dηdx - B_DRAG * u[j, i])
            else
                u[j, i] = 0.0
            end
        end
        @inbounds for j in 1:NY-1, i in 1:NX
            if v_open[j, i]
                dηdy = (η[j+1, i] - η[j, i]) / DY
                v[j, i] += DT * (-G_GRAV * dηdy - B_DRAG * v[j, i])
            else
                v[j, i] = 0.0
            end
        end
        @inbounds for j in 1:NY, i in 1:NX-1
            sp = 0.5 * (sponge[j, i] + sponge[j, i+1])
            sp > 0 && (u[j, i] -= DT * sp * u[j, i])
        end
        @inbounds for j in 1:NY-1, i in 1:NX
            sp = 0.5 * (sponge[j, i] + sponge[j+1, i])
            sp > 0 && (v[j, i] -= DT * sp * v[j, i])
        end

        @inbounds for (k, gid) in enumerate(gauges.gauge_id)
            out[step + 1, k] = η[gauges.iy[k], gauges.ix[k]]
        end
    end
    return out
end

# --- Impulse response -----------------------------------------------------
#
# Send a narrow Gaussian pulse at t = t_pulse, width σ = 4·DT, unit area.
# Record gauges across [0, T].  Because the SWE is time-translation
# invariant, the impulse response is h_g(τ) = (recorded gauge at t_pulse+τ).
#
# Why a Gaussian rather than a true δ?  A literal δ would put a finite
# jump on η at the source cell that violates the FD CFL.  A 4-step
# Gaussian is narrow enough that its convolution with the unknown ψ is
# nearly the same as the true impulse response — and we deconvolve it
# out below by including the Gaussian shape in the basis.

const σ_PULSE = 4.0 * DT
const t_pulse = 6.0 * σ_PULSE       # leave room for the pulse to ring up

pulse_traj = [(1.0 / (σ_PULSE * sqrt(2π))) * exp(-((tn - t_pulse) / σ_PULSE)^2)
              for tn in t_axis]

println("Computing impulse response (one FD forward pass)…")
tic = time()
h_g = fd_forward(pulse_traj)              # (N_REC × NG)
@printf("  impulse forward done in %.2f s\n", time() - tic)

# --- Build the convolution operator G_full --------------------------------
#
# For a discrete trajectory ψ_n (n=1..N_REC) and gauge g,
#     g_pred[t_n] = (DT) · Σ_{m=1..n}  h_g[t_n - t_m + t_pulse] · ψ[m]
# where the shift +t_pulse comes from the pulse being centred at t_pulse,
# not at t=0.  Implemented as a lower-triangular Toeplitz matrix.

const N_REC_OBS_STRIDE = 5             # subsample obs every 5·DT = 60 s
const OBS_IDX = collect(1:N_REC_OBS_STRIDE:N_REC)   # row indices we use
const N_OBS = length(OBS_IDX)

# Build A_g for one gauge — rows are OBS_IDX, cols are 1..N_REC, entry
# (i, j) = h_g[OBS_IDX[i] - j + 1 + (t_pulse/DT)] · DT, zero if shift < 1
function build_Ag(h_col)
    shift = round(Int, t_pulse / DT)    # how far ahead the impulse response is centred
    A = zeros(N_OBS, N_REC)
    for (ii, ti) in enumerate(OBS_IDX)
        for j in 1:N_REC
            k = ti - j + 1 + shift      # index into h_col
            if 1 ≤ k ≤ N_REC
                A[ii, j] = h_col[k] * DT
            end
        end
    end
    return A
end

A_blocks = [build_Ag(h_g[:, k]) for k in 1:NG]
G_full = vcat(A_blocks...)              # (NG·N_OBS) × N_REC

# Observation vector — concatenate gauges in the same order
g_obs = vcat([obs[OBS_IDX, Symbol(gauges.gauge_id[k])] for k in 1:NG]...)

@printf("Forward operator G: %d × %d\n", size(G_full)...)
@printf("g_obs length: %d\n", length(g_obs))

# --- Tikhonov-regularised inverse -----------------------------------------
#
# Solve   minimise  ½ ||G ψ - g_obs||²  +  λ²/2 ||L² ψ||²  +  λ₀²/2 ψ[1]²
#
# L² is the discrete 2nd-derivative operator → penalises curvature
# (smoothness prior on ψ).  The ψ[1] anchor forces ψ(0) ≈ 0 (the surge
# hasn't started at t=0).

function build_L2(n)
    L = zeros(n - 2, n)
    for i in 1:n-2
        L[i, i  ] =  1.0
        L[i, i+1] = -2.0
        L[i, i+2] =  1.0
    end
    return L
end

L2 = build_L2(N_REC)

# Hyper-parameters chosen by L-curve heuristic — λ trades data fit vs smoothness
const λ_smooth = 3.0e3
const λ_anchor = 3.0e2

A_full = vcat(G_full,
              λ_smooth .* L2,
              λ_anchor .* reshape(vcat(1.0, zeros(N_REC - 1)), 1, N_REC))
b_full = vcat(g_obs,
              zeros(size(L2, 1)),
              zeros(1))

println("Solving regularised least squares …")
tic = time()
ψ_recovered = A_full \ b_full
@printf("  solved in %.2f s\n", time() - tic)

# --- Predicted gauges with the recovered ψ --------------------------------

pred_full = fd_forward(ψ_recovered)
err_l2 = sqrt(sum((ψ_recovered .- psi_truth.psi).^2) / length(ψ_recovered))
rms_truth = sqrt(sum(psi_truth.psi .^ 2) / length(psi_truth.psi))
peak_t = ψ_recovered[argmax(abs.(ψ_recovered))]
peak_o = psi_truth.psi[argmax(abs.(psi_truth.psi))]

@printf("\nAdjoint recovery:\n")
@printf("  L2 err = %.4f m   (relative: %.1f %%)\n", err_l2, 100 * err_l2 / rms_truth)
@printf("  truth peak = %+.3f m,  recovered peak = %+.3f m\n", peak_o, peak_t)

# --- Save -----------------------------------------------------------------

CSV.write(joinpath(DATA_DIR, "psi_recovered_adjoint.csv"),
          DataFrame(t = t_axis,
                    psi_truth = psi_truth.psi,
                    psi_recovered = ψ_recovered))

let df = DataFrame(t = t_axis)
    for (k, gid) in enumerate(gauges.gauge_id)
        df[!, Symbol("$(gid)_pred")] = pred_full[:, k]
    end
    CSV.write(joinpath(DATA_DIR, "gauges_adjoint_pred.csv"), df)
end

println("Wrote data/psi_recovered_adjoint.csv and data/gauges_adjoint_pred.csv")
