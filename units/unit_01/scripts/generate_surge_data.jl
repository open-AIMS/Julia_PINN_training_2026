#!/usr/bin/env julia
# Run the linearised shallow-water forward solve on the Moreton Bay grid
# built by build_bay.jl, with a synthetic Brisbane-River surge ψ(t)
# imposed at the river-mouth cell. Output:
#
#   data/gauges_observed.csv   (t, η_G1, η_G2, η_G3, η_G4)  + noise
#   data/gauges_clean.csv      (t, η_G1, η_G2, η_G3, η_G4)  noise-free
#   data/psi_truth.csv         (t, ψ_truth)                 ground truth
#   data/snapshots.bin         Float32 Nx × Ny × Nframes   for animation
#   data/snapshots_meta.json   metadata for snapshots.bin
#
# Model: Arakawa C-grid, forward-backward time stepping, linearised SWE
# (de Wolff eq. 6–7).  Land cells: η doesn't update, velocity faces
# touching land are clamped to zero.  East-edge sponge layer (Rayleigh
# damping) lets surge waves leave the domain.  Source: Dirichlet on η at
# the river-mouth cell so the "unknown" ψ(t) is exactly the surface
# elevation perturbation at the source.

using DelimitedFiles, CSV, DataFrames, JSON3, Random, Printf

const HERE     = @__DIR__
const DATA_DIR = joinpath(HERE, "..", "data")
const FIG_DIR  = joinpath(HERE, "..", "figures")
isdir(FIG_DIR) || mkpath(FIG_DIR)

# --- Load bay model --------------------------------------------------------

H_raw  = readdlm(joinpath(DATA_DIR, "bay_bathymetry.csv"), ',', Float64)
mask   = readdlm(joinpath(DATA_DIR, "bay_mask.csv"),       ',', Int)
gauges = CSV.read(joinpath(DATA_DIR, "bay_gauges.csv"),   DataFrame)
river  = CSV.read(joinpath(DATA_DIR, "river_source.csv"), DataFrame)
meta   = JSON3.read(read(joinpath(DATA_DIR, "bay_meta.json"), String))

const NX  = Int(meta["nx"])
const NY  = Int(meta["ny"])
const DX  = Float64(meta["dx_m"])
const DY  = Float64(meta["dy_m"])

# Replace NaN (land) with a placeholder; mask gates everything anyway
H = copy(H_raw)
H[isnan.(H)] .= 5.0

# --- Numerics --------------------------------------------------------------

const G_GRAV = 9.81           # gravitational acceleration (m/s²)
const B_DRAG = 5.0e-5         # linear drag coefficient (1/s)  — small
const T_END  = 6 * 3600.0     # 6 h simulation
const DT     = 12.0           # time step (s) — CFL safe for H_max ≈ 45 m
const NT     = Int(floor(T_END / DT))
const FRAME_STRIDE = 25       # snapshot every 25·DT = 300 s = 5 min
const NFRAMES      = Int(floor(NT / FRAME_STRIDE)) + 1

# CFL check
const C_MAX = sqrt(G_GRAV * maximum(H[mask .== 1]))
const CFL   = C_MAX * DT * sqrt(1/DX^2 + 1/DY^2)
@printf("CFL number = %.3f (must be < 1)\n", CFL)
@assert CFL < 0.95 "CFL violated — reduce DT or coarsen grid"

# --- Allocate fields -------------------------------------------------------

η  = zeros(Float64, NY, NX)
u  = zeros(Float64, NY, NX - 1)   # u at vertical faces  (j, i+½)
v  = zeros(Float64, NY - 1, NX)   # v at horizontal faces (j+½, i)

# Face depths (average of two adjacent cell depths)
Hu = zeros(Float64, NY, NX - 1)
Hv = zeros(Float64, NY - 1, NX)
@inbounds for j in 1:NY, i in 1:NX-1
    Hu[j, i] = 0.5 * (H[j, i] + H[j, i+1])
end
@inbounds for j in 1:NY-1, i in 1:NX
    Hv[j, i] = 0.5 * (H[j, i] + H[j+1, i])
end

# Face masks: a velocity face is open only when both neighbouring cells
# are water.  This is how we enforce u·n̂ = 0 on the coast.
u_open = falses(NY, NX - 1)
v_open = falses(NY - 1, NX)
@inbounds for j in 1:NY, i in 1:NX-1
    u_open[j, i] = (mask[j, i] == 1 && mask[j, i+1] == 1)
end
@inbounds for j in 1:NY-1, i in 1:NX
    v_open[j, i] = (mask[j, i] == 1 && mask[j+1, i] == 1)
end

# Sponge layer along the eastern edge of the domain (outside Moreton Is.)
# Absorbs outgoing waves so the bay doesn't ring forever.
sponge = zeros(Float64, NY, NX)
const SPONGE_WIDTH = 5             # cells
const SPONGE_PEAK  = 1.0 / 60      # 1/s → e-folding time 60 s at the edge
@inbounds for j in 1:NY, i in 1:NX
    di = i - (NX - SPONGE_WIDTH)
    if di > 0
        sponge[j, i] = SPONGE_PEAK * (di / SPONGE_WIDTH)^2
    end
end

# --- Surge profile ψ(t) ----------------------------------------------------
#
# Synthetic flood surge: a leading freshwater pulse from heavy rain in the
# Brisbane catchment, followed by a smaller follow-on bulge.  Amplitude
# scaled so peak surface anomaly at the source is ~0.45 m — consistent
# with an Australian Height Datum anomaly observed during minor-to-
# moderate Brisbane River floods (de-tided).
#
# ψ(t) = A₁ · gauss(t; μ₁, σ₁)  +  A₂ · gauss(t; μ₂, σ₂)

ψ_truth(t) = 0.45 * exp(-((t - 2.0 * 3600) / (0.55 * 3600))^2) +
             0.18 * exp(-((t - 4.3 * 3600) / (0.55 * 3600))^2)

# River-mouth cell
const IR = river.ix[1]
const JR = river.iy[1]

# Gauge cells
const GAUGE_IDS = gauges.gauge_id
const GAUGE_IJ  = [(gauges.ix[k], gauges.iy[k]) for k in 1:nrow(gauges)]

println("River-mouth source @ cell (", IR, ", ", JR, ")  H=",
        round(H[JR, IR]; digits=1), " m")
for (k, (i, j)) in enumerate(GAUGE_IJ)
    println("Gauge ", GAUGE_IDS[k], " @ cell (", i, ", ", j,
            ")  H=", round(H[j, i]; digits=1), " m")
end

# --- Time integration ------------------------------------------------------

# Outputs
gauge_clean = zeros(Float64, NT + 1, length(GAUGE_IJ))
ψ_record    = zeros(Float64, NT + 1)
snapshots   = zeros(Float32, NY, NX, NFRAMES)
frame_times = zeros(Float64, NFRAMES)

# Initial condition: rest state
fill!(η, 0.0); fill!(u, 0.0); fill!(v, 0.0)
snapshots[:, :, 1] .= Float32.(η)
frame_times[1] = 0.0
ψ_record[1] = ψ_truth(0.0)
for (k, (i, j)) in enumerate(GAUGE_IJ)
    gauge_clean[1, k] = η[j, i]
end

println("\nIntegrating ", NT, " steps × ", DT, "s = ",
        round(NT * DT / 3600; digits=2), " h …")
const TIC = time()

# Wrap the loop in a function so Julia's scoping handles the closures and
# loop-local mutation cleanly.
function integrate!(η, u, v, snapshots, frame_times, gauge_clean, ψ_record)
    frame_idx = 1
    for step in 1:NT
    t = step * DT

    # 1. Continuity:  ∂η/∂t = -∂(Hu·u)/∂x - ∂(Hv·v)/∂y
    @inbounds for j in 2:NY-1, i in 2:NX-1
        mask[j, i] == 1 || continue
        # flux through east face minus flux through west face
        flux_x_e = Hu[j, i  ] * u[j, i  ]
        flux_x_w = Hu[j, i-1] * u[j, i-1]
        flux_y_n = Hv[j,   i] * v[j,   i]
        flux_y_s = Hv[j-1, i] * v[j-1, i]
        η[j, i] -= DT * ((flux_x_e - flux_x_w) / DX +
                         (flux_y_n - flux_y_s) / DY)
    end

    # 2. Sponge — damp η near eastern boundary
    @inbounds for j in 1:NY, i in 1:NX
        sponge[j, i] > 0 && (η[j, i] -= DT * sponge[j, i] * η[j, i])
    end

    # 3. Apply river-mouth source (Dirichlet on η at source cell)
    η[JR, IR] = ψ_truth(t)

    # 4. Momentum (using the *new* η):
    #    ∂u/∂t = -g ∂η/∂x - b·u
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

    # 5. Sponge — damp velocities near eastern boundary as well
    @inbounds for j in 1:NY, i in 1:NX-1
        sp = 0.5 * (sponge[j, i] + sponge[j, i+1])
        sp > 0 && (u[j, i] -= DT * sp * u[j, i])
    end
    @inbounds for j in 1:NY-1, i in 1:NX
        sp = 0.5 * (sponge[j, i] + sponge[j+1, i])
        sp > 0 && (v[j, i] -= DT * sp * v[j, i])
    end

    # 6. Record gauge readings + ψ for this step
    ψ_record[step + 1] = ψ_truth(t)
    @inbounds for (k, (i, j)) in enumerate(GAUGE_IJ)
        gauge_clean[step + 1, k] = η[j, i]
    end

    # 7. Snapshot every FRAME_STRIDE steps
    if step % FRAME_STRIDE == 0 && frame_idx < NFRAMES
        frame_idx += 1
        snapshots[:, :, frame_idx]  .= Float32.(η)
        frame_times[frame_idx]      = t
    end
    end  # for step
    return frame_idx
end  # integrate!

frame_idx = integrate!(η, u, v, snapshots, frame_times, gauge_clean, ψ_record)
@printf("FD solve done in %.2f s\n", time() - TIC)

# --- Add observation noise to gauges --------------------------------------

Random.seed!(0x4007)
const σ_OBS = 0.015   # 1.5 cm — typical tide-gauge noise after de-tiding
gauge_noisy = gauge_clean .+ σ_OBS .* randn(size(gauge_clean))

# --- Persist outputs -------------------------------------------------------

t_axis = collect(0:DT:NT*DT)

let df = DataFrame(t = t_axis)
    for (k, gid) in enumerate(GAUGE_IDS)
        df[!, Symbol(gid)] = gauge_clean[:, k]
    end
    CSV.write(joinpath(DATA_DIR, "gauges_clean.csv"), df)
end

let df = DataFrame(t = t_axis)
    for (k, gid) in enumerate(GAUGE_IDS)
        df[!, Symbol(gid)] = gauge_noisy[:, k]
    end
    CSV.write(joinpath(DATA_DIR, "gauges_observed.csv"), df)
end

CSV.write(joinpath(DATA_DIR, "psi_truth.csv"),
          DataFrame(t = t_axis, psi = ψ_record))

# Trim unused frames if any
snap_used = snapshots[:, :, 1:frame_idx]
ft_used   = frame_times[1:frame_idx]

# Save snapshots as a raw Float32 binary
open(joinpath(DATA_DIR, "snapshots.bin"), "w") do io
    write(io, snap_used)
end

snap_meta = Dict(
    "nx"        => NX,
    "ny"        => NY,
    "nframes"   => Int(frame_idx),
    "dt_s"      => DT,
    "frame_stride" => FRAME_STRIDE,
    "frame_dt_s"   => DT * FRAME_STRIDE,
    "t_end_s"   => NT * DT,
    "frame_times_s" => collect(ft_used),
    "dtype"     => "Float32",
    "shape"     => "Ny, Nx, Nframes (column-major)",
    "obs_noise_std_m" => σ_OBS,
)
open(joinpath(DATA_DIR, "snapshots_meta.json"), "w") do io
    JSON3.pretty(io, snap_meta)
end

@printf("\nWrote:\n")
@printf("  data/gauges_clean.csv     (%d rows × %d gauges)\n",
        size(gauge_clean, 1), size(gauge_clean, 2))
@printf("  data/gauges_observed.csv  (σ = %.3f m)\n", σ_OBS)
@printf("  data/psi_truth.csv\n")
@printf("  data/snapshots.bin        (%d × %d × %d Float32)\n",
        NY, NX, frame_idx)
@printf("  data/snapshots_meta.json\n")

println("\nMax |η| at any gauge:")
for (k, gid) in enumerate(GAUGE_IDS)
    @printf("  %s : clean=%.4f m   noisy=%.4f m\n",
            gid, maximum(abs, gauge_clean[:, k]),
            maximum(abs, gauge_noisy[:, k]))
end
@printf("Max |ψ_truth|: %.4f m\n", maximum(abs, ψ_record))
