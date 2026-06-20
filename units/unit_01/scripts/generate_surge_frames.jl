#!/usr/bin/env julia
# Render one PNG per FD snapshot of η(x, y, t).
#
# These frames feed an HTML range-slider widget in unit_01.qmd that lets
# the reader scrub through the surge propagating across Moreton Bay in
# their own time.  Each PNG is small (~25-50 kB) and named with a
# zero-padded index so the slider can address them by integer.
#
# Output:
#   figures/surge_frames/frame_000.png … frame_072.png
#   figures/surge_frames/frames_meta.json  — t in seconds + ψ(t) per frame

using DelimitedFiles, CSV, DataFrames, JSON3, Plots, Printf
include(joinpath(@__DIR__, "_mapfig.jl"))   # rotated landscape bay maps (N ◀ left)

const HERE     = @__DIR__
const DATA_DIR = joinpath(HERE, "..", "data")
const FIG_DIR  = joinpath(HERE, "..", "figures", "surge_frames")
isdir(FIG_DIR) || mkpath(FIG_DIR)

mask   = readdlm(joinpath(DATA_DIR, "bay_mask.csv"), ',', Int)
gauges = CSV.read(joinpath(DATA_DIR, "bay_gauges.csv"),   DataFrame)
river  = CSV.read(joinpath(DATA_DIR, "river_source.csv"), DataFrame)
psi    = CSV.read(joinpath(DATA_DIR, "psi_truth.csv"),    DataFrame)
meta   = JSON3.read(read(joinpath(DATA_DIR, "snapshots_meta.json"), String))
bay_meta = JSON3.read(read(joinpath(DATA_DIR, "bay_meta.json"), String))

const NX = Int(meta["nx"])
const NY = Int(meta["ny"])
const NF = Int(meta["nframes"])
const DX = Float64(bay_meta["dx_m"])
const DY = Float64(bay_meta["dy_m"])
const FRAME_TIMES = Float64.(meta["frame_times_s"])

snap_flat = Array{Float32}(undef, NY * NX * NF)
read!(joinpath(DATA_DIR, "snapshots.bin"), snap_flat)
snap = reshape(snap_flat, (NY, NX, NF))

# axes in km for physical labels
xkm = [(i - 0.5) * DX / 1000 for i in 1:NX]
ykm = [(j - 0.5) * DY / 1000 for j in 1:NY]

# Gauge / river positions in km (cell-centre)
gx = [(gauges.ix[k] - 0.5) * DX / 1000 for k in 1:nrow(gauges)]
gy = [(gauges.iy[k] - 0.5) * DY / 1000 for k in 1:nrow(gauges)]
rx = (river.ix[1] - 0.5) * DX / 1000
ry = (river.iy[1] - 0.5) * DY / 1000

const DXKM = DX / 1000

# Colour scale — symmetric, saturate slightly so far-field wakes are visible
const ηmax = maximum(abs, snap)
const VLIM = 0.55 * ηmax

println("Rendering ", NF, " frames into figures/surge_frames/  (η ∈ ±",
        round(VLIM; digits = 3), " m)")

# Helper: nearest index in a sorted vector
function nearest_idx(v, x)
    i = searchsortedfirst(v, x)
    i == 1 && return 1
    i > length(v) && return length(v)
    return abs(v[i] - x) < abs(v[i - 1] - x) ? i : i - 1
end

gr()
default(legend = false, fontfamily = "Helvetica")

frame_records = Vector{Dict{String, Any}}()

for f in 1:NF
    η_frame = copy(snap[:, :, f])

    t_s = FRAME_TIMES[f]
    t_hr = t_s / 3600
    psi_t = psi.psi[nearest_idx(psi.t, t_s)]

    p = bay_map(η_frame, mask, DXKM;
        clims = (-VLIM, VLIM), cmap = :balance, clabel = "η  (m)",
        title = @sprintf("Brisbane River surge — t = %.2f h    ψ(t) = %+.3f m",
                         t_hr, psi_t),
        gauge_e = gx, gauge_n = gy, gauge_id = String.(gauges.gauge_id),
        river_e = rx, river_n = ry)

    fname = @sprintf("frame_%03d.png", f - 1)
    savefig(p, joinpath(FIG_DIR, fname))

    push!(frame_records,
          Dict("idx" => f - 1, "file" => fname,
               "t_s" => t_s, "t_hr" => t_hr, "psi_m" => psi_t))
    if f % 10 == 0
        println("  rendered frame ", f, " / ", NF)
    end
end

open(joinpath(FIG_DIR, "frames_meta.json"), "w") do io
    JSON3.pretty(io, Dict("nframes" => NF,
                           "vlim_m"  => VLIM,
                           "η_max_m" => ηmax,
                           "frames"  => frame_records))
end
println("Wrote ", NF, " PNGs + frames_meta.json")
