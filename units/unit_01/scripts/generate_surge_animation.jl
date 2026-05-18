#!/usr/bin/env julia
# Render an animation of the surge propagating across Moreton Bay.
# Reads snapshots.bin produced by generate_surge_data.jl, overlays the
# land mask (grey), gauge markers, and river-mouth marker.
# Output: figures/surge_animation.gif

using DelimitedFiles, CSV, DataFrames, JSON3, Plots, Printf

const HERE     = @__DIR__
const DATA_DIR = joinpath(HERE, "..", "data")
const FIG_DIR  = joinpath(HERE, "..", "figures")

# helper: find nearest index in a sorted vector
function nearest_idx(v, x)
    i = searchsortedfirst(v, x)
    if i == 1
        return 1
    elseif i > length(v)
        return length(v)
    else
        return abs(v[i] - x) < abs(v[i-1] - x) ? i : i-1
    end
end

mask   = readdlm(joinpath(DATA_DIR, "bay_mask.csv"), ',', Int)
gauges = CSV.read(joinpath(DATA_DIR, "bay_gauges.csv"),   DataFrame)
river  = CSV.read(joinpath(DATA_DIR, "river_source.csv"), DataFrame)
psi    = CSV.read(joinpath(DATA_DIR, "psi_truth.csv"),    DataFrame)
meta   = JSON3.read(read(joinpath(DATA_DIR, "snapshots_meta.json"), String))

const NX = Int(meta["nx"])
const NY = Int(meta["ny"])
const NF = Int(meta["nframes"])
const FRAME_TIMES = Float64.(meta["frame_times_s"])

# Read snapshots — Float32 array stored column-major as (Ny, Nx, Nframes)
snap_flat = Array{Float32}(undef, NY * NX * NF)
read!(joinpath(DATA_DIR, "snapshots.bin"), snap_flat)
snap = reshape(snap_flat, (NY, NX, NF))

# Colorbar limits — symmetric, fixed across all frames
const ηmax  = maximum(abs, snap)
const VLIM  = 0.55 * ηmax    # saturate slightly so far-field wakes are visible

println("Rendering ", NF, " frames; η ∈ [", -ηmax, ", ", ηmax, "] m  →  GIF…")

# Pre-compute a land overlay matrix (NaN over water, 1 on land)
land_layer = fill(NaN, NY, NX)
land_layer[mask .== 0] .= 1.0

# Use GR backend for native GIF output
gr()
default(legend = false, fontfamily = "Helvetica")

anim = @animate for f in 1:NF
    η_frame = copy(snap[:, :, f])
    η_frame[mask .== 0] .= NaN    # transparent on land

    t_hr  = FRAME_TIMES[f] / 3600
    psi_t = psi.psi[nearest_idx(psi.t, FRAME_TIMES[f])]

    p = heatmap(1:NX, 1:NY, η_frame;
                c = :balance, clims = (-VLIM, VLIM),
                aspect_ratio = 1, size = (600, 950),
                xlims = (0.5, NX + 0.5), ylims = (0.5, NY + 0.5),
                xlabel = "east →  (cell · 1 km)",
                ylabel = "north →  (cell · 1 km)",
                colorbar_title = "η (m)",
                background_color_inside = :white,
                title = @sprintf("Brisbane River surge — t = %.2f h   ψ(t) = %+.3f m",
                                 t_hr, psi_t),
                titlefontsize = 11)
    # land overlay (grey)
    heatmap!(p, 1:NX, 1:NY, land_layer;
             c = cgrad([:grey80, :grey80]), clims = (0, 1),
             colorbar = false, alpha = 1.0)
    # gauges
    scatter!(p, gauges.ix, gauges.iy;
             marker = :circle, c = :gold, ms = 6, msc = :black, msw = 1.2)
    for r in eachrow(gauges)
        annotate!(p, r.ix + 1.5, r.iy, text(r.gauge_id, 8, :left, :black))
    end
    # river source
    scatter!(p, river.ix, river.iy;
             marker = :utriangle, c = :crimson, ms = 9, msc = :black, msw = 1.2)
    annotate!(p, river.ix[1] - 1.5, river.iy[1] + 1.5,
              text("ψ", 11, :right, :crimson))
end

gif(anim, joinpath(FIG_DIR, "surge_animation.gif"); fps = 12)
println("wrote figures/surge_animation.gif (", NF, " frames at 12 fps)")
