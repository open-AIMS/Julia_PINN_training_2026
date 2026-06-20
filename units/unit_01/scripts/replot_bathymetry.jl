#!/usr/bin/env julia
# Re-render figures/bathymetry.png from the EXISTING data CSVs, in the rotated
# landscape orientation (North ◀ left, East ▲ up). This does NOT regenerate the
# bay data — it only redraws the figure, so it is safe to run any time.
# (build_bay.jl writes the same figure when the data is rebuilt.)

using DelimitedFiles, CSV, DataFrames
include(joinpath(@__DIR__, "_mapfig.jl"))

const DATA_DIR = normpath(joinpath(@__DIR__, "..", "data"))
const FIG_DIR  = normpath(joinpath(@__DIR__, "..", "figures"))
const DXKM     = 0.5                       # 500 m grid

bathy  = readdlm(joinpath(DATA_DIR, "bay_bathymetry.csv"), ',')
mask   = round.(Int, readdlm(joinpath(DATA_DIR, "bay_mask.csv"), ',', Float64))
gauges = CSV.read(joinpath(DATA_DIR, "bay_gauges.csv"),   DataFrame)
river  = CSV.read(joinpath(DATA_DIR, "river_source.csv"), DataFrame)

# bathy has NaN on land; replace with NaN explicitly handled by bay_map via mask
bathy = map(x -> x isa AbstractString ? NaN : Float64(x), bathy)
NY, NX = size(bathy)

depths = filter(!isnan, vec(bathy))
clims  = (floor(minimum(depths)), ceil(maximum(depths)))

ge = [(gauges.ix[k] - 0.5) * DXKM for k in 1:nrow(gauges)]
gn = [(gauges.iy[k] - 0.5) * DXKM for k in 1:nrow(gauges)]
re = (river.ix[1] - 0.5) * DXKM
rn = (river.iy[1] - 0.5) * DXKM

p = bay_map(bathy, mask, DXKM;
    clims = clims, cmap = cgrad(:deep),
    clabel = "depth  (m)",
    title  = "Moreton Bay bathymetry — $(NX)×$(NY) at 500 m  (N ◀ left)",
    gauge_e = ge, gauge_n = gn, gauge_id = String.(gauges.gauge_id),
    river_e = re, river_n = rn)

isdir(FIG_DIR) || mkpath(FIG_DIR)
savefig(p, joinpath(FIG_DIR, "bathymetry.png"))
println("wrote figures/bathymetry.png  (rotated landscape, North ◀ left)")
