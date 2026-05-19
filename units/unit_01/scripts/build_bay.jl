#!/usr/bin/env julia
# Build a tractable, plausible Moreton Bay bathymetry + land mask + gauge layout.
#
# Coordinates: the SWE domain is a rectangle in local Cartesian (x,y) metres.
# We pin the SW corner of the domain to (lon=153.02°E, lat=-27.85°N) and use
# a local equirectangular projection (good enough for ~50 × 95 km).
#
# Output CSVs (written to ../data/):
#   bay_meta.json            domain bounds, grid spacing, projection params
#   bay_bathymetry.csv       Ny × Nx depth (m) — land cells stored as NaN
#   bay_mask.csv             Ny × Nx mask (1 = water, 0 = land)
#   bay_gauges.csv           gauge_id, name, lat, lon, ix, iy, blurb
#   river_source.csv         source row: name, lat, lon, ix, iy
#
# Land / bathymetry are heuristic: they encode the *shape* of Moreton Bay
# (mainland coast, Moreton Island, South Stradbroke, the eastern deep
# channel, the dredged Brisbane River entrance) without claiming to be a
# bathymetric chart. Good enough for the Unit 1 teaser, with a callout
# pointing students at AusBathyTopo / GEBCO for real work.

using Printf, DelimitedFiles, JSON3

const REPO_ROOT  = normpath(joinpath(@__DIR__, "..", "..", ".."))
const DATA_DIR   = normpath(joinpath(@__DIR__, "..", "data"))
isdir(DATA_DIR) || mkpath(DATA_DIR)

# --- Domain definition -----------------------------------------------------

const LON_MIN = 153.020             # SW corner longitude
const LAT_MIN = -27.850             # SW corner latitude
const LAT_REF = -27.40              # reference latitude for E-W scale

# metres per degree at our reference latitude
const M_PER_DEG_LAT = 111_000.0
const M_PER_DEG_LON = 111_000.0 * cosd(LAT_REF)   # ≈ 98_500 m/deg

const DX = 500.0                    # grid spacing (m) — 500 m chosen so the
const DY = 500.0                    # narrow South-Stradbroke spit (≈1.5 km
const NX = 100                      # wide) and the Broadwater channel
const NY = 190                      # behind it are resolved by several cells.
const LX = NX * DX                  # Domain is the same 50 × 95 km rectangle
const LY = NY * DY                  # as before — just at higher resolution.

# cell-centre coordinate helpers
xc(i) = (i - 0.5) * DX
yc(j) = (j - 0.5) * DY
lon_of_x(x) = LON_MIN + x / M_PER_DEG_LON
lat_of_y(y) = LAT_MIN + y / M_PER_DEG_LAT
x_of_lon(λ) = (λ - LON_MIN) * M_PER_DEG_LON
y_of_lat(φ) = (φ - LAT_MIN) * M_PER_DEG_LAT

# --- Land polygons (lat/lon) ----------------------------------------------
# Hand-traced from a mental map of Moreton Bay. Closed polygons (last vertex
# need not repeat the first — we close inside the inpolygon test).

# Mainland coast — runs the W side of the bay with the Brisbane River
# indenting inward at lat ≈ -27.38, and Cleveland Point bulging into the
# bay near (-27.49, 153.27).
const MAINLAND = [
    (153.020, -26.994),   # NW corner of domain (cuts inland)
    (153.190, -26.994),   # N edge / Redcliffe peninsula clipped
    (153.160, -27.100),
    (153.115, -27.150),
    (153.135, -27.230),
    (153.155, -27.320),
    (153.130, -27.360),   # heading into Brisbane R. mouth from N
    (153.080, -27.370),   # Brisbane R. mouth — north shore
    (153.040, -27.380),
    (153.020, -27.390),   # river goes off the W edge of the domain
    # (south of river)
    (153.080, -27.410),
    (153.130, -27.440),
    (153.180, -27.475),
    (153.235, -27.500),   # Cleveland Point bulge
    (153.215, -27.530),
    (153.195, -27.580),   # back of Cleveland Bay
    (153.230, -27.640),   # mainland heads south past Macleay Is
    (153.260, -27.720),
    (153.290, -27.790),
    (153.320, -27.840),
    (153.020, -27.850),   # SW corner closes
]

# Moreton Island — long sand island, NNE-SSW, ~38 km
const MORETON = [
    (153.430, -27.020),   # NE tip
    (153.460, -27.090),
    (153.480, -27.170),
    (153.490, -27.260),
    (153.485, -27.350),
    (153.470, -27.420),
    (153.435, -27.470),   # S tip
    (153.385, -27.460),
    (153.355, -27.420),
    (153.340, -27.330),
    (153.330, -27.230),
    (153.330, -27.130),
    (153.345, -27.050),
    (153.380, -27.010),   # NW tip
]

# North Stradbroke Island — the main island chunk in the SE.
const STRADBROKE = [
    (153.430, -27.495),
    (153.490, -27.510),
    (153.520, -27.560),
    (153.528, -27.620),
    (153.520, -27.690),
    (153.480, -27.720),
    (153.440, -27.720),
    (153.420, -27.680),
    (153.405, -27.610),
    (153.395, -27.555),
]

# South Stradbroke Island — long, narrow sand spit running ~N-S from
# Jumpinpin Bar (≈-27.745) down past the southern domain edge. Modelled as a
# single clean lozenge ~3 km wide, with the southern vertices pushed just
# past LAT_MIN so the polygon cleanly cuts the domain edge (avoids a thin
# disconnected sliver at the rasterisation).
const SOUTH_STRADBROKE = [
    (153.435, -27.745),   # N tip at Jumpinpin
    (153.465, -27.770),
    (153.475, -27.810),
    (153.475, -27.870),   # SE, past domain edge
    (153.435, -27.870),   # SW, past domain edge
    (153.430, -27.810),
    (153.420, -27.770),
]

# Russell / Macleay / Lamb / Karragarra island cluster — these crowd
# the southern bay between Cleveland Point and the Logan River mouth.
# Their presence severely restricts flow into the southern bay and
# is the reason the area is treated as estuarine in reality.
const RUSSELL_GROUP = [   # Russell + Lamb (main cluster)
    (153.355, -27.605),
    (153.385, -27.615),
    (153.390, -27.650),
    (153.375, -27.685),
    (153.345, -27.685),
    (153.335, -27.660),
    (153.340, -27.625),
]
const MACLEAY = [          # Macleay Is. — separate larger blob to the N
    (153.355, -27.555),
    (153.385, -27.560),
    (153.395, -27.590),
    (153.380, -27.605),
    (153.350, -27.605),
    (153.340, -27.585),
]
const KARRAGARRA = [       # Karragarra/Coochiemudlo — small islets
    (153.310, -27.575),
    (153.330, -27.585),
    (153.330, -27.605),
    (153.310, -27.610),
    (153.300, -27.595),
]

# Small islands inside the bay (St Helena, Mud, Peel, Coochiemudlo).
# Cluster of mid-bay islets — kept tiny and treated as a single blob so we
# don't pretend more bathymetric realism than we have.
const MID_BAY_ISLAND_A = [   # Peel / Mud island cluster
    (153.245, -27.420),
    (153.265, -27.420),
    (153.275, -27.460),
    (153.255, -27.480),
    (153.230, -27.470),
    (153.225, -27.440),
]

const ALL_LAND = [MAINLAND, MORETON, STRADBROKE, SOUTH_STRADBROKE,
                  RUSSELL_GROUP, MACLEAY, KARRAGARRA, MID_BAY_ISLAND_A]

# --- Point-in-polygon (ray casting) ----------------------------------------
function inpoly(λ, φ, poly)
    n = length(poly)
    inside = false
    j = n
    @inbounds for i in 1:n
        xi, yi = poly[i]
        xj, yj = poly[j]
        if ((yi > φ) != (yj > φ)) &&
           (λ < (xj - xi) * (φ - yi) / (yj - yi + eps()) + xi)
            inside = !inside
        end
        j = i
    end
    return inside
end

is_land(λ, φ) = any(p -> inpoly(λ, φ, p), ALL_LAND)

# --- Bathymetry construction ----------------------------------------------
#
# Strategy: define depth as a sum/clamp of simple analytic features that
# echo the real bay:
#   * base bay depth   ~  5 m
#   * eastern trough   ~ 18-22 m (runs N-S inside the bay along the eastern
#                                 deep channel)
#   * dredged Brisbane River channel  ~ 13 m (groove from river mouth
#                                            heading ESE into the trough)
#   * outside Moreton Is. (E of the island)  ~ 35-50 m
#   * shoals near river mouth  ~ 3-4 m
# Depths are positive (m).

# Eastern deep trough — a curved corridor from (-27.10, 153.36) down to
# (-27.45, 153.41). Modelled as a Gaussian groove around a polyline.
const TROUGH_LINE = [
    (153.355, -27.080),
    (153.360, -27.150),
    (153.370, -27.220),
    (153.380, -27.290),
    (153.395, -27.360),
    (153.410, -27.440),
]

# Brisbane River dredged channel — from river mouth heading E and bending SE
# to merge into the eastern trough WELL before reaching Moreton Island
# (Moreton's W edge sits near lon 153.34–153.36 at lat ≈ -27.42).
const RIVER_CHANNEL = [
    (153.080, -27.382),
    (153.150, -27.388),
    (153.220, -27.395),
    (153.280, -27.405),
    (153.310, -27.420),   # joins TROUGH_LINE near (153.395, -27.360)
]

# Northwest Channel — the deep N-S passage that real Moreton Bay uses to
# exchange water with the open ocean through its NORTHERN entrance. Sits
# between the mainland coast and Moreton Island's western shore, then bends
# slightly W toward the northern domain edge where the bay opens to the sea.
const NORTH_CHANNEL = [
    (153.300, -27.060),   # northern entrance (top of domain)
    (153.310, -27.140),
    (153.320, -27.220),
    (153.335, -27.300),
    (153.355, -27.370),   # merges into the eastern trough
]

# Distance from a point to a polyline (in metres, using local projection)
function dist_to_polyline(λ, φ, line)
    px, py = x_of_lon(λ), y_of_lat(φ)
    dmin = Inf
    @inbounds for k in 1:length(line)-1
        ax, ay = x_of_lon(line[k][1]),   y_of_lat(line[k][2])
        bx, by = x_of_lon(line[k+1][1]), y_of_lat(line[k+1][2])
        # project P onto segment AB
        dx, dy = bx - ax, by - ay
        seglen2 = dx*dx + dy*dy
        t = clamp(((px - ax) * dx + (py - ay) * dy) / max(seglen2, 1.0), 0.0, 1.0)
        qx, qy = ax + t * dx, ay + t * dy
        d = hypot(px - qx, py - qy)
        d < dmin && (dmin = d)
    end
    return dmin
end

# Decide if a point is east of Moreton Is (outside the bay) — quick test
# using the island polygon: any (λ, φ) east of the eastern envelope of
# MORETON at that latitude is "open ocean".
function is_east_of_moreton(λ, φ)
    # Take the eastern boundary by finding max λ in MORETON near this φ
    λ_east = -Inf
    @inbounds for (l, p) in MORETON
        if abs(p - φ) < 0.30
            l > λ_east && (λ_east = l)
        end
    end
    λ_east == -Inf && return false
    return λ > λ_east + 0.003     # 300m buffer to keep beach face wet
end

function depth_at(λ, φ)
    # outside Moreton Is — open shelf
    if is_east_of_moreton(λ, φ)
        # ramp from 18 m at the island shore to 50 m at the eastern edge of
        # the domain
        x_local = x_of_lon(λ)
        ramp = clamp((x_local - x_of_lon(153.49)) / (x_of_lon(153.528) - x_of_lon(153.49)),
                     0.0, 1.0)
        return 18.0 + ramp * 32.0
    end

    # inside the bay
    d_base = 6.0

    # eastern deep trough — Gaussian groove ~3 km wide, +14 m depth at centre
    d_trough = dist_to_polyline(λ, φ, TROUGH_LINE)
    bonus_trough = 14.0 * exp(-(d_trough / 2500.0)^2)

    # dredged river channel — narrower (1.5 km), +7 m on top of base
    d_river = dist_to_polyline(λ, φ, RIVER_CHANNEL)
    bonus_river = 7.0 * exp(-(d_river / 1200.0)^2)

    # Northwest Channel — deep northern entrance, ~2.5 km wide, +12 m
    d_nwc = dist_to_polyline(λ, φ, NORTH_CHANNEL)
    bonus_nwc = 12.0 * exp(-(d_nwc / 2200.0)^2)

    # shoals near river mouth (subtract a bit of depth in a small patch)
    d_mouth = hypot(x_of_lon(λ) - x_of_lon(153.07),
                    y_of_lat(φ) - y_of_lat(-27.385))
    shoal_pen = -2.5 * exp(-(d_mouth / 2500.0)^2)

    return max(1.5, d_base + bonus_trough + bonus_river + bonus_nwc + shoal_pen)
end

# --- Rasterise -------------------------------------------------------------

bathy = fill(NaN, NY, NX)
mask  = zeros(Int, NY, NX)

for j in 1:NY, i in 1:NX
    λ = lon_of_x(xc(i))
    φ = lat_of_y(yc(j))
    if is_land(λ, φ)
        bathy[j, i] = NaN
        mask[j, i]  = 0
    else
        bathy[j, i] = depth_at(λ, φ)
        mask[j, i]  = 1
    end
end

# --- Gauges + river source -------------------------------------------------
#
# Real Moreton Bay tide gauges (or close analogues). Lat/lon rounded; ix/iy
# are snapped to the nearest *water* cell.

const GAUGES = [
    # G1 is placed at Mud Island (real Moreton Bay site, ~18 km
    # NNE of the Luggage Point river mouth). The obvious choice —
    # Brisbane Bar / Pile Light — sits only ~1.5 km from the source
    # cell, which makes the η_G1 → ψ inverse trivially close to a
    # one-to-one read-off. Mud Island gives a clean mid-bay
    # near-source observer whose signal has been smoothed enough by
    # the bay's Green's function that the inversion is genuinely
    # informative.
    ("G1", "Mud Island",        -27.210, 153.215,
        "mid-bay · nearest informative gauge"),
    ("G2", "Wellington Point",  -27.480, 153.236,
        "central bay · across from river"),
    ("G3", "Tangalooma Roads",  -27.205, 153.320,
        "Moreton Is. lee · NE bay"),
    ("G4", "Russell Channel",   -27.660, 153.290,
        "southern bay · west of Russell Is."),
]

const RIVER_MOUTH = ("RM", "Brisbane River mouth", -27.378, 153.165,
    "unknown surge inflow ψ(t)")

function snap_to_water(lat, lon; max_r = 10)
    i0 = clamp(round(Int, x_of_lon(lon) / DX + 0.5), 1, NX)
    j0 = clamp(round(Int, y_of_lat(lat) / DY + 0.5), 1, NY)
    if mask[j0, i0] == 1
        return (i0, j0)
    end
    # spiral outward — return nearest water cell within max_r
    best = (0, 0)
    bestd = Inf
    for r in 1:max_r, di in -r:r, dj in -r:r
        ii, jj = i0 + di, j0 + dj
        (1 ≤ ii ≤ NX && 1 ≤ jj ≤ NY) || continue
        mask[jj, ii] == 1 || continue
        d = hypot(di, dj)
        if d < bestd
            bestd = d
            best = (ii, jj)
        end
        bestd ≤ r && r ≥ 1 && return best   # found something at this radius
    end
    best == (0, 0) && error("no water cell within $max_r cells of ($lat, $lon)")
    return best
end

snapped_gauges = map(GAUGES) do (id, name, lat, lon, blurb)
    i, j = snap_to_water(lat, lon)
    (id, name, lat, lon, i, j, blurb,
     lon_of_x(xc(i)), lat_of_y(yc(j)), bathy[j, i])
end

snapped_river = let (id, name, lat, lon, blurb) = RIVER_MOUTH
    i, j = snap_to_water(lat, lon)
    (id, name, lat, lon, i, j, blurb,
     lon_of_x(xc(i)), lat_of_y(yc(j)), bathy[j, i])
end

# --- Write CSVs ------------------------------------------------------------

writedlm(joinpath(DATA_DIR, "bay_bathymetry.csv"), bathy, ',')
writedlm(joinpath(DATA_DIR, "bay_mask.csv"),       mask,  ',')

open(joinpath(DATA_DIR, "bay_gauges.csv"), "w") do io
    println(io, "gauge_id,name,lat,lon,ix,iy,blurb,snap_lon,snap_lat,depth_m")
    for g in snapped_gauges
        @printf(io, "%s,%s,%.4f,%.4f,%d,%d,%s,%.4f,%.4f,%.2f\n",
                g[1], g[2], g[3], g[4], g[5], g[6], g[7], g[8], g[9], g[10])
    end
end

open(joinpath(DATA_DIR, "river_source.csv"), "w") do io
    println(io, "source_id,name,lat,lon,ix,iy,blurb,snap_lon,snap_lat,depth_m")
    g = snapped_river
    @printf(io, "%s,%s,%.4f,%.4f,%d,%d,%s,%.4f,%.4f,%.2f\n",
            g[1], g[2], g[3], g[4], g[5], g[6], g[7], g[8], g[9], g[10])
end

meta = Dict(
    "lon_min"        => LON_MIN,
    "lat_min"        => LAT_MIN,
    "lat_ref"        => LAT_REF,
    "m_per_deg_lat"  => M_PER_DEG_LAT,
    "m_per_deg_lon"  => M_PER_DEG_LON,
    "dx_m"           => DX,
    "dy_m"           => DY,
    "nx"             => NX,
    "ny"             => NY,
    "lx_m"           => LX,
    "ly_m"           => LY,
    "lon_max"        => LON_MIN + LX / M_PER_DEG_LON,
    "lat_max"        => LAT_MIN + LY / M_PER_DEG_LAT,
    "depth_units"    => "m, positive down; NaN = land",
)
open(joinpath(DATA_DIR, "bay_meta.json"), "w") do io
    JSON3.pretty(io, meta)
end

n_water = sum(mask .== 1)
n_land  = sum(mask .== 0)
println("Built bay model:")
@printf("  grid:        %d × %d (dx=dy=%.0f m)\n", NX, NY, DX)
@printf("  water cells: %d (%.1f%%)\n", n_water, 100 * n_water / (NX*NY))
@printf("  land cells:  %d (%.1f%%)\n", n_land,  100 * n_land  / (NX*NY))
@printf("  depth range (water): %.1f to %.1f m\n",
        minimum(skipmissing(filter(!isnan, vec(bathy)))),
        maximum(skipmissing(filter(!isnan, vec(bathy)))))
println("\nGauges (snapped to nearest water cell):")
for g in snapped_gauges
    @printf("  %s %-18s  lat,lon = (%.4f, %.4f)  cell=(%3d,%3d)  H=%.1f m\n",
            g[1], g[2], g[3], g[4], g[5], g[6], g[10])
end
let g = snapped_river
    @printf("\nRiver source:\n  %s %-22s  lat,lon = (%.4f, %.4f)  cell=(%3d,%3d)  H=%.1f m\n",
            g[1], g[2], g[3], g[4], g[5], g[6], g[10])
end

# --- Render bathymetry figure for the qmd --------------------------------

using Plots
gr()

FIG_DIR = normpath(joinpath(@__DIR__, "..", "figures"))
isdir(FIG_DIR) || mkpath(FIG_DIR)

display_bathy = copy(bathy)
display_bathy[mask .== 0] .= NaN

# X/Y axes in km (cell-centre coordinates) so the figure reads physically
xkm = [(i - 0.5) * DX / 1000 for i in 1:NX]
ykm = [(j - 0.5) * DY / 1000 for j in 1:NY]

p = heatmap(xkm, ykm, display_bathy;
            c = cgrad(:deep, rev = false),
            yflip = false,
            aspect_ratio = 1,
            size = (560, 870),
            background_color_inside = RGB(0.85, 0.85, 0.85),  # grey land
            xlabel = "east  (km)", ylabel = "north  (km)",
            colorbar_title = "depth  (m)",
            title = "Moreton Bay bathymetry — $(NX)×$(NY) at $(round(Int, DX)) m",
            titlefontsize = 11)

# overlay gauges
scatter!(p,
         [g[8] |> x -> x_of_lon(x) / 1000 for g in snapped_gauges],   # snap_lon → km
         [g[9] |> x -> y_of_lat(x) / 1000 for g in snapped_gauges];
         marker = :circle, c = :gold, ms = 6, msc = :black, msw = 1.0,
         label = "tide gauges")
for g in snapped_gauges
    xk = x_of_lon(g[8]) / 1000
    yk = y_of_lat(g[9]) / 1000
    annotate!(p, xk + 1.0, yk, text(g[1], 8, :left, :black))
end

# river source
let g = snapped_river
    xk = x_of_lon(g[8]) / 1000
    yk = y_of_lat(g[9]) / 1000
    scatter!(p, [xk], [yk];
             marker = :utriangle, c = :crimson, ms = 9, msc = :black, msw = 1.2,
             label = "river source ψ(t)")
    annotate!(p, xk - 1.0, yk + 1.5, text("ψ", 11, :right, :crimson))
end

plot!(p, legend = :topright, legendfontsize = 7)

savefig(p, joinpath(FIG_DIR, "bathymetry.png"))
println("\nWrote figures/bathymetry.png")
