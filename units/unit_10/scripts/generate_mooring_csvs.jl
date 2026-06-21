# Generate three-site synthetic mooring CSV data for the capstone inverse
# problem.  All three sites see a common Gaussian storm event (scenario_4)
# starting ~day 10 of a 30-day window; each responds differently because of
# local depth, mixing closure, and upwelling magnitude.
#
# Sites along the central GBR cross-shelf transect off Townsville:
#   A — Cleveland Bay  (-19.20°S, 146.81°E,  15 m, mixing-dominated)
#   B — Davies Reef    (-18.83°S, 147.65°E,  60 m, classical thermocline)
#   C — Myrmidon Reef  (-18.27°S, 147.39°E, 100 m, advection-dominated)
#
# Outputs (under ../data/):
#   sites_metadata.csv  — id, name, lat, lon, depth, sensor depths
#   mooring_A.csv       — hourly temperature traces, Site A
#   mooring_B.csv       — hourly temperature traces, Site B
#   mooring_C.csv       — hourly temperature traces, Site C
#
# Run via:
#   julia --project=. units/unit_10/scripts/generate_mooring_csvs.jl

include(joinpath(@__DIR__, "column_fd.jl"))
using Random, Printf

const SITES = (
    (id = "A", name = "Cleveland Bay",  lat = -19.200, lon = 146.810,
     H = 15.0,  κ_m = 5.0e-3, κ_b = 2.0e-4, h_m = 5.0,  w0_storm = 5.0e-6,
     T_surface = 28.5, T_deep = 26.0, z_t = -8.0,  δ_t = 3.0,
     depths = [1.0, 4.0, 8.0, 12.0, 14.0]),
    (id = "B", name = "Davies Reef",    lat = -18.830, lon = 147.647,
     H = 60.0,  κ_m = 1.0e-3, κ_b = 1.0e-5, h_m = 20.0, w0_storm = 5.0e-5,
     T_surface = 28.0, T_deep = 22.0, z_t = -25.0, δ_t = 5.0,
     depths = [2.0, 10.0, 25.0, 45.0, 58.0]),
    (id = "C", name = "Myrmidon Reef",  lat = -18.270, lon = 147.390,
     H = 100.0, κ_m = 2.0e-4, κ_b = 5.0e-6, h_m = 30.0, w0_storm = 8.0e-5,
     T_surface = 27.5, T_deep = 18.0, z_t = -40.0, δ_t = 6.0,
     depths = [2.0, 15.0, 40.0, 70.0, 95.0]),
)

const σ_noise = 0.05                  # observation noise (°C)
const data_dir = joinpath(@__DIR__, "..", "data")
mkpath(data_dir)

# Site-specific PARAMS override (NamedTuple spread).
function site_params(s)
    (PARAMS...,
     H         = s.H,
     κ_m       = s.κ_m,
     κ_b       = s.κ_b,
     h_m       = s.h_m,
     w0_storm  = s.w0_storm,
     T_surface = s.T_surface,
     T_deep    = s.T_deep,
     z_t       = s.z_t,
     δ_t       = s.δ_t)
end

function generate_site_csv(s; rng = Random.MersenneTwister(Int(s.H)))
    p  = site_params(s)
    dz = max(s.H / 40, 0.4)            # ~40 cells per column
    r  = run_scenario(scenario_4(); dz = dz, p = p)
    zg, tg, Tg = _grids(r)
    hours = tg ./ 3600.0

    Tsens = Matrix{Float64}(undef, length(tg), length(s.depths))
    for (j, d) in enumerate(s.depths)
        zi = argmin(abs.(zg .+ d))     # depth measured *down* from surface
        Tsens[:, j] = Tg[zi, :] .+ σ_noise .* randn(rng, length(tg))
    end

    out = joinpath(data_dir, "mooring_$(s.id).csv")
    open(out, "w") do io
        depth_cols = join(["T_$(round(d; digits=1))m" for d in s.depths], ",")
        println(io, "time_hours,", depth_cols)
        for k in eachindex(tg)
            row = join((@sprintf("%.4f", Tsens[k, j]) for j in eachindex(s.depths)), ",")
            println(io, @sprintf("%.3f", hours[k]), ",", row)
        end
    end
    @info "wrote $out  ($(length(tg)) rows, $(length(s.depths)) sensors, σ_noise=$(σ_noise)°C)"
    out
end

# ---- sites metadata ----
open(joinpath(data_dir, "sites_metadata.csv"), "w") do io
    println(io, "site_id,name,lat,lon,H_m,sensor_depths_m")
    for s in SITES
        depths_q = '"' * join(s.depths, ',') * '"'
        println(io, "$(s.id),$(s.name),$(s.lat),$(s.lon),$(s.H),$depths_q")
    end
end
@info "wrote sites_metadata.csv"

# ---- per-site mooring data ----
for s in SITES
    generate_site_csv(s)
end

@info "done — $(length(SITES)) site CSVs + metadata in $data_dir"
