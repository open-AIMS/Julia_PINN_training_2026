# Generate a schematic map showing the three capstone mooring sites along
# the central Great Barrier Reef cross-shelf transect.  Stylised — no real
# coastline data — just enough to anchor the geometry in the reader's head.
#
# Output: ../figures/site_map.png

using CairoMakie

const SITES = (
    (id="A", name="Cleveland Bay",  lat=-19.200, lon=146.810, H=15.0,
     blurb="inshore · 15 m · tidal mixing"),
    (id="B", name="Davies Reef",    lat=-18.830, lon=147.647, H=60.0,
     blurb="mid-shelf · 60 m · classical thermocline"),
    (id="C", name="Myrmidon Reef",  lat=-18.270, lon=147.390, H=100.0,
     blurb="outer shelf · 100 m · advection-dominated"),
)

# Simplified Queensland-coast polyline (a few hand-picked points around
# Townsville).  Not geodetically accurate — just orients the viewer.
const COAST = [
    (146.55, -18.45),
    (146.60, -18.70),
    (146.65, -18.95),
    (146.75, -19.15),
    (146.85, -19.25),  # near Townsville
    (147.10, -19.45),
    (147.30, -19.70),
]

# Approximate 200 m shelf-break contour, sketched as a straight-ish line
# offshore of the sites.
const SHELF_BREAK = [
    (147.55, -18.10),
    (147.50, -18.45),
    (147.45, -18.80),
    (147.40, -19.10),
    (147.45, -19.40),
]

function draw_map(outpath)
    fig = Figure(size = (900, 700))
    ax = Axis(fig[1, 1];
        xlabel = "Longitude (°E)",
        ylabel = "Latitude (°S, shown as negative)",
        title  = "Capstone mooring transect — central Great Barrier Reef (schematic)",
        aspect = DataAspect())

    # Land (west of the coast polyline) shaded lightly.
    coast_x = first.(COAST); coast_y = last.(COAST)
    land_x = vcat(coast_x, 146.30, 146.30, coast_x[1])
    land_y = vcat(coast_y, coast_y[end], coast_y[1], coast_y[1])
    poly!(ax, Point2f.(land_x, land_y); color = (:tan, 0.35),
          strokecolor = :sienna, strokewidth = 0.5)

    # Coastline overlay
    lines!(ax, coast_x, coast_y; color = :sienna, linewidth = 2,
           label = "QLD coast (schematic)")

    # Continental shelf break (200 m isobath, schematic)
    sb_x = first.(SHELF_BREAK); sb_y = last.(SHELF_BREAK)
    lines!(ax, sb_x, sb_y; color = :steelblue, linewidth = 2,
           linestyle = :dash, label = "~200 m shelf break")

    # Sites — label A to the right, B and C to the left so the text
    # doesn't run off the outer-shelf side of the figure.
    for s in SITES
        scatter!(ax, [s.lon], [s.lat]; markersize = 18, color = :crimson,
                 strokecolor = :black, strokewidth = 1.5)
        if s.id == "A"
            text!(ax, s.lon + 0.04, s.lat + 0.03,
                  text = "Site $(s.id) — $(s.name)\n$(s.blurb)",
                  align = (:left, :bottom), fontsize = 12)
        else
            text!(ax, s.lon - 0.04, s.lat + 0.03,
                  text = "Site $(s.id) — $(s.name)\n$(s.blurb)",
                  align = (:right, :bottom), fontsize = 12)
        end
    end

    # Townsville reference label
    text!(ax, 146.82, -19.27, text = "Townsville",
          align = (:left, :top), fontsize = 11, color = :sienna)

    xlims!(ax, 146.3, 148.0)
    ylims!(ax, -19.6, -18.0)
    axislegend(ax; position = :rb, framevisible = false, labelsize = 11)

    mkpath(dirname(outpath))
    save(outpath, fig)
    @info "saved $outpath"
end

outpath = joinpath(@__DIR__, "..", "figures", "site_map.png")
draw_map(outpath)
