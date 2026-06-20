# Shared helper for all Unit 1 bay MAPS (bathymetry, surge frames, GPU field).
#
# Orientation convention — stated once, used everywhere:
#   * Figures are LANDSCAPE (the bay's long N–S axis runs horizontally), so they
#     fit the page instead of being a tall, mostly-empty strip.
#   * NORTH points LEFT  (← N)   and   EAST points UP  (↑ E).
# Model arrays are (NY, NX) = (north, east). We transpose the field and put north
# on a *flipped* horizontal axis and east on the vertical axis to get that view.

using Plots

# (NY,NX) model field -> (NX,NY) matrix for heatmap(north_km, east_km, ·)
rot(A) = permutedims(A)

# A clear "North is to the left" compass, drawn in the open NE corner (top-left
# of the rotated frame). We use a bold text glyph rather than a plotted arrow:
# under `xflip` a plotted arrowhead points the wrong way, but text is reliable.
function north_arrow!(p, Ln, Le)
    annotate!(p, 0.93 * Ln, 0.92 * Le, text("← N", 14, :black, :left))
end

"""
    bay_map(field, mask, dx_km; clims, cmap, title, clabel, gauge_e, gauge_n,
            gauge_id, river_e, river_n, land_color)

Rotated landscape heatmap of a `(NY,NX)` `field`. Land (`mask .== 0`) is greyed,
optional gauge/river markers are drawn, and a North arrow points left. Marker
coordinate args are EAST / NORTH in km (model convention).
"""
function bay_map(field, mask, dx_km;
        clims, cmap, title = "", clabel = "",
        gauge_e = Float64[], gauge_n = Float64[], gauge_id = String[],
        river_e = nothing, river_n = nothing,
        land_color = RGB(0.82, 0.82, 0.82))

    NY, NX = size(field)
    nkm = [(j - 0.5) * dx_km for j in 1:NY]      # north axis (horizontal, flipped)
    ekm = [(i - 0.5) * dx_km for i in 1:NX]      # east axis  (vertical)
    Ln, Le = NY * dx_km, NX * dx_km

    # Land = NaN, shown via the grey plot background (a single heatmap series so
    # the colour axis stays tied to `clims` and the colourbar renders correctly —
    # a second land-overlay heatmap would hijack the shared colour range).
    Z = Float64.(field); Z[mask .== 0] .= NaN
    p = heatmap(nkm, ekm, rot(Z);
        c = cmap, clims = clims, xflip = true, aspect_ratio = 1,
        size = (1040, 650), xlims = (0, Ln), ylims = (0, Le),
        xlabel = "north  (km)        ← N", ylabel = "east  (km)    E →",
        colorbar = true, colorbar_title = clabel, legend = false,
        title = title, titlefontsize = 11,
        right_margin = 9Plots.mm, left_margin = 4Plots.mm,
        background_color_inside = land_color, fontfamily = "Helvetica")

    if !isempty(gauge_e)
        scatter!(p, gauge_n, gauge_e;
            marker = :circle, c = :gold, ms = 6, msc = :black, msw = 1.2)
        for k in eachindex(gauge_id)
            annotate!(p, gauge_n[k], gauge_e[k] + 1.8, text(gauge_id[k], 8, :black))
        end
    end
    if river_e !== nothing
        scatter!(p, [river_n], [river_e];
            marker = :utriangle, c = :crimson, ms = 9, msc = :black, msw = 1.2)
        annotate!(p, river_n, river_e + 2.4, text("ψ", 11, :crimson))
    end

    north_arrow!(p, Ln, Le)
    return p
end
