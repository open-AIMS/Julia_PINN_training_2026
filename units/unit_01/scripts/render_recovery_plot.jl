#!/usr/bin/env julia
# Render the final ψ(t) recovery comparison figure for Unit 1.
# Top panel:    truth ψ(t)  vs  adjoint inverse  vs  vanilla PINN.
# Bottom:       observed-vs-predicted at all four gauges (G1–G4), 2×2 grid.
# Output: figures/inverse_recovery.png

using CSV, DataFrames, Plots

const HERE     = @__DIR__
const DATA_DIR = joinpath(HERE, "..", "data")
const FIG_DIR  = joinpath(HERE, "..", "figures")

pinn = CSV.read(joinpath(DATA_DIR, "psi_recovered.csv"),          DataFrame)
adj  = CSV.read(joinpath(DATA_DIR, "psi_recovered_adjoint.csv"),  DataFrame)
obs  = CSV.read(joinpath(DATA_DIR, "gauges_observed.csv"),        DataFrame)
gpred_adj  = CSV.read(joinpath(DATA_DIR, "gauges_adjoint_pred.csv"), DataFrame)
gpred_pinn = CSV.read(joinpath(DATA_DIR, "gauges_pinn_pred.csv"),    DataFrame)

t_hr_psi = adj.t  ./ 3600
t_hr_obs = obs.t  ./ 3600
t_hr_pinn = pinn.t ./ 3600

gr()

p_top = plot(t_hr_psi, adj.psi_truth;
    label = "truth ψ(t)",
    lw    = 3.2, c = :red,
    xlabel = "", ylabel = "ψ (m)",
    legend = :topright, legendfontsize = 8,
    title  = "Recovering the Brisbane River surge from sparse bay gauges",
    titlefontsize = 11)
plot!(p_top, t_hr_psi, adj.psi_recovered;
    label = "adjoint inverse  (uses the SWE solver as forward map)",
    lw = 2.2, c = :steelblue)
plot!(p_top, t_hr_pinn, pinn.psi_recovered;
    label = "naive PINN  (smooth-MLP η-network, ψ read at source)",
    lw = 2.0, c = :darkorange, ls = :dash)

# 2x2 gauge grid: top row (G1, G2), bottom row (G3, G4).
# Bottom-row panels carry the x-axis label; legends only on G1 to avoid clutter.
function gauge_panel(id::Symbol; xlabel="", show_legend=false)
    obs_col = obs[!, id]
    adj_col = gpred_adj[!, Symbol(string(id) * "_pred")]
    pinn_col = gpred_pinn[!, Symbol(string(id) * "_pred")]
    p = plot(t_hr_obs, obs_col;
        seriestype = :scatter, ms = 1.5, c = :black, alpha = 0.45,
        label = show_legend ? "observed" : "",
        legend = show_legend ? :topright : false,
        legendfontsize = 7,
        xlabel = xlabel, ylabel = "η at $(id) (m)")
    plot!(p, t_hr_psi, adj_col;
        label = show_legend ? "adjoint pred" : "",
        lw = 1.8, c = :steelblue)
    plot!(p, t_hr_pinn, pinn_col;
        label = show_legend ? "naive PINN pred" : "",
        lw = 1.6, c = :darkorange, ls = :dash)
    return p
end

p_g1 = gauge_panel(:G1; show_legend = true)
p_g2 = gauge_panel(:G2)
p_g3 = gauge_panel(:G3; xlabel = "time (h)")
p_g4 = gauge_panel(:G4; xlabel = "time (h)")

p_gauges = plot(p_g1, p_g2, p_g3, p_g4;
                layout = (2, 2), link = :x)

p = plot(p_top, p_gauges;
         layout = grid(2, 1, heights = [0.40, 0.60]),
         size = (900, 880),
         left_margin = 8 * Plots.mm)

savefig(p, joinpath(FIG_DIR, "inverse_recovery.png"))
println("wrote figures/inverse_recovery.png")
