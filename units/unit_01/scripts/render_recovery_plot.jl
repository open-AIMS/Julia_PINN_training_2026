#!/usr/bin/env julia
# Render the final ψ(t) recovery comparison figure for Unit 1.
# Top panel:    truth ψ(t)  vs  adjoint inverse  vs  vanilla PINN.
# Bottom panel: observed-vs-predicted at gauges G1 and G2.
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

p_g1 = plot(t_hr_obs, obs.G1;
    seriestype = :scatter, ms = 1.7, c = :black, alpha = 0.45,
    label = "G1 observed", legend = :topright, legendfontsize = 7,
    xlabel = "", ylabel = "η at G1 (m)")
plot!(p_g1, t_hr_psi, gpred_adj.G1_pred; label = "adjoint pred",
    lw = 2, c = :steelblue)
plot!(p_g1, t_hr_pinn, gpred_pinn.G1_pred; label = "naive PINN pred",
    lw = 1.8, c = :darkorange, ls = :dash)

p_g2 = plot(t_hr_obs, obs.G2;
    seriestype = :scatter, ms = 1.7, c = :black, alpha = 0.45,
    label = "G2 observed", legend = :topright, legendfontsize = 7,
    xlabel = "time (h)", ylabel = "η at G2 (m)")
plot!(p_g2, t_hr_psi, gpred_adj.G2_pred; label = "adjoint pred",
    lw = 2, c = :steelblue)
plot!(p_g2, t_hr_pinn, gpred_pinn.G2_pred; label = "naive PINN pred",
    lw = 1.8, c = :darkorange, ls = :dash)

p = plot(p_top, p_g1, p_g2; layout = grid(3, 1, heights = [0.50, 0.25, 0.25]),
         size = (820, 760), link = :x,
         left_margin = 8 * Plots.mm)

savefig(p, joinpath(FIG_DIR, "inverse_recovery.png"))
println("wrote figures/inverse_recovery.png")
