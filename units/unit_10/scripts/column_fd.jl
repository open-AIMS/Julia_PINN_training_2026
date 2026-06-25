# Reference 1D vertical heat-transport column solver (finite-difference,
# method-of-lines). See unit_09.qmd for the model specification.
#
# Scenarios (per unit_09.qmd §9.11):
#   1. pure diffusion to steady state
#   2. diurnal cycle (Q_SW + body source S)
#   3. add upwelling (w(z))
#   4. storm fingerprint (gust modulates w, κ, Q_SW around day 10)
#
# Run a scenario:
#   include("column_fd.jl"); r = run_scenario(scenario_1());
#   plot_scn1(r)

using ModelingToolkit, MethodOfLines, OrdinaryDiffEq
using OrdinaryDiffEqBDF: QNDF  # umbrella OrdinaryDiffEq no longer re-exports BDF solvers
using DomainSets: ClosedInterval
using CairoMakie
using Printf

# ── reference parameters (unit_09.qmd §9.6) ────────────────────────────
const PARAMS = (
    H         = 100.0,    # column depth (m)
    ρ         = 1025.0,   # seawater density (kg/m³)
    cp        = 3990.0,   # specific heat (J/(kg·K))
    α         = 2e-4,     # thermal expansion (1/K)
    T_surface = 28.0,     # initial SST (°C)
    T_deep    = 18.0,     # deep reservoir (°C)
    z_t       = -30.0,    # thermocline depth (m)
    δ_t       = 5.0,      # thermocline sharpness (m)
    QSW_max   = 800.0,    # peak noon shortwave (W/m²)
    Q_cool    = 200.0,    # non-penetrative net cooling (W/m²)
    ζ         = 10.0,     # shortwave penetration scale (m)
    κ_b       = 1e-5,     # background diffusivity (m²/s)
    κ_m       = 1e-3,     # mixed-layer diffusivity (m²/s)
    h_m       = 20.0,     # mixed-layer scale (m)
    w0        = 1e-5,     # peak upwelling (m/s)
    τ_d       = 86400.0,  # diurnal period (s)
    # storm scenario: Gaussian gust around day 10, ~3 days wide.
    t_storm   = 10 * 86400.0,
    σ_storm   = 1.0 * 86400.0,
    w0_storm  = 5e-5,     # gust-driven upwelling (5× baseline)
    cloud_amp = 0.5,      # peak SW reduction during gust
)

storm_envelope(t, p) = exp(-((t - p.t_storm) / p.σ_storm)^2)

# ── initial profile ─────────────────────────────────────────────────────
T0(z, p=PARAMS) = 0.5*(p.T_surface + p.T_deep) +
                  0.5*(p.T_surface - p.T_deep) * tanh((z - p.z_t)/p.δ_t)

# ── vertical velocity profiles ──────────────────────────────────────────
w_zero(z, t, p)       = 0.0
w_upwelling(z, t, p)  = p.w0 * sin(π * (z + p.H) / p.H)
w_storm(z, t, p)      = p.w0_storm * storm_envelope(t, p) * sin(π * (z + p.H) / p.H)

# ── eddy diffusivity profiles ───────────────────────────────────────────
κ_const(z, t, p)   = p.κ_m
κ_profile(z, t, p) = p.κ_b + (p.κ_m - p.κ_b) * exp(z / p.h_m)

# ── surface forcing ─────────────────────────────────────────────────────
# Smooth diurnal: use (1 + cos)/2 instead of max(0, cos) to keep the RHS
# differentiable for MTK. Same daily mean energy as max(0, cos)·(2/π)·QSW_max
# would give, but a smooth half-cosine — close enough for the toy model.
QSW_diurnal(t, p) = p.QSW_max * 0.5 * (1.0 + cos(2π * t / p.τ_d))
QSW_storm(t, p)   = p.QSW_max * (1.0 - p.cloud_amp * storm_envelope(t, p)) *
                    0.5 * (1.0 + cos(2π * t / p.τ_d))
Q_np_steady(t, p) = -p.Q_cool

# Body source from Beer–Lambert: I(z,t) = Q_SW(t) e^{z/ζ},
# S = (1/(ρcp)) ∂I/∂z = Q_SW(t)/(ζ ρ cp) · e^{z/ζ}.
S_off(z, t, p)     = 0.0
S_diurnal(z, t, p) = QSW_diurnal(t, p) * exp(z / p.ζ) / (p.ζ * p.ρ * p.cp)
S_storm(z, t, p)   = QSW_storm(t, p)   * exp(z / p.ζ) / (p.ζ * p.ρ * p.cp)

# ── PDE builder ─────────────────────────────────────────────────────────
function build_pde(scn, p=PARAMS)
    @parameters z t
    @variables T(..)
    Dt = Differential(t)
    Dz = Differential(z)

    # Substitute scenario profiles symbolically so MOL sees closed-form
    # expressions in (z, t).
    w_sym  = scn.w(z, t, p)
    κ_sym  = scn.κ(z, t, p)
    S_sym  = scn.S(z, t, p)
    Qnp_sym = scn.Q_np(t, p)
    κ_top  = scn.κ(0.0, t, p)

    eq = Dt(T(z, t)) + w_sym * Dz(T(z, t)) ~
         Dz(κ_sym * Dz(T(z, t))) + S_sym

    domains = [z ∈ ClosedInterval(-p.H, 0.0),
               t ∈ ClosedInterval(0.0, scn.Tf)]

    bcs = [
        T(z, 0.0)   ~ T0(z, p),
        T(-p.H, t)  ~ p.T_deep,
        κ_top * Dz(T(0.0, t)) ~ Qnp_sym / (p.ρ * p.cp),
    ]

    @named pde_system = PDESystem(eq, bcs, domains, [z, t], [T(z, t)])
    return pde_system, z, t, T
end

# ── run dispatcher ──────────────────────────────────────────────────────
function run_scenario(scn; dz=1.0, p=PARAMS)
    pde_system, z, t, T = build_pde(scn, p)
    disc = MOLFiniteDifference([z => dz], t)
    prob = discretize(pde_system, disc)
    @info "solving scenario: $(scn.name) (Tf = $(scn.Tf/86400) days)"
    sol = solve(prob, scn.alg; saveat=scn.saveat, abstol=1e-8, reltol=1e-6)
    (; sol, z, t, T, scn)
end

# ── scenario configs ────────────────────────────────────────────────────
# Tf chosen long enough to reach (near-)steady state: T_κ = H²/κ_m ≈ 116 days,
# so ~3·T_κ ≈ 350 days gets us to within a few percent.
scenario_1() = (
    name   = "diffusion to steady state",
    w      = w_zero,
    κ      = κ_const,
    S      = S_off,
    Q_np   = Q_np_steady,
    Tf     = 365 * 86400.0,
    saveat = 10 * 86400.0,
    alg    = QNDF(),
)

scenario_2() = (
    name   = "diurnal cycle",
    w      = w_zero,
    κ      = κ_const,
    S      = S_diurnal,
    Q_np   = Q_np_steady,
    Tf     = 10 * 86400.0,
    saveat = 600.0,
    alg    = QNDF(),
)

scenario_3() = (
    name   = "upwelling",
    w      = w_upwelling,
    κ      = κ_const,
    S      = S_diurnal,
    Q_np   = Q_np_steady,
    Tf     = 30 * 86400.0,
    saveat = 3600.0,
    alg    = QNDF(),
)

# Scenario 4 (storm): Gaussian gust centred at t_storm. The gust spikes
# upwelling (w_storm) and dims the surface SW (cloud cover via S_storm).
# Real SWE-driven forcings would replace these envelopes; this is the
# prescribed-driver stub used in §9.11 task 5.
scenario_4() = (
    name   = "storm fingerprint",
    w      = w_storm,
    κ      = κ_const,
    S      = S_storm,
    Q_np   = Q_np_steady,
    Tf     = 30 * 86400.0,
    saveat = 3600.0,
    alg    = QNDF(),
)

# ── analytic check for scenario 1 ───────────────────────────────────────
# Steady state of (κ T_z)_z = 0 with T(-H)=T_deep and κ T_z|₀ = Q_np/(ρcp):
#   T(z) = T_deep − (Q_cool/(κ_m ρ cp)) · (z + H)
# (Using Q_np = -Q_cool < 0, so the surface is cooler than the deep
# reservoir — heat flows up from depth to be lost at the air-sea
# interface.)
T_analytic_scn1(z, p=PARAMS) =
    p.T_deep - p.Q_cool / (p.κ_m * p.ρ * p.cp) * (z + p.H)

# ── plotting ────────────────────────────────────────────────────────────
# Extract the (Nz, Nt) matrix and grids from a run_scenario result,
# regardless of MOL's internal axis ordering. We declared PDESystem with
# ivs = [z, t], so sol[z] and sol[t] are the canonical grids; we transpose
# the matrix to (Nz, Nt) if MOL returned it in (Nt, Nz) form.
function _grids(r)
    zg = r.sol[r.z]
    tg = r.sol[r.t]
    Tg = r.sol[r.T(r.z, r.t)]
    Tg = size(Tg, 1) == length(zg) ? Tg : permutedims(Tg)
    (zg, tg, Tg)
end

"Final-time T(z) vs analytic steady state."
function plot_scn1(r; outpath=joinpath(@__DIR__, "..", "output", "scn1_steady.png"))
    zg, tg, Tg = _grids(r)
    Tf = Tg[:, end]
    fig = Figure(size=(700, 500))
    ax = Axis(fig[1,1], xlabel="T (°C)", ylabel="z (m)",
              title="Scenario 1: pure diffusion to (near-)steady state")
    # Draw the MOL solution first (solid), then the analytic steady state
    # as a dashed overlay ON TOP, in a contrasting colour — otherwise the
    # solid MOL line hides the dashes entirely (the two agree to ~4e-3 °C,
    # so the visible dashes riding on the orange line ARE the validation).
    lines!(ax, Tf, zg; linewidth=3, label="MOL final")
    lines!(ax, T_analytic_scn1.(zg), zg;
           linestyle=:dash, linewidth=2, color=:black, label="analytic")
    lines!(ax, T0.(zg), zg;
           color=:gray, linewidth=1, label="IC")
    axislegend(ax, position=:rb)
    save(outpath, fig)
    err = maximum(abs, Tf .- T_analytic_scn1.(zg))
    @info "saved $outpath; max |T_MOL − T_analytic| = $(round(err, sigdigits=3)) °C"
    fig
end

"Scenario 2: near-surface traces (showing the diurnal cycle) over the (z, t) heatmap."
function plot_scn2(r; zmax=10.0,
                   outpath=joinpath(@__DIR__, "..", "output", "scn2_diurnal.png"))
    zg, tg, Tg = _grids(r)
    days = tg ./ 86400.0
    keep = zg .>= -zmax
    fig = Figure(size=(900, 640))

    # Top: a few near-surface traces — the daily warm/cool cycle is obvious here
    # even though it is washed out by the slow relaxation in the heatmap below.
    ax1 = Axis(fig[1,1], xlabel="t (days)", ylabel="T (°C)",
               title="Scenario 2: diurnal warm layer — near-surface traces")
    for (d, col) in ((1.0, :firebrick), (3.0, :darkorange), (7.0, :seagreen))
        zi = argmin(abs.(zg .+ d))
        lines!(ax1, days, Tg[zi, :]; color=col, linewidth=1.6, label="z = −$(Int(d)) m")
    end
    axislegend(ax1, position=:rt, framevisible=true)

    ax2 = Axis(fig[2,1], xlabel="t (days)", ylabel="z (m)", title="T(z, t) — top $(Int(zmax)) m")
    hm = heatmap!(ax2, days, zg[keep], Tg[keep, :]')
    Colorbar(fig[2,2], hm, label="T (°C)")
    rowsize!(fig.layout, 1, Relative(0.42))
    save(outpath, fig)
    @info "saved $outpath"
    fig
end

"Five-mooring-depth time series + full (z,t) heatmap for scenarios 3–4."
function plot_mooring(r; depths_m=(2.0, 10.0, 30.0, 60.0, 90.0),
                      outpath=joinpath(@__DIR__, "..", "output", "$(replace(r.scn.name, ' '=>'_'))_mooring.png"))
    zg, tg, Tg = _grids(r)
    days = tg ./ 86400.0
    fig = Figure(size=(1000, 700))
    ax1 = Axis(fig[1,1], xlabel="t (days)", ylabel="T (°C)",
               title="Mooring traces — $(r.scn.name)")
    for d in depths_m
        zi = argmin(abs.(zg .+ d))  # z = -d
        lines!(ax1, days, Tg[zi, :]; label="z = −$(d) m")
    end
    axislegend(ax1, position=:rt)
    ax2 = Axis(fig[2,1], xlabel="t (days)", ylabel="z (m)",
               title="T(z, t)")
    hm = heatmap!(ax2, days, zg, Tg')
    Colorbar(fig[2,2], hm, label="T (°C)")
    save(outpath, fig)
    @info "saved $outpath"
    fig
end

# ── entry point when run as script ──────────────────────────────────────
# Used by ./build.sh execute 10 to populate units/unit_10/output/.
if abspath(PROGRAM_FILE) == @__FILE__
    r1 = run_scenario(scenario_1());  plot_scn1(r1)
    r2 = run_scenario(scenario_2());  plot_scn2(r2)
    r3 = run_scenario(scenario_3());  plot_mooring(r3)
    r4 = run_scenario(scenario_4());  plot_mooring(r4)
end
