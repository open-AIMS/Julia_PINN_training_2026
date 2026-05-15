# Reference 1D vertical heat-transport column solver (finite-difference,
# method-of-lines). See unit_11.qmd for the model specification.
#
# Scenarios (per unit_11.qmd §11.8):
#   1. pure diffusion to steady state
#   2. diurnal cycle (Q_SW + body source S)
#   3. add upwelling (w(z))
#   4. storm fingerprint (gust modulates w, κ, Q_SW around day 10)
#
# Run a scenario:
#   include("column_fd.jl"); sol = run_scenario(scenario_1());
#   plot_scn1(sol)
#
# NOTE on the surface BC sign convention: §11.4 writes
#   -κ ∂T/∂z|₀ = Q_np/(ρ cp)
# with the prose "Q_np > 0 = heat into ocean". Those two are inconsistent
# (with Q_np = -Q_cool < 0, the formula yields surface warmer than deep
# in steady state). We implement the formula *literally* so the analytic
# check below matches; the physical interpretation should be reconciled
# in the qmd before the PINN forward problem.

using ModelingToolkit, MethodOfLines, OrdinaryDiffEq
using DomainSets: ClosedInterval
using CairoMakie
using Printf

# ── reference parameters (unit_11.qmd §11.6) ────────────────────────────
const PARAMS = (
    H         = 100.0,    # column depth (m)
    ρ         = 1025.0,   # seawater density (kg/m³)
    cp        = 4000.0,   # specific heat (J/(kg·K))
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
)

# ── initial profile ─────────────────────────────────────────────────────
T0(z, p=PARAMS) = 0.5*(p.T_surface + p.T_deep) +
                  0.5*(p.T_surface - p.T_deep) * tanh((z - p.z_t)/p.δ_t)

# ── vertical velocity profiles ──────────────────────────────────────────
w_zero(z, t, p)       = 0.0
w_upwelling(z, t, p)  = p.w0 * sin(π * (z + p.H) / p.H)

# ── eddy diffusivity profiles ───────────────────────────────────────────
κ_const(z, t, p)   = p.κ_m
κ_profile(z, t, p) = p.κ_b + (p.κ_m - p.κ_b) * exp(z / p.h_m)

# ── surface forcing ─────────────────────────────────────────────────────
# Smooth diurnal: use (1 + cos)/2 instead of max(0, cos) to keep the RHS
# differentiable for MTK. Same daily mean energy as max(0, cos)·(2/π)·QSW_max
# would give, but a smooth half-cosine — close enough for the toy model.
QSW_diurnal(t, p) = p.QSW_max * 0.5 * (1.0 + cos(2π * t / p.τ_d))
Q_np_steady(t, p) = -p.Q_cool

# Body source from Beer–Lambert: I(z,t) = Q_SW(t) e^{z/ζ},
# S = (1/(ρcp)) ∂I/∂z = Q_SW(t)/(ζ ρ cp) · e^{z/ζ}.
S_off(z, t, p)     = 0.0
S_diurnal(z, t, p) = QSW_diurnal(t, p) * exp(z / p.ζ) / (p.ζ * p.ρ * p.cp)

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
        -κ_top * Dz(T(0.0, t)) ~ Qnp_sym / (p.ρ * p.cp),
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
    return sol
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

# Scenario 4 (storm) — placeholder; gust modulation TBD once the SWE
# driver is in place. For now reuses scenario 3 with the profile κ.
scenario_4() = (
    name   = "storm fingerprint (placeholder)",
    w      = w_upwelling,
    κ      = κ_profile,
    S      = S_diurnal,
    Q_np   = Q_np_steady,
    Tf     = 20 * 86400.0,
    saveat = 3600.0,
    alg    = QNDF(),
)

# ── analytic check for scenario 1 ───────────────────────────────────────
# Steady state of (κ T_z)_z = 0 with T(-H)=T_deep and -κ T_z|₀ = Q_np/(ρcp):
#   T(z) = T_deep + (Q_cool/(κ_m ρ cp)) · (z + H)
# (Using Q_np = -Q_cool. See the BC-sign NOTE at the top.)
T_analytic_scn1(z, p=PARAMS) =
    p.T_deep + p.Q_cool / (p.κ_m * p.ρ * p.cp) * (z + p.H)

# ── plotting ────────────────────────────────────────────────────────────
"Final-time T(z) vs analytic steady state."
function plot_scn1(sol; outpath=joinpath(@__DIR__, "scn1_steady.png"))
    p = PARAMS
    zg = sol[sol.ivs[1]]      # spatial grid (z)
    Tf = sol[sol.dvs[1]][:, end]
    fig = Figure(size=(700, 500))
    ax = Axis(fig[1,1], xlabel="T (°C)", ylabel="z (m)",
              title="Scenario 1: pure diffusion to (near-)steady state")
    lines!(ax, T_analytic_scn1.(zg), zg;
           linestyle=:dash, linewidth=2, label="analytic")
    lines!(ax, Tf, zg; linewidth=2, label="MOL final")
    lines!(ax, T0.(zg), zg;
           color=:gray, linewidth=1, label="IC")
    axislegend(ax, position=:rb)
    save(outpath, fig)
    @info "saved $outpath"
    fig
end

"Near-surface (z, t) heatmap for scenario 2."
function plot_scn2(sol; zmax=10.0,
                   outpath=joinpath(@__DIR__, "scn2_diurnal.png"))
    zg = sol[sol.ivs[1]]
    tg = sol[sol.ivs[2]]
    Tg = sol[sol.dvs[1]]                  # size (Nz, Nt)
    keep = zg .>= -zmax
    fig = Figure(size=(900, 450))
    ax = Axis(fig[1,1], xlabel="t (days)", ylabel="z (m)",
              title="Scenario 2: diurnal warm layer")
    hm = heatmap!(ax, tg ./ 86400.0, zg[keep], Tg[keep, :]')
    Colorbar(fig[1,2], hm, label="T (°C)")
    save(outpath, fig)
    @info "saved $outpath"
    fig
end

# ── entry point when run as script ──────────────────────────────────────
if abspath(PROGRAM_FILE) == @__FILE__
    sol1 = run_scenario(scenario_1())
    plot_scn1(sol1)
end
