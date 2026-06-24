# SINDy on the four Unit-1 tide gauges — exploratory side study (sindy-explore
# branch only; NOT part of the rendered course site).
# State x = [G1,G2,G3,G4]; the bay is a linearised SWE FORCED by the river
# surge ψ(t). We try (A) autonomous SINDy ẋ=Θ(x)Ξ and (B) SINDy-with-control
# ẋ=Θ(x,ψ)Ξ, and show only the controlled fit reconstructs the gauges.
#
# Run from the repo root:  julia --project=. explore/sindy_gauges/sindy_gauges.jl
# Writes sindy_gauges.png next to this script. Deps are all in the course env.
using CSV, DataFrames, LinearAlgebra, Statistics, Printf, Plots

const DATA = joinpath(@__DIR__, "..", "..", "units", "unit_01", "data")
G = CSV.read(joinpath(DATA, "gauges_clean.csv"), DataFrame)   # noise-free truth
P = CSV.read(joinpath(DATA, "psi_truth.csv"),    DataFrame)
t  = Float64.(G.t)
dt = t[2] - t[1]
X  = Matrix(G[:, [:G1, :G2, :G3, :G4]])           # N×4
ψ  = Float64.(P.psi)                               # N forcing
N  = size(X, 1)
@printf("N=%d  dt=%.1f s  span=%.1f h   ψ peak=%.3f\n", N, dt, t[end]/3600, maximum(ψ))

# central-difference derivatives (data is clean → fine); trim the ends
d  = 2:(N-1)
Ẋ  = (X[d.+1, :] .- X[d.-1, :]) ./ (2dt)           # (N-2)×4

# ── candidate libraries (each row = features of state x at one time) ────────
function lib_auto(Xr)                              # constant + linear (deg 1)
    n = size(Xr, 1); cols = Vector{Vector{Float64}}(); nm = String[]
    push!(cols, ones(n)); push!(nm, "1")
    for i in 1:4; push!(cols, Xr[:, i]); push!(nm, "G$i"); end
    reduce(hcat, cols), nm
end

# ── sequential thresholded least squares (the SINDy solver) ────────────────
# Columns of Θ span wildly different scales (const ~31, ψ ~12, quadratics ~0.3),
# so we normalise each column to unit norm, threshold on the SCALE-FAIR
# normalised coefficients (relative to the biggest in each equation), then map
# the surviving coefficients back to physical units.
function stlsq(Θ, Ẋ; λ = 0.05, iters = 12)
    s  = [norm(Θ[:, j]) for j in 1:size(Θ, 2)]      # column norms
    Θn = Θ ./ s'
    W  = Θn \ Ẋ                                       # normalised coefficients
    for _ in 1:iters
        for k in 1:size(Ẋ, 2)
            small = abs.(W[:, k]) .< λ * maximum(abs.(W[:, k]))
            W[small, k] .= 0
            big = .!small
            any(big) && (W[big, k] = Θn[:, big] \ Ẋ[:, k])
        end
    end
    W ./ s                                            # back to physical units
end

# RK4 rollout of a discovered model ẋ = f(x,k) (k = time index for forcing)
function simulate(f, x0, n)
    xs = zeros(4, n); xs[:, 1] = x0; x = copy(x0)
    for k in 1:n-1
        k1 = f(x, k);            k2 = f(x .+ 0.5dt.*k1, k)
        k3 = f(x .+ 0.5dt.*k2, k); k4 = f(x .+ dt.*k3, k+1)
        x = x .+ (dt/6).*(k1 .+ 2k2 .+ 2k3 .+ k4); xs[:, k+1] = x
    end
    xs
end
relerr(xr) = [100*norm(xr[i, :] .- X[:, i]) / norm(X[:, i]) for i in 1:4]

# ════ (A) autonomous SINDy ════════════════════════════════════════════════
Θa, nma = lib_auto(X[d, :])
Ξa = stlsq(Θa, Ẋ; λ = 3e-5)
fa(x, k) = vec(first(lib_auto(reshape(x, 1, 4))) * Ξa)
Xa = simulate(fa, X[1, :], N)
@printf("\n(A) AUTONOMOUS SINDy — terms kept: %d / %d ; per-gauge rel err: %s\n",
        count(!iszero, Ξa), length(Ξa), string(round.(relerr(Xa); digits = 1)))

# ════ (B) SINDy-with-control: append ψ to the library ══════════════════════
function lib_ctrl(Xr, ψr)
    Θ, nm = lib_auto(Xr)
    hcat(Θ, ψr), vcat(nm, "ψ")
end
Θc, nmc = lib_ctrl(X[d, :], ψ[d])
Ξc = stlsq(Θc, Ẋ; λ = 3e-5)
fc(x, k) = vec(first(lib_ctrl(reshape(x, 1, 4), [ψ[clamp(k, 1, N)]])) * Ξc)
Xc = simulate(fc, X[1, :], N)
@printf("(B) CONTROLLED  SINDy — terms kept: %d / %d ; per-gauge rel err: %s\n",
        count(!iszero, Ξc), length(Ξc), string(round.(relerr(Xc); digits = 1)))

# discovered linear coupling A (rows = gauges) and forcing vector b
A = Ξc[2:5, :]'                                    # 4×4  (∂ẋ/∂x)
b = Ξc[end, :]                                     # 4    (∂ẋ/∂ψ)
ev = eigvals(A)
@printf("\neig(A): %s\n", join([@sprintf("%.1e%+.1ei", real(e), imag(e)) for e in ev], ", "))
@printf("max Re(eig) = %.2e  (≤0 ⇒ stable forward sim)\n", maximum(real, ev))
println("Discovered forcing sensitivity b (G1..G4): ", round.(b; digits = 5))
println("Largest |A| couplings:")
for i in 1:4, j in 1:4
    abs(A[i,j]) > 5e-4 && @printf("   dG%d/dt ⟵ %+.4f·G%d\n", i, A[i,j], j)
end

# full system dump (×1e4 for readability), LaTeX-friendly
println("\nA_TIMES_1e4 (rows dG_i/dt, cols G_j):")
for i in 1:4
    println("  ", join([@sprintf("%+8.3f", 1e4*A[i,j]) for j in 1:4], " "))
end
println("b_TIMES_1e4: ", join([@sprintf("%+.3f", 1e4*b[i]) for i in 1:4], " "))
println("EIG_PERIODS_h: ", join([@sprintf("%.2f", 2π/abs(imag(e))/3600) for e in ev if imag(e) > 0], ", "))
println("EIG_DAMP_efold_h: ", join([@sprintf("%.1f", -1/real(e)/3600) for e in ev if imag(e) > 0], ", "))

# ── figure: data vs autonomous vs controlled, per gauge ────────────────────
plt = plot(layout = (2, 2), size = (1000, 640))
for i in 1:4
    plot!(plt[i], t./3600, X[:, i]; lw = 3, c = :black, label = i==1 ? "gauge data" : "",
          title = "G$i", xlabel = "time (h)", ylabel = "η (m)", framestyle = :box)
    plot!(plt[i], t./3600, Xa[i, :]; lw = 2, c = :firebrick, ls = :dash,
          label = i==1 ? "autonomous SINDy" : "")
    plot!(plt[i], t./3600, Xc[i, :]; lw = 2, c = :dodgerblue,
          label = i==1 ? "SINDy + ψ control" : "")
end
savefig(plt, joinpath(@__DIR__, "sindy_gauges.png"))
println("\nwrote sindy_gauges.png")
