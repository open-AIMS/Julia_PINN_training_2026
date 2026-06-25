# Warm-starting a neighbour (§7.1) — transfer the Laplace-on-a-disk PINN to a
# mildly elliptical domain by REUSING the trained disk weights as the initial
# parameters, and measure the speed-up against a from-scratch (random-init) run.
#
# Everything is a hand-rolled Lux + Zygote + Optimisers PINN (the Unit 5 §5.3
# pattern): the network is N(x,y); the Laplacian u_xx + u_yy is taken by the
# finite-difference-in-input stencil (Zygote-safe, no nested AD); the domain —
# disk vs ellipse — enters ONLY through where collocation/boundary points sit,
# so the *same* (x,y) network transfers directly. Writes a small results table.
#   julia --project=. units/unit_07/scripts/ellipse_warmstart_solve.jl

using Lux, Random, Zygote, Optimisers, Statistics, Printf

const H  = 1f-2          # finite-difference step for the input-space Laplacian
const A  = 1.05f0        # ellipse semi-axis in x (a = b = 1 would be the disk)
const B  = 0.95f0        # ellipse semi-axis in y  → a very mild ellipse
const λb = 50f0          # soft boundary-condition weight

net = Lux.Chain(Lux.Dense(2 => 32, tanh), Lux.Dense(32 => 32, tanh),
                Lux.Dense(32 => 32, tanh), Lux.Dense(32 => 1))

# Area-uniform interior points and boundary points for a disk stretched to an
# ellipse with semi-axes (a, b). a = b = 1 recovers the unit disk.
function sample_domain(a, b; n_int = 1200, n_bnd = 240, seed = 1)
    rng = Random.MersenneTwister(seed)
    r   = sqrt.(rand(rng, Float32, n_int))        # √U ⇒ uniform in area
    θ   = 2f0π .* rand(rng, Float32, n_int)
    XY  = vcat((a .* r .* cos.(θ))', (b .* r .* sin.(θ))')   # 2 × n_int
    θb  = Float32.(range(0, 2π; length = n_bnd + 1)[1:end-1])
    XYb = vcat((a .* cos.(θb))', (b .* sin.(θb))')           # 2 × n_bnd
    gb  = reshape(sin.(3f0 .* θb), 1, :)          # Dirichlet data u = sin(3θ)
    return XY, XYb, gb
end

# One Adam training run. `ps0` is the initial parameter set — random for a
# cold start, or the trained disk parameters for a warm start.
function train(ps0, st, XY, XYb, gb; iters, lr = 1f-3)
    N(ps, X) = first(Lux.apply(net, X, ps, st))
    ex = Float32[H, 0]; ey = Float32[0, H]
    function loss(ps)
        lap = (N(ps, XY .+ ex) .+ N(ps, XY .- ex) .+
               N(ps, XY .+ ey) .+ N(ps, XY .- ey) .- 4f0 .* N(ps, XY)) ./ H^2
        mean(abs2, lap) + λb * mean(abs2, N(ps, XYb) .- gb)
    end
    ps  = ps0
    opt = Optimisers.setup(Optimisers.Adam(lr), ps)
    hist = Vector{Float32}(undef, iters)
    for i in 1:iters
        g = first(Zygote.gradient(loss, ps))
        opt, ps = Optimisers.update(opt, ps, g)
        hist[i] = loss(ps)
    end
    return ps, hist
end

# First Adam step whose loss drops below `tol` (or `missing` if never).
steps_to(hist, tol) = (k = findfirst(<(tol), hist); k === nothing ? missing : k)

# Relative L² error of the disk fit against the exact harmonic r³ sin 3θ.
function disk_relL2(ps, st)
    N(ps, X) = first(Lux.apply(net, X, ps, st))
    rng = Random.MersenneTwister(7)
    r = sqrt.(rand(rng, Float32, 4000)); θ = 2f0π .* rand(rng, Float32, 4000)
    XY = vcat((r .* cos.(θ))', (r .* sin.(θ))')
    ue = (r .^ 3) .* sin.(3f0 .* θ)
    up = vec(N(ps, XY))
    sqrt(sum(abs2, up .- ue) / sum(abs2, ue))
end

# ── 1. pre-train the disk PINN (the reusable "neighbour") ──────────────
rng = Random.MersenneTwister(0)
ps_init, st = Lux.setup(rng, net)
XYd, XYbd, gbd = sample_domain(1f0, 1f0)
@printf("pre-training disk PINN (3000 Adam)…\n"); flush(stdout)
ps_disk, h_disk = train(ps_init, st, XYd, XYbd, gbd; iters = 3000)
@printf("  disk: final loss %.2e | relative L² vs r³sin3θ = %.3f\n",
        h_disk[end], disk_relL2(ps_disk, st))

# ── 2. the ellipse, two ways: warm-started vs from scratch ─────────────
XYe, XYbe, gbe = sample_domain(A, B)
tol  = 2f-3
EP   = 6000                             # equal budget — long enough that even
snap = 200                              # the cold start reaches the tolerance

ps_warm0 = deepcopy(ps_disk)            # ← warm start: reuse the trained weights
_, h_warm = train(ps_warm0, st, XYe, XYbe, gbe; iters = EP)

rng2 = Random.MersenneTwister(42)
ps_cold0, _ = Lux.setup(rng2, net)      # ← cold start: fresh random init
_, h_cold = train(ps_cold0, st, XYe, XYbe, gbe; iters = EP)

sw = steps_to(h_warm, tol); sc = steps_to(h_cold, tol)
@printf("\nellipse (a=%.2f, b=%.2f), tol=%.0e, %d-step budget:\n", A, B, tol, EP)
@printf("  warm  start: loss@1=%.2e  loss@%d=%.2e  steps→tol=%s\n",
        h_warm[1], snap, h_warm[snap], string(sw))
@printf("  cold  start: loss@1=%.2e  loss@%d=%.2e  steps→tol=%s\n",
        h_cold[1], snap, h_cold[snap], string(sc))
speedup = (sw === missing || sc === missing) ? NaN : sc / sw
@printf("  warm-start speed-up to tol: %.1f×\n", speedup)

# ── 3. write the table the unit includes ───────────────────────────────
out = joinpath(@__DIR__, "..", "output", "ellipse_warmstart.md")
open(out, "w") do io
    println(io, "| ellipse run | initial weights | loss after $snap steps | Adam steps to loss < $(tol) |")
    println(io, "|---|---|---|---|")
    println(io, @sprintf("| warm start | trained disk PINN | %.1e | %s |",
                         h_warm[snap], sw === missing ? "—" : string(sw)))
    println(io, @sprintf("| from scratch | random init | %.1e | %s |",
                         h_cold[snap], sc === missing ? "—" : string(sc)))
end
println("\nwrote ", out)
