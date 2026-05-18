# Poisson on the unit disk — three methods side by side.
#
# Problem:  -Δu = 1 on r < 1, u = 0 on r = 1.
# Closed form: u(r) = (1 - r²)/4.  The constant-source case is the
# smallest 2-D Poisson PINN with a one-line analytical benchmark.
#
# The same network (2 → 32 → 32 → 1, tanh) and the same RNG seed are
# used for all three runs, so the difference is the loss, not the
# architecture or initialisation. Hand-rolled in Lux + Zygote +
# ForwardDiff (same style as the §5.2 ODE PINN).
#
#   1. Forward PINN  — equation + BC, no data.
#   2. Inverse PINN  — equation + BC + sparse data; the source
#                       magnitude c is unknown, recovered jointly with
#                       the field u_θ.
#   3. Data-only MLP — same scattered measurements + boundary
#                       observations, no PDE term.
#
# Run via ./build.sh execute 5 (writes ../output/pinn_poisson_disk.md).

using Lux, Random, Zygote, ForwardDiff, Optimisers, Statistics, Printf

const SEED = 0

# ── shared bits ─────────────────────────────────────────────────────
function fresh_net()
    rng = Random.MersenneTwister(SEED)
    m = Lux.Chain(
        Lux.Dense(2 => 32, tanh),
        Lux.Dense(32 => 32, tanh),
        Lux.Dense(32 => 1),
    )
    ps, st = Lux.setup(rng, m)
    m, ps, st
end

u_net(m, x, y, ps, st) = first(first(m([x, y], ps, st)))

# Laplacian via forward-over-forward AD on (x, y); Zygote then
# differentiates the loss with respect to the parameters.
function lap(m, x, y, ps, st)
    uxx = ForwardDiff.derivative(
        a -> ForwardDiff.derivative(b -> u_net(m, b, y, ps, st), a), x)
    uyy = ForwardDiff.derivative(
        a -> ForwardDiff.derivative(b -> u_net(m, x, b, ps, st), a), y)
    uxx + uyy
end

function disk_samples(rng, N_int, N_bdy)
    r = sqrt.(rand(rng, N_int))                   # uniform area
    θ = 2π .* rand(rng, N_int)
    xs_int = Float32.(r .* cos.(θ))
    ys_int = Float32.(r .* sin.(θ))
    θb = 2π .* rand(rng, N_bdy)
    xs_bdy = Float32.(cos.(θb))
    ys_bdy = Float32.(sin.(θb))
    xs_int, ys_int, xs_bdy, ys_bdy
end

exact(x, y) = (1 - x^2 - y^2) / 4

function eval_grid(m, ps, st)
    errs = Float64[]
    n = 60
    for x in range(-1f0, 1f0; length = n), y in range(-1f0, 1f0; length = n)
        x^2 + y^2 >= 0.999 && continue
        push!(errs, abs(u_net(m, x, y, ps, st) - exact(x, y)))
    end
    maximum(errs), mean(errs)
end

# ── 1. Forward PINN ─────────────────────────────────────────────────
function forward_pinn(; N_int = 128, N_bdy = 48, epochs = 1200,
                       λ_bc = 100f0, lr = 5f-3)
    rng = Random.MersenneTwister(SEED + 1)
    m, ps, st = fresh_net()
    xs_int, ys_int, xs_bdy, ys_bdy = disk_samples(rng, N_int, N_bdy)

    function loss(ps)
        L_pde = sum(i -> (lap(m, xs_int[i], ys_int[i], ps, st) + 1f0)^2,
                    1:N_int) / N_int
        L_bc  = sum(i -> u_net(m, xs_bdy[i], ys_bdy[i], ps, st)^2,
                    1:N_bdy) / N_bdy
        L_pde + λ_bc * L_bc
    end
    opt = Optimisers.setup(Optimisers.Adam(lr), ps)
    for _ in 1:epochs
        g = first(Zygote.gradient(loss, ps))
        opt, ps = Optimisers.update(opt, ps, g)
    end
    eval_grid(m, ps, st)
end

# ── 2. Inverse PINN ─────────────────────────────────────────────────
function inverse_pinn(; N_int = 128, N_bdy = 48, N_d = 20,
                       σ_noise = 0.005f0, epochs = 1200,
                       λ_bc = 100f0, λ_d = 200f0, lr = 5f-3)
    rng = Random.MersenneTwister(SEED + 2)
    m, ps_net, st = fresh_net()
    xs_int, ys_int, xs_bdy, ys_bdy = disk_samples(rng, N_int, N_bdy)

    r = sqrt.(rand(rng, N_d))
    θ = 2π .* rand(rng, N_d)
    xs_d = Float32.(r .* cos.(θ))
    ys_d = Float32.(r .* sin.(θ))
    u_obs = Float32.(exact.(xs_d, ys_d)) .+ σ_noise .* randn(rng, Float32, N_d)

    ps_all = (net = ps_net, c = Float32[0.5])     # c starts off the truth
    function loss(ps_all)
        c = ps_all.c[1]
        L_pde = sum(i -> (lap(m, xs_int[i], ys_int[i], ps_all.net, st) + c)^2,
                    1:N_int) / N_int
        L_bc  = sum(i -> u_net(m, xs_bdy[i], ys_bdy[i], ps_all.net, st)^2,
                    1:N_bdy) / N_bdy
        L_d   = sum(i -> (u_net(m, xs_d[i], ys_d[i], ps_all.net, st) - u_obs[i])^2,
                    1:N_d) / N_d
        L_pde + λ_bc * L_bc + λ_d * L_d
    end
    opt = Optimisers.setup(Optimisers.Adam(lr), ps_all)
    for _ in 1:epochs
        g = first(Zygote.gradient(loss, ps_all))
        opt, ps_all = Optimisers.update(opt, ps_all, g)
    end
    em, ea = eval_grid(m, ps_all.net, st)
    em, ea, ps_all.c[1]
end

# ── 3. Data-only MLP ────────────────────────────────────────────────
function data_only(; N_d = 20, N_bd = 32, σ_noise = 0.005f0,
                    epochs = 3000, lr = 5f-3)
    rng = Random.MersenneTwister(SEED + 3)
    m, ps, st = fresh_net()

    r = sqrt.(rand(rng, N_d))
    θ = 2π .* rand(rng, N_d)
    xs_d = Float32.(r .* cos.(θ))
    ys_d = Float32.(r .* sin.(θ))
    u_obs = Float32.(exact.(xs_d, ys_d)) .+ σ_noise .* randn(rng, Float32, N_d)

    θb = 2π .* rand(rng, N_bd)
    xs_b = Float32.(cos.(θb))
    ys_b = Float32.(sin.(θb))

    xs_all = vcat(xs_d, xs_b)
    ys_all = vcat(ys_d, ys_b)
    u_all  = vcat(u_obs, zeros(Float32, N_bd))
    N      = length(xs_all)

    function loss(ps)
        sum(i -> (u_net(m, xs_all[i], ys_all[i], ps, st) - u_all[i])^2, 1:N) / N
    end
    opt = Optimisers.setup(Optimisers.Adam(lr), ps)
    for _ in 1:epochs
        g = first(Zygote.gradient(loss, ps))
        opt, ps = Optimisers.update(opt, ps, g)
    end
    eval_grid(m, ps, st)
end

# ── run all three ───────────────────────────────────────────────────
println("Poisson on the unit disk — three methods")
println("  -Δu = 1, u|∂Ω = 0, exact u(r) = (1 - r²)/4")
println("  network: 2 → 32 → 32 → 1 tanh, Adam @ 5e-3")
println()

t = @elapsed (fwd_max, fwd_mean) = forward_pinn()
@printf "[1/3] forward PINN  (%.1fs)\n" t
@printf "      max|u-exact| = %0.3e   mean = %0.3e\n\n" fwd_max fwd_mean

t = @elapsed (inv_max, inv_mean, ĉ) = inverse_pinn()
@printf "[2/3] inverse PINN  (%.1fs)\n" t
@printf "      max|u-exact| = %0.3e   mean = %0.3e   ĉ = %0.4f (true 1.000)\n\n" inv_max inv_mean ĉ

t = @elapsed (dat_max, dat_mean) = data_only()
@printf "[3/3] data-only MLP (%.1fs)\n" t
@printf "      max|u-exact| = %0.3e   mean = %0.3e\n\n" dat_max dat_mean

println("──────── summary ────────")
@printf "method            max error    mean error\n"
@printf "forward PINN      %0.3e    %0.3e\n" fwd_max fwd_mean
@printf "inverse PINN      %0.3e    %0.3e   (ĉ recovered: %0.4f)\n" inv_max inv_mean ĉ
@printf "data-only MLP     %0.3e    %0.3e\n" dat_max dat_mean
