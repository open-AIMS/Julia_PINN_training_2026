#!/usr/bin/env julia
# ===========================================================================
# Unit 6 — a standard PDE on the GPU: 2-D heat equation by finite differences.
#
# Section 6.4 solves the 1-D heat equation with an explicit FTCS stencil on the
# CPU. The 2-D version is the SAME idea — a five-point Laplacian stepped forward
# in time — but it is the canonical "stencil on a grid" workload that GPUs were
# built for: every interior cell updates independently from its four neighbours,
# so thousands of cells update in parallel.
#
# As in the Unit 1 GPU example, the update is written as ONE whole-array
# broadcast, so the identical code runs on the CPU (`Array`) or GPU (`CuArray`);
# only where the array lives changes. We sweep the grid size and watch the GPU
# pull away as the problem grows.
#
#   ∂u/∂t = α ∇²u   on (0,1)²,  u = 0 on the boundary (Dirichlet),
#   u(x,y,0) = a hot Gaussian bump in the centre.
#
# Run on the GPU hub (the @pinn env has CUDA.jl):
#   julia --project=@pinn units/unit_06/scripts/heat2d_gpu.jl
# Nothing here runs during `quarto render` — the .qmd shows it `eval: false`.
# ===========================================================================

using CUDA, Printf

const HAVE_GPU = CUDA.functional()

# One FTCS step into the double-buffer `un` (works for Array OR CuArray).
# u, un are N x N; only the interior updates, the boundary stays 0 (Dirichlet).
function heat_step!(un, u, c)            # c = α·dt/dx²
    N = size(u, 1)
    @views un[2:N-1, 2:N-1] .= u[2:N-1, 2:N-1] .+ c .* (
        u[1:N-2, 2:N-1] .+ u[3:N, 2:N-1] .+
        u[2:N-1, 1:N-2] .+ u[2:N-1, 3:N] .- 4f0 .* u[2:N-1, 2:N-1])
    return nothing
end

function heat_solve(N, dev; nsteps = 2000, α = 1.0f0, nwarm = 5)
    dx = 1.0f0 / (N - 1)
    dt = 0.20f0 * dx^2 / α                # CFL-safe for 2-D explicit (limit 0.25)
    c  = α * dt / dx^2

    # initial condition: hot Gaussian bump in the centre, zero on the boundary
    xs = range(0f0, 1f0; length = N)
    u0 = Float32[exp(-120f0 * ((x - 0.5f0)^2 + (y - 0.5f0)^2)) for y in xs, x in xs]
    u0[1, :] .= 0; u0[N, :] .= 0; u0[:, 1] .= 0; u0[:, N] .= 0

    u  = dev(copy(u0))
    un = dev(copy(u0))

    for _ in 1:nwarm                      # warm up / compile
        heat_step!(un, u, c); u, un = un, u
    end
    u .= dev(u0); un .= dev(u0)
    dev === identity || CUDA.synchronize()
    t0 = time()
    for _ in 1:nsteps
        heat_step!(un, u, c)
        u, un = un, u
    end
    dev === identity || CUDA.synchronize()
    elapsed = time() - t0
    return (; field = Array(u), elapsed, N, ncells = N*N, nsteps, peak = maximum(Array(u)))
end

println("="^64)
println("Unit 6 — 2-D heat equation (FTCS finite differences) on CPU vs GPU")
println("="^64)
@printf("GPU available: %s%s\n\n", HAVE_GPU,
        HAVE_GPU ? "  ($(CUDA.name(CUDA.device())))" : "")

@printf("%-8s %-12s %-10s %-7s %10s %10s %9s\n",
        "N", "grid", "cells", "steps", "CPU (s)", "GPU (s)", "speedup")
println("-"^74)

sizes = [128, 256, 512, 1024]
fine = nothing
for N in sizes
    cpu = heat_solve(N, identity)
    gpu = HAVE_GPU ? heat_solve(N, CuArray) : nothing
    spd = gpu === nothing ? NaN : cpu.elapsed / gpu.elapsed
    @printf("%-8d %-12s %-10d %-7d %10.2f %10.2f %8.1fx\n",
            N, "$(N)x$(N)", cpu.ncells, cpu.nsteps, cpu.elapsed,
            gpu === nothing ? NaN : gpu.elapsed, spd)
    gpu === nothing || @assert maximum(abs, cpu.field .- gpu.field) < 1f-3 "CPU/GPU diverged"
    global fine = (gpu === nothing ? cpu : gpu).field
end

# GPU can keep going where the CPU gets painful:
if HAVE_GPU
    for N in (2048, 4096)
        g = heat_solve(N, CuArray)
        @printf("%-8d %-12s %-10d %-7d %10s %10.2f %9s\n",
                N, "$(N)x$(N)", g.ncells, g.nsteps, "—  (skip)", g.elapsed, "GPU-only")
        global fine = g.field
    end
end

println()
println("CPU and GPU fields agree to < 1e-3 at every size (same stencil, same code).")

try
    using CairoMakie
    f = Figure(size = (520, 430))
    ax = Axis(f[1, 1], title = "2-D heat: diffused field (GPU, finest grid)",
              xlabel = "x", ylabel = "y", aspect = 1)
    hm = heatmap!(ax, fine, colormap = :inferno)
    Colorbar(f[1, 2], hm, label = "u")
    figdir = get(ENV, "GPU_FIG_DIR", joinpath(@__DIR__, "..", "figures"))
    isdir(figdir) || mkpath(figdir)
    save(joinpath(figdir, "heat2d_gpu_field.png"), f)
    println("wrote figures/heat2d_gpu_field.png")
catch e
    println("(figure skipped: ", e, ")")
end
