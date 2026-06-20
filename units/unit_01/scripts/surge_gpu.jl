#!/usr/bin/env julia
# ===========================================================================
# Unit 1 — the Moreton Bay surge solver, on the GPU.
#
# This is the SAME linearised shallow-water model as scripts/generate_surge_data.jl
# (Arakawa C-grid, forward-backward stepping, land mask, eastern sponge, a
# Dirichlet river-mouth source), but rewritten in a VECTORISED, backend-agnostic
# style: every update is a whole-array broadcast, no scalar loops. The exact same
# code therefore runs on the CPU (`Array`) or the GPU (`CuArray`) — the only
# difference is where the arrays live.
#
# Why bother? The reference solver runs a 100x190 grid in a few seconds on a CPU.
# Real bathymetric studies want metre-scale resolution over the whole bay, which
# is 10-100x more cells and 10-100x more time steps (CFL ties dt to dx). That is
# exactly the regime where a GPU's thousands of cores win: this script refines the
# bay by an integer factor and shows the CPU vs GPU wall-clock crossover.
#
# Run on the GPU hub (the @pinn env has CUDA.jl):
#   julia --project=@pinn units/unit_01/scripts/surge_gpu.jl
#
# It prints a benchmark table and writes figures/surge_gpu_field.png.
# Nothing here executes during `quarto render` — the .qmd shows it `eval: false`
# and includes the captured output.
# ===========================================================================

using Printf, Statistics, JSON3

# CUDA is optional: on the GPU hub it drives the real benchmark; on a CPU-only
# machine (e.g. a laptop regenerating just the figure) we skip it gracefully and
# the identical vectorised solver runs on `Array`. The field is the same either way.
const HAVE_CUDA = try
    @eval using CUDA
    true
catch
    @info "CUDA.jl not available — running CPU-only (benchmark GPU column blank; figure still rendered)"
    false
end
const HAVE_GPU = HAVE_CUDA && CUDA.functional()

# --- Bay geometry ----------------------------------------------------------
# Prefer the real Unit 1 bathymetry/mask if we can find them (on the hub the
# course repo is at /home/efs/_shared/course-materials); otherwise synthesise an
# idealised bay with the same character: a sloping basin, a shallower western
# shelf near the river mouth, a barrier-island gap to the east, land to the west.

function load_or_make_bay()
    candidates = String[]
    haskey(ENV, "GPU_DATA_DIR") && push!(candidates, ENV["GPU_DATA_DIR"])
    push!(candidates, joinpath(@__DIR__, "..", "data"))
    push!(candidates, "/home/efs/_shared/course-materials/units/unit_01/data")
    for d in candidates
        bf = joinpath(d, "bay_bathymetry.csv")
        mf = joinpath(d, "bay_mask.csv")
        if isfile(bf) && isfile(mf)
            H = _readcsv_float(bf)
            M = round.(Int, _readcsv_float(mf))
            H[isnan.(H)] .= 5.0
            src0 = _read_source(joinpath(d, "river_source.csv"), M)
            return (Float32.(H), M, src0, "real Moreton Bay bathymetry ($(size(H,1))x$(size(H,2)))")
        end
    end
    # --- idealised fallback (NY x NX = 100 x 190, same aspect as the real bay)
    NY, NX = 100, 190
    H = fill(5.0f0, NY, NX)
    M = ones(Int, NY, NX)
    @inbounds for j in 1:NY, i in 1:NX
        # depth deepens to the east; shallow shelf in the west near the source
        depth = 3.0f0 + 42.0f0 * (i / NX)^1.3f0
        H[j, i] = depth
        # western land wedge (mainland coast) + a barrier island band near i≈0.8NX
        west_land = i < (6 + 10 * sin(3.0 * j / NY))
        barrier   = (abs(i - round(Int, 0.82NX)) ≤ 1) && !(38 ≤ j ≤ 52)  # gap = tidal inlet
        if west_land || barrier
            M[j, i] = 0
            H[j, i] = 1.0f0
        end
    end
    jr0 = clamp(round(Int, 0.33NY), 2, NY-1)
    return (H, M, (jr0, _first_water_col(M, jr0)), "idealised bay (100x190)")
end

function _readcsv_float(path)
    rows = Vector{Vector{Float64}}()
    for ln in eachline(path)
        isempty(strip(ln)) && continue
        push!(rows, [(t == "NaN" || t == "nan" || isempty(t)) ? NaN : parse(Float64, t)
                     for t in split(ln, ',')])
    end
    return reduce(vcat, [permutedims(r) for r in rows])
end

_first_water_col(M, row) = (for i in 1:size(M, 2); M[row, i] == 1 && return max(i, 3); end; 3)

# The REAL Brisbane River-mouth cell (row = iy = north, col = ix = east) read from
# river_source.csv, so the GPU surge starts where the actual source is — not at a
# synthetic 1/3-up-the-coast guess. Falls back to that heuristic if the file is absent.
function _read_source(path, M)
    if isfile(path)
        lines = readlines(path)
        if length(lines) >= 2
            cols = split(strip(lines[2]), ',')
            if length(cols) >= 6
                ix = tryparse(Int, cols[5]); iy = tryparse(Int, cols[6])
                ix !== nothing && iy !== nothing && return (iy, ix)
            end
        end
    end
    jr = clamp(round(Int, 0.33size(M, 1)), 2, size(M, 1) - 1)
    return (jr, _first_water_col(M, jr))
end

# Nearest-neighbour integer refinement of a field (each cell -> r x r block).
refine_field(A, r) = r == 1 ? A : repeat(A, inner = (r, r))

# --- Build all the static arrays a solve needs, on a given backend ----------
# `dev` is `identity` for CPU or `CuArray` for GPU.
function build_state(Hc, Mc, src0, refine, dev)
    H  = Float32.(refine_field(Hc, refine))
    M  = refine_field(Mc, refine)
    NY, NX = size(H)
    maskf = Float32.(M)

    # face depths and open-face masks (water-water faces only)
    Hu = 0.5f0 .* (H[:, 1:NX-1] .+ H[:, 2:NX])
    Hv = 0.5f0 .* (H[1:NY-1, :] .+ H[2:NY, :])
    uopen = Float32.((M[:, 1:NX-1] .== 1) .& (M[:, 2:NX] .== 1))
    vopen = Float32.((M[1:NY-1, :] .== 1) .& (M[2:NY, :] .== 1))

    # absorbing sponge on every open rim — northern entrance, eastern ocean
    # strip, southern Broadwater outlet (west is solid mainland → no sponge).
    sponge = zeros(Float32, NY, NX)
    SPONGE_W = 5 * refine
    @inbounds for j in 1:NY, i in 1:NX
        de = i - (NX - SPONGE_W)
        dn = j - (NY - SPONGE_W)
        ds = (SPONGE_W + 1) - j
        s = 0.0f0
        de > 0 && (s = max(s, (1.0f0 / 60) * (de / SPONGE_W)^2))
        dn > 0 && (s = max(s, (1.0f0 / 60) * (dn / SPONGE_W)^2))
        ds > 0 && (s = max(s, (1.0f0 / 60) * (ds / SPONGE_W)^2))
        sponge[j, i] = s
    end
    spu = 0.5f0 .* (sponge[:, 1:NX-1] .+ sponge[:, 2:NX])
    spv = 0.5f0 .* (sponge[1:NY-1, :] .+ sponge[2:NY, :])

    # river-mouth source cell — the REAL mouth (src0 = base-grid row/col), mapped
    # into the refined grid and nudged onto water if the centre lands on land.
    jr0, ir0 = src0
    half = cld(refine, 2)
    jr = clamp((jr0 - 1) * refine + half, 2, NY - 1)
    ir = clamp((ir0 - 1) * refine + half, 2, NX - 1)
    if M[jr, ir] == 0
        best = (jr, ir); bestd = typemax(Int)
        @inbounds for jj in 1:NY, ii in 1:NX
            M[jj, ii] == 1 || continue
            d = (jj - jr)^2 + (ii - ir)^2
            d < bestd && (bestd = d; best = (jj, ii))
        end
        jr, ir = best
    end
    src = zeros(Float32, NY, NX); src[jr, ir] = 1.0f0

    return (; H, maskf, Hu, Hv, uopen, vopen, sponge, spu, spv,
            src = dev(src), NY, NX,
            maskf_d = dev(maskf), Hu_d = dev(Hu), Hv_d = dev(Hv),
            uopen_d = dev(uopen), vopen_d = dev(vopen),
            sponge_d = dev(sponge), spu_d = dev(spu), spv_d = dev(spv))
end

# Surge profile imposed at the source (two flood pulses), in metres.
psi(t) = 0.45f0 * exp(-((t - 2.0f0*3600) / (0.55f0*3600))^2) +
         0.18f0 * exp(-((t - 4.3f0*3600) / (0.55f0*3600))^2)

# --- One fully-vectorised time step (works for Array OR CuArray) ------------
const G_GRAV = 9.81f0
const B_DRAG = 5.0f-5

function swe_solve(Hc, Mc, src0, refine, dev; dx0 = 500.0f0, t_end = 3*3600.0f0,
                   nwarm = 5, capture::Int = 0)
    s = build_state(Hc, Mc, src0, refine, dev)
    NY, NX = s.NY, s.NX
    dx = dx0 / refine; dy = dx
    cmax = sqrt(G_GRAV * maximum(s.H))
    dt = 0.45f0 / (cmax * sqrt(1/dx^2 + 1/dy^2))        # CFL-safe
    nt = Int(floor(t_end / dt))
    frames = Matrix{Float32}[]; ftimes = Float64[]      # for the movie (capture>0)
    cap_every = capture > 0 ? max(1, nt ÷ capture) : typemax(Int)

    η = dev(zeros(Float32, NY, NX))
    u = dev(zeros(Float32, NY, NX-1))
    v = dev(zeros(Float32, NY-1, NX))
    divx = similar(η); divy = similar(η)
    fill!(divx, 0); fill!(divy, 0)

    function step!(tt)
        Fx = s.Hu_d .* u
        Fy = s.Hv_d .* v
        @views divx[:, 2:NX-1] .= (Fx[:, 2:NX-1] .- Fx[:, 1:NX-2]) ./ dx
        @views divy[2:NY-1, :] .= (Fy[2:NY-1, :] .- Fy[1:NY-2, :]) ./ dy
        η .-= dt .* (divx .+ divy) .* s.maskf_d          # continuity (water only)
        η .*= (1f0 .- dt .* s.sponge_d)                  # sponge on η
        η .= η .* (1f0 .- s.src) .+ (psi(tt) .* s.src)   # river-mouth Dirichlet
        dηdx = (η[:, 2:NX] .- η[:, 1:NX-1]) ./ dx
        dηdy = (η[2:NY, :] .- η[1:NY-1, :]) ./ dy
        u .= (u .+ dt .* (.-G_GRAV .* dηdx .- B_DRAG .* u)) .* s.uopen_d
        v .= (v .+ dt .* (.-G_GRAV .* dηdy .- B_DRAG .* v)) .* s.vopen_d
        u .*= (1f0 .- dt .* s.spu_d)                     # sponge on velocities
        v .*= (1f0 .- dt .* s.spv_d)
        return nothing
    end

    for n in 1:nwarm; step!(n*dt); end                   # warm up / compile
    fill!(η, 0); fill!(u, 0); fill!(v, 0)
    dev === identity || CUDA.synchronize()
    t0 = time()
    for n in 1:nt
        step!(n*dt)
        if capture > 0 && n % cap_every == 0
            push!(frames, Array(η)); push!(ftimes, n*dt)
        end
    end
    dev === identity || CUDA.synchronize()
    elapsed = time() - t0

    env = maximum(abs, Array(η))
    return (; field = Array(η), elapsed, nt, dt, NY, NX, ncells = NY*NX, env,
            frames, ftimes)
end

# ---------------------------------------------------------------------------
println("="^64)
println("Unit 1 — Moreton Bay shallow-water surge on CPU vs GPU")
println("="^64)
Hc, Mc, src0, src_desc = load_or_make_bay()
@printf("geometry: %s\n", src_desc)
@printf("GPU available: %s%s\n", HAVE_GPU,
        HAVE_GPU ? "  ($(CUDA.name(CUDA.device())), $(round(CUDA.totalmem(CUDA.device())/2^30; digits=1)) GiB)" : "")
println()

@printf("%-8s %-12s %-9s %-7s %10s %10s %9s\n",
        "refine", "grid", "cells", "steps", "CPU (s)", "GPU (s)", "speedup")
println("-"^72)

refines = [1, 2, 3, 4]
fine_field = nothing
for r in refines
    cpu = swe_solve(Hc, Mc, src0, r, identity)
    gpu = HAVE_GPU ? swe_solve(Hc, Mc, src0, r, CuArray) : nothing
    spd = gpu === nothing ? NaN : cpu.elapsed / gpu.elapsed
    @printf("%-8d %-12s %-9d %-7d %10.2f %10.2f %8.1fx\n",
            r, "$(cpu.NY)x$(cpu.NX)", cpu.ncells, cpu.nt,
            cpu.elapsed, gpu === nothing ? NaN : gpu.elapsed, spd)
    if gpu !== nothing
        @assert maximum(abs, cpu.field .- gpu.field) < 1f-2 "CPU/GPU fields diverged"
    end
    global fine_field = (gpu === nothing ? cpu : gpu).field
end

println()
println("CPU and GPU fields agree to < 1e-2 m at every resolution (same code, same physics).")

# --- Figure + movie: the surge on the finest grid --------------------------
# Rotated landscape, North ← left, land greyed, km axes — same look as the rest
# of Unit 1's bay maps (units/unit_01/scripts/_mapfig.jl). We re-run the finest
# grid once more (NOT timed) capturing frames, then render a static field PNG and
# a sequence of movie frames that unit_01.qmd plays back with a JS widget.
try
    include(joinpath(@__DIR__, "_mapfig.jl"))
    rfac      = last(refines)
    fine_mask = refine_field(Mc, rfac)
    dxkm      = 0.5 / rfac                       # 500 m base grid, refined ×rfac

    run  = swe_solve(Hc, Mc, src0, rfac, HAVE_GPU ? CuArray : identity;
                     t_end = 5*3600.0f0, capture = 48)   # movie runs to t = 5 h
    NYf, NXf = size(run.field)
    vlim = max(0.05f0, 0.6f0 * maximum(maximum(abs, F) for F in run.frames))

    figdir = get(ENV, "GPU_FIG_DIR", joinpath(@__DIR__, "..", "figures"))
    isdir(figdir) || mkpath(figdir)

    # static field (final state)
    p = bay_map(run.field, fine_mask, dxkm;
        clims = (-vlim, vlim), cmap = :balance, clabel = "η  (m)",
        title = "Surge η at t = 5 h — refined $(NYf)×$(NXf) Moreton Bay grid")
    savefig(p, joinpath(figdir, "surge_gpu_field.png"))
    println("wrote figures/surge_gpu_field.png")

    # movie frames + metadata (played back by the widget in unit_01.qmd)
    moviedir = joinpath(figdir, "surge_gpu_frames")
    isdir(moviedir) || mkpath(moviedir)
    for f in readdir(moviedir; join = true); endswith(f, ".png") && rm(f); end
    recs = Dict{String, Any}[]
    for (k, (Fk, tk)) in enumerate(zip(run.frames, run.ftimes))
        pp = bay_map(Fk, fine_mask, dxkm;
            clims = (-vlim, vlim), cmap = :balance, clabel = "η  (m)",
            title = @sprintf("GPU surge (refined %d×%d) — t = %.2f h", NYf, NXf, tk/3600))
        fn = @sprintf("frame_%03d.png", k - 1)
        savefig(pp, joinpath(moviedir, fn))
        push!(recs, Dict("idx" => k - 1, "file" => fn, "t_hr" => tk/3600))
    end
    open(joinpath(moviedir, "frames_meta.json"), "w") do io
        JSON3.pretty(io, Dict("nframes" => length(recs),
                              "grid" => "$(NYf)x$(NXf)", "frames" => recs))
    end
    println("wrote ", length(recs), " GPU movie frames into figures/surge_gpu_frames/")
catch e
    println("(figure/movie skipped: ", e, ")")
end
