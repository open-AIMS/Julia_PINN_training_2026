#!/usr/bin/env julia
# ===========================================================================
# Unit 2.7 — a real deep-learning training run on the GPU (MNIST, Lux.jl).
#
# Section 2.7 trains an MLP on MNIST in scikit-learn and PyTorch. The PyTorch
# version already moves tensors to `cuda`; this
# is the *Julia* side of that story: the identical Lux model trained on the CPU
# and on the GPU, so you can see (a) that the move is a one-liner — push the
# params and the data to the device — and (b) the speed-up on an image-scale
# dataset where the GPU's batched matmuls actually pay off.
#
# MNIST is fetched directly from a public mirror and parsed in a few lines of
# pure Julia (no MLDatasets / PyCall dependency). Model: 784-256-128-10 MLP.
#
# Run on the GPU hub (the @pinn env has Lux + LuxCUDA + cuDNN + CUDA):
#   julia --project=@pinn units/unit_02/scripts/mnist_gpu_lux.jl
# Nothing here runs during `quarto render` — the .qmd shows it `eval: false`.
# ===========================================================================

using Lux, LuxCUDA, CUDA, Optimisers, Zygote, Random, Printf, Statistics, Downloads

# --- MNIST loader (direct download + pure-Julia IDX parse) -----------------
const MIRRORS = ["https://storage.googleapis.com/cvdf-datasets/mnist/",
                 "https://ossci-datasets.s3.amazonaws.com/mnist/"]

function fetch_idx(name)
    cache = joinpath(get(ENV, "MNIST_DIR", "/tmp/mnist"), name)
    raw   = replace(cache, ".gz" => "")
    isdir(dirname(cache)) || mkpath(dirname(cache))
    if !isfile(raw)
        ok = false
        for m in MIRRORS
            try
                Downloads.download(m * name, cache); ok = true; break
            catch; end
        end
        ok || error("could not download $name from any mirror")
        run(`gunzip -f $cache`)
    end
    return read(raw)
end

function load_mnist()
    function images(name)
        b = fetch_idx(name)
        n  = Int(b[5])<<24 | Int(b[6])<<16 | Int(b[7])<<8 | Int(b[8])
        nr = Int(b[9])<<24 | Int(b[10])<<16 | Int(b[11])<<8 | Int(b[12])
        nc = Int(b[13])<<24 | Int(b[14])<<16 | Int(b[15])<<8 | Int(b[16])
        px = reshape(b[17:end], nr*nc, n)
        return Float32.(px) ./ 255f0          # (784, n), normalised
    end
    function labels(name)
        b = fetch_idx(name)
        n = Int(b[5])<<24 | Int(b[6])<<16 | Int(b[7])<<8 | Int(b[8])
        return Int.(b[9:8+n])                  # 0..9
    end
    Xtr = images("train-images-idx3-ubyte.gz"); ytr = labels("train-labels-idx1-ubyte.gz")
    Xte = images("t10k-images-idx3-ubyte.gz");  yte = labels("t10k-labels-idx1-ubyte.gz")
    return Xtr, ytr, Xte, yte
end

onehot(y) = (Y = zeros(Float32, 10, length(y)); for (j, c) in enumerate(y); Y[c+1, j] = 1f0; end; Y)

# numerically-stable softmax cross-entropy (works on Array or CuArray)
function logitce(logits, Y)
    m  = maximum(logits; dims = 1)
    ls = logits .- m .- log.(sum(exp.(logits .- m); dims = 1))
    return -sum(Y .* ls) / size(Y, 2)
end

function accuracy(model, ps, st, X, y, dev)
    ŷ, _ = model(dev(X), ps, st)
    pred = vec(map(i -> i[1] - 1, argmax(Array(ŷ); dims = 1)))
    return mean(pred .== y)
end

# --- one training epoch (mini-batch SGD/Adam); returns seconds --------------
function train_epoch!(model, ps, st, opt, Xd, Yd, n, bs)
    order = randperm(n)
    for s in 1:bs:n
        idx = order[s:min(s+bs-1, n)]
        xb = Xd[:, idx]; yb = Yd[:, idx]
        gs = Zygote.gradient(p -> logitce(first(model(xb, p, st)), yb), ps)[1]
        opt, ps = Optimisers.update(opt, ps, gs)
    end
    return ps, opt
end

# ---------------------------------------------------------------------------
println("="^64); println("Unit 2.7 — MNIST MLP training on CPU vs GPU (Lux.jl)"); println("="^64)
have_gpu = CUDA.functional()
@printf("GPU available: %s%s\n", have_gpu, have_gpu ? "  ($(CUDA.name(CUDA.device())))" : "")

print("loading MNIST … "); Xtr, ytr, Xte, yte = load_mnist()
Ytr = onehot(ytr)
@printf("train=%d  test=%d  (28x28 → 784)\n\n", size(Xtr, 2), size(Xte, 2))

make_model() = Chain(Dense(784 => 256, relu), Dense(256 => 128, relu), Dense(128 => 10))
const BS = 128

# --- per-epoch wall-clock: CPU vs GPU (same model, same data) --------------
println("Per-epoch wall-clock (batch=$BS, full 60k train set):")
@printf("%-8s %12s\n", "device", "sec / epoch"); println("-"^24)
results = Dict{String,Float64}()
for (tag, dev) in (have_gpu ? (("CPU", identity), ("GPU", gpu_device())) : (("CPU", identity),))
    rng = Xoshiro(0)
    model = make_model(); ps, st = Lux.setup(rng, model)
    ps = ps |> dev; st = st |> dev
    Xd = Xtr |> dev; Yd = Ytr |> dev
    ps, _ = train_epoch!(model, ps, st, Optimisers.setup(Adam(1f-3), ps), Xd, Yd, size(Xtr,2), BS) # warmup/compile
    dev === identity || CUDA.synchronize()
    t0 = time()
    ps, _ = train_epoch!(model, ps, st, Optimisers.setup(Adam(1f-3), ps), Xd, Yd, size(Xtr,2), BS)
    dev === identity || CUDA.synchronize()
    results[tag] = time() - t0
    @printf("%-8s %12.2f\n", tag, results[tag])
end
if haskey(results, "GPU")
    @printf("\nGPU speed-up: %.1fx per epoch\n", results["CPU"] / results["GPU"])
end

# --- a full training run on the GPU (or CPU fallback) to real accuracy ------
dev = have_gpu ? gpu_device() : identity
println("\nFull training run on $(have_gpu ? "GPU" : "CPU") — 10 epochs:")
rng = Xoshiro(1); model = make_model(); ps, st = Lux.setup(rng, model)
ps = ps |> dev; st = st |> dev
Xd = Xtr |> dev; Yd = Ytr |> dev
opt = Optimisers.setup(Adam(1f-3), ps)
hist = Float64[]
for epoch in 1:10
    global ps, opt = train_epoch!(model, ps, st, opt, Xd, Yd, size(Xtr,2), BS)
    acc = accuracy(model, ps, st, Xte, yte, dev)
    push!(hist, acc)
    @printf("  epoch %2d   test accuracy = %.4f\n", epoch, acc)
end
@printf("\nFinal MNIST test accuracy: %.4f\n", hist[end])

try
    using CairoMakie
    f = Figure(size = (560, 380))
    ax = Axis(f[1, 1], title = "MNIST MLP on the GPU — test accuracy",
              xlabel = "epoch", ylabel = "test accuracy")
    lines!(ax, 1:length(hist), hist, linewidth = 3)
    scatter!(ax, 1:length(hist), hist, markersize = 9)
    figdir = get(ENV, "GPU_FIG_DIR", joinpath(@__DIR__, "..", "figures"))
    isdir(figdir) || mkpath(figdir)
    save(joinpath(figdir, "mnist_gpu_accuracy.png"), f)
    println("wrote figures/mnist_gpu_accuracy.png")
catch e
    println("(figure skipped: ", e, ")")
end
