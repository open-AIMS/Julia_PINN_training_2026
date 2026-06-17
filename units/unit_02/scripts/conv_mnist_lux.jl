#!/usr/bin/env julia
# ===========================================================================
# Unit 2.6 — a convolutional network in Lux.jl (LeNet-5 on MNIST).
#
# A real, multi-layer CONVOLUTIONAL network — the architecture family of
# *Mathematical Engineering of Deep Learning* Chapter 6 (Convolutional Neural
# Networks), and the headline model of §2.6. It is the classic LeNet-5
# (LeCun et al., 1998): two conv+pool blocks that learn local 2-D features,
# then three dense layers that classify them.
#
# Why convolutions? The softmax (§2.3) and MLP (§2.6) flatten each 28×28 image
# to a length-784 vector, throwing away the 2-D spatial structure. A conv layer
# instead slides small learnable filters over the image, sharing weights across
# locations — far fewer parameters, and translation-aware. On the same MNIST
# data that gave us random forest ~97%, softmax ~92% and the MLP ~98%, the CNN
# reaches ~99%, the best in the unit.
#
# Convolutions are also where the GPU really earns its keep: each layer is many
# small matmuls, so the per-epoch speed-up is much larger than the MLP's ~3×.
#
# Run on the GPU hub (the @pinn env has Lux + LuxCUDA + cuDNN + CUDA):
#   julia --project=@pinn units/unit_02/scripts/conv_mnist_lux.jl
# Nothing here runs during `quarto render` — the .qmd shows it `eval: false`.
# ===========================================================================

using Lux, LuxCUDA, CUDA, Optimisers, Zygote, Random, Printf, Statistics, Downloads

# --- MNIST loader (direct download + pure-Julia IDX parse) → image tensors ---
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
        b  = fetch_idx(name)
        n  = Int(b[5])<<24 | Int(b[6])<<16 | Int(b[7])<<8 | Int(b[8])
        nr = Int(b[9])<<24 | Int(b[10])<<16 | Int(b[11])<<8 | Int(b[12])
        nc = Int(b[13])<<24 | Int(b[14])<<16 | Int(b[15])<<8 | Int(b[16])
        # Lux/cuDNN want WHCN: (width, height, channels, batch)
        return reshape(Float32.(b[17:end]) ./ 255f0, nr, nc, 1, n)
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

# LeNet-5: conv→pool→conv→pool→dense×3. ~62k parameters.
make_cnn() = Chain(
    Conv((5, 5), 1 => 6, relu, pad = 2),   # 28×28×6
    MaxPool((2, 2)),                        # 14×14×6
    Conv((5, 5), 6 => 16, relu),            # 10×10×16
    MaxPool((2, 2)),                        #  5×5×16
    FlattenLayer(),                         # 400
    Dense(400 => 120, relu),
    Dense(120 => 84, relu),
    Dense(84 => 10),
)

function accuracy(model, ps, st, X, y, dev; bs = 1000)
    correct = 0
    for s in 1:bs:size(X, 4)
        idx = s:min(s + bs - 1, size(X, 4))
        ŷ, _ = model(dev(X[:, :, :, idx]), ps, st)
        pred = vec(map(i -> i[1] - 1, argmax(Array(ŷ); dims = 1)))
        correct += sum(pred .== y[idx])
    end
    return correct / length(y)
end

# one mini-batch SGD/Adam epoch; returns updated (ps, opt)
function train_epoch!(model, ps, st, opt, Xd, Yd, n, bs)
    order = randperm(n)
    for s in 1:bs:n
        idx = order[s:min(s + bs - 1, n)]
        xb = Xd[:, :, :, idx]; yb = Yd[:, idx]
        gs = Zygote.gradient(p -> logitce(first(model(xb, p, st)), yb), ps)[1]
        opt, ps = Optimisers.update(opt, ps, gs)
    end
    return ps, opt
end

# ---------------------------------------------------------------------------
println("="^64); println("Unit 2.6 — MNIST convolutional network (LeNet-5, Lux.jl)"); println("="^64)
have_gpu = CUDA.functional()
@printf("GPU available: %s%s\n", have_gpu, have_gpu ? "  ($(CUDA.name(CUDA.device())))" : "")

print("loading MNIST … "); Xtr, ytr, Xte, yte = load_mnist()
Ytr = onehot(ytr)
@printf("train=%d  test=%d  (28×28×1 image tensors)\n", size(Xtr, 4), size(Xte, 4))
@printf("LeNet-5: 2 conv blocks + 3 dense layers, %d parameters\n\n", Lux.parameterlength(make_cnn()))
const BS = 128

# --- per-epoch wall-clock: CPU vs GPU (convolutions favour the GPU heavily) --
println("Per-epoch wall-clock (batch=$BS, full 60k train set):")
@printf("%-8s %12s\n", "device", "sec / epoch"); println("-"^24)
results = Dict{String,Float64}()
for (tag, dev) in (have_gpu ? (("CPU", identity), ("GPU", gpu_device())) : (("CPU", identity),))
    rng = Xoshiro(0)
    model = make_cnn(); ps, st = Lux.setup(rng, model)
    ps = ps |> dev; st = st |> dev
    Xd = Xtr |> dev; Yd = Ytr |> dev
    ps, _ = train_epoch!(model, ps, st, Optimisers.setup(Adam(1f-3), ps), Xd, Yd, size(Xtr,4), BS) # warmup/compile
    dev === identity || CUDA.synchronize()
    t0 = time()
    ps, _ = train_epoch!(model, ps, st, Optimisers.setup(Adam(1f-3), ps), Xd, Yd, size(Xtr,4), BS)
    dev === identity || CUDA.synchronize()
    results[tag] = time() - t0
    @printf("%-8s %12.2f\n", tag, results[tag])
end
if haskey(results, "GPU")
    @printf("\nGPU speed-up: %.1fx per epoch (vs the MLP's ~3x — convolutions are far more compute-dense)\n",
            results["CPU"] / results["GPU"])
end

# --- a full training run on the GPU (or CPU fallback) to real accuracy -------
dev = have_gpu ? gpu_device() : identity
println("\nFull training run on $(have_gpu ? "GPU" : "CPU") — 10 epochs:")
rng = Xoshiro(1); model = make_cnn(); ps, st = Lux.setup(rng, model)
ps = ps |> dev; st = st |> dev
Xd = Xtr |> dev; Yd = Ytr |> dev
opt = Optimisers.setup(Adam(1f-3), ps)
hist = Float64[]
for epoch in 1:10
    global ps, opt = train_epoch!(model, ps, st, opt, Xd, Yd, size(Xtr,4), BS)
    acc = accuracy(model, ps, st, Xte, yte, dev)
    push!(hist, acc)
    @printf("  epoch %2d   test accuracy = %.4f\n", epoch, acc)
end
@printf("\nFinal MNIST test accuracy: %.4f  (forest ~0.97, softmax ~0.92, MLP ~0.98)\n", hist[end])
