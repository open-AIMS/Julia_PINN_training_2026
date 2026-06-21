#!/usr/bin/env julia
# Softmax-regression baseline on MNIST in Lux.jl.
#
# A single 784→10 linear layer + softmax, fit by mini-batch SGD with
# Adam. The point is to land at ~92% test accuracy with no
# nonlinearity, then in the next section see what depth + width buy.
#
# Requires MLDatasets.jl (not in workshop Project.toml by default):
#   julia> using Pkg; Pkg.add("MLDatasets")

using MLDatasets, Lux, Random, Zygote, Optimisers, Statistics

# data ────────────────────────────────────────────────────────────────
train_x, train_y = MLDatasets.MNIST(split = :train)[:]
test_x,  test_y  = MLDatasets.MNIST(split = :test)[:]
flatten(x) = reshape(Float32.(x), 28 * 28, size(x, 3))
X_train, X_test = flatten(train_x), flatten(test_x)
y_train, y_test = Int.(train_y), Int.(test_y)

onehot(k, K = 10) = (e = zeros(Float32, K); e[k + 1] = 1f0; e)
Y_train = reduce(hcat, onehot.(y_train))

# model ───────────────────────────────────────────────────────────────
rng = Random.MersenneTwister(0)
model = Lux.Chain(Lux.Dense(784 => 10), Lux.softmax)
ps, st = Lux.setup(rng, model)

function loss_fn(ps, st, X, Y)
    ŷ, st = model(X, ps, st)
    -mean(sum(Y .* log.(ŷ .+ 1f-9); dims = 1)), st
end

# mini-batch training with Adam ───────────────────────────────────────
opt_state = Optimisers.setup(Optimisers.Adam(1f-2), ps)
batch     = 128
nb        = size(X_train, 2) ÷ batch
for epoch in 1:10
    perm = randperm(rng, size(X_train, 2))
    for b in 1:nb
        idx = perm[(b - 1) * batch + 1 : b * batch]
        gs  = first(Zygote.gradient(
                p -> first(loss_fn(p, st, X_train[:, idx], Y_train[:, idx])),
                ps,
              ))
        opt_state, ps = Optimisers.update(opt_state, ps, gs)
    end
    L, _ = loss_fn(ps, st, X_train, Y_train)
    @info "epoch $epoch  train-loss = $(round(L; digits = 4))"
end

ŷ_test, _ = model(X_test, ps, st)
preds = vec(map(argmax, eachcol(ŷ_test))) .- 1
acc = mean(preds .== y_test)
@info "MNIST test accuracy (Lux softmax) = $(round(acc; digits = 4))"
