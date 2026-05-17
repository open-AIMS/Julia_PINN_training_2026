# Small MLP on the iris dataset, in Lux.jl + Zygote + Adam.
# Demonstrates the explicit-parameter Lux style and a hand-rolled
# training loop — the same skeleton we use in later units when the
# loss starts to include PDE residuals.
#
# Run via ./build.sh execute 2 (writes output to ../output/iris_lux.md).

using Lux, Random, Zygote, Optimisers, Statistics
using MLJ: load_iris, unpack

# ── data ────────────────────────────────────────────────────────────────
iris = load_iris()
y, X = unpack(iris, ==(:target); rng = 123)
class_index = Dict(c => i - 1 for (i, c) in enumerate(unique(y)))
y_int       = [class_index[v] for v in y]                       # 0..2
# `unpack` returns X as a NamedTuple of feature columns; stack into a
# (4, 150) Float32 matrix with one sample per column.
X_mat = Float32.(permutedims(reduce(hcat, [collect(c) for c in values(X)])))

rng = Random.MersenneTwister(123)
perm = randperm(rng, length(y_int))
ntrain = 120
i_train, i_test = perm[1:ntrain], perm[ntrain+1:end]

# one-hot targets
onehot(k, K) = (e = zeros(Float32, K); e[k+1] = 1f0; e)
Y_train = reduce(hcat, [onehot(y_int[i], 3) for i in i_train])
Y_test  = reduce(hcat, [onehot(y_int[i], 3) for i in i_test])

X_train = X_mat[:, i_train]
X_test  = X_mat[:, i_test]

# ── model ───────────────────────────────────────────────────────────────
model = Lux.Chain(
    Lux.Dense(4 => 16, tanh),
    Lux.Dense(16 => 3),
)
ps, st = Lux.setup(rng, model)

# ── loss + accuracy ─────────────────────────────────────────────────────
function logits_then_softmax_xent(logits, Y)
    z = logits .- maximum(logits; dims = 1)           # stability
    logp = z .- log.(sum(exp.(z); dims = 1))
    -mean(sum(Y .* logp; dims = 1))
end

function loss_fn(ps, st, X, Y)
    logits, st = model(X, ps, st)
    logits_then_softmax_xent(logits, Y), st
end

function accuracy(ps, st, X, Y)
    logits, _ = model(X, ps, st)
    preds  = vec(map(argmax, eachcol(logits))) .- 1
    truth  = vec(map(argmax, eachcol(Y))) .- 1
    mean(preds .== truth)
end

# ── training loop with Adam (Optimisers.jl) ────────────────────────────
function train!(ps, st, X_train, Y_train, X_test, Y_test;
                epochs = 500, lr = 1.0f-2)
    opt_state = Optimisers.setup(Optimisers.Adam(lr), ps)
    for epoch in 1:epochs
        gs = first(Zygote.gradient(
            p -> first(loss_fn(p, st, X_train, Y_train)), ps,
        ))
        opt_state, ps = Optimisers.update(opt_state, ps, gs)
        if epoch % 100 == 0
            L, _ = loss_fn(ps, st, X_train, Y_train)
            a_tr = accuracy(ps, st, X_train, Y_train)
            a_te = accuracy(ps, st, X_test,  Y_test)
            @info "epoch $epoch  loss=$(round(L; digits=4))  train=$(round(a_tr; digits=3))  test=$(round(a_te; digits=3))"
        end
    end
    ps
end

ps = train!(ps, st, X_train, Y_train, X_test, Y_test)

# ── final report ────────────────────────────────────────────────────────
println("\nfinal train accuracy: ", round(accuracy(ps, st, X_train, Y_train); digits = 3))
println("final test  accuracy: ", round(accuracy(ps, st, X_test,  Y_test);  digits = 3))
