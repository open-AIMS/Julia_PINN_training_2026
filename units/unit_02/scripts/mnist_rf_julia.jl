#!/usr/bin/env julia
# Random-forest baseline on MNIST in DecisionTree.jl.
# Cross-language pair with mnist_rf_sklearn.py.
#
# Requires MLDatasets.jl (not in the workshop Project.toml by default):
#   julia> using Pkg; Pkg.add("MLDatasets")

using MLDatasets, DecisionTree, Statistics

train_x, train_y = MLDatasets.MNIST(split = :train)[:]
test_x,  test_y  = MLDatasets.MNIST(split = :test)[:]

# (28, 28, N) → (N, 784), normalise to [0, 1]
flatten(x) = reshape(Float32.(x), 28 * 28, size(x, 3))' ./ 255f0
X_train, X_test = flatten(train_x), flatten(test_x)
y_train, y_test = Int.(train_y), Int.(test_y)

# Native DecisionTree.jl API:
#   build_forest(labels, features, n_subfeatures, n_trees, partial_sampling, max_depth, ...)
forest = build_forest(
    y_train, X_train,
    28,            # n_subfeatures (≈ √784)
    100,           # n_trees
    0.7,           # partial_sampling (fraction of rows per tree)
    -1,            # max_depth (-1 = unbounded)
    1, 0.0, 0;     # min_samples_leaf, min_samples_split, min_purity_increase
    rng = 0,
)
ŷ = apply_forest(forest, X_test)

acc = mean(ŷ .== y_test)
@info "MNIST test accuracy (DecisionTree.jl RF, 100 trees) = $(round(acc; digits = 4))"
