#!/usr/bin/env julia
# Display a handful of MNIST digits (28×28 grayscale) before we train on them.
# Both MLDatasets and Plots are in the workshop `@pinn` environment, so this runs
# with no extra installation. In a notebook, end on `plt` and it renders inline;
# this script saves it to a PNG instead.

using MLDatasets, Plots

# train_x is 28×28×60000 with pixel values already in [0, 1]; train_y are the labels 0–9.
train_x, train_y = MLDatasets.MNIST(split = :train)[:]

# First 40 digits in a 5×8 grid, each panel titled with its label.
panels = map(1:40) do i
    heatmap(transpose(train_x[:, :, i]);             # transpose + yflip ⇒ upright digit
            yflip = true, color = :grays, aspect_ratio = 1,
            axis = false, ticks = false, colorbar = false, legend = false,
            title = string(train_y[i]), titlefontsize = 7)
end

plt = plot(panels...; layout = (5, 8), size = (760, 500),
           plot_title = "MNIST — first 40 training digits", plot_titlefontsize = 11)

savefig(plt, joinpath(@__DIR__, "..", "figures", "mnist_digits.png"))
