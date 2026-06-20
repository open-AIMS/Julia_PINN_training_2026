# Activation functions used through the workshop. The smooth, bounded ones
# (sigmoid, tanh) are the default for PINNs because their higher derivatives are
# nonzero everywhere — see unit_02.qmd §2.4. Generates figures/activations.png.

using Plots

σ(z)     = 1 / (1 + exp(-z))     # sigmoid
swish(z) = z * σ(z)              # Swish (β = 1)
relu(z)  = max(0, z)

zs = range(-5, 5; length = 400)
plot(zs, σ.(zs);     lw = 2, label = "sigmoid  σ(z)=1/(1+e⁻ᶻ)",
     xlabel = "z", ylabel = "activation", title = "Activation functions",
     legend = :topleft, framestyle = :zerolines)
plot!(zs, tanh.(zs); lw = 2, label = "tanh(z)")
plot!(zs, relu.(zs); lw = 2, label = "ReLU  max(0,z)")
plot!(zs, swish.(zs); lw = 2, ls = :dash, label = "Swish  z·σ(z)")
savefig(joinpath(@__DIR__, "..", "figures", "activations.png"))
