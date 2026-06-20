#!/usr/bin/env julia
# A clean schematic of one residual (skip-connection) block for Unit 4 §4.2:
#   hₗ₊₁ = hₗ + f_θ(hₗ).  The residual branch f_θ (two weight layers + an
#   activation) is added back to the untouched input via the skip connection.
# Output: figures/resnet_block.png

using Plots
gr()

rect(cx, cy, w, h) = Shape([cx-w/2, cx+w/2, cx+w/2, cx-w/2], [cy-h/2, cy-h/2, cy+h/2, cy+h/2])
const W, H = 3.0, 0.9
const CX = 4.0

plt = plot(; xlim = (-0.5, 9.5), ylim = (-0.3, 12.3), legend = false,
           framestyle = :none, size = (560, 720), background_color = :white)

# vertical flow boxes (bottom -> top)
boxes = [(1.0, "hₗ   (input)",            RGB(0.93,0.93,0.93)),
         (3.0, "weight layer  W₁",        RGB(0.80,0.90,0.98)),
         (4.4, "activation  σ",           RGB(0.86,0.95,0.86)),
         (5.8, "weight layer  W₂",        RGB(0.80,0.90,0.98)),
         (8.4, "⊕   add",                 RGB(0.99,0.92,0.80)),
         (9.8, "activation  σ",           RGB(0.86,0.95,0.86)),
         (11.6,"hₗ₊₁   (output)",         RGB(0.93,0.93,0.93))]
for (cy, label, c) in boxes
    plot!(plt, rect(CX, cy, W, H); c = c, lw = 1.2, lc = :gray30)
    annotate!(plt, CX, cy, text(label, 10, :black))
end

# upward flow arrows between consecutive boxes
ys = [b[1] for b in boxes]
for k in 1:length(ys)-1
    plot!(plt, [CX, CX], [ys[k]+H/2, ys[k+1]-H/2]; arrow = arrow(:closed), lw = 2, c = :gray30)
end

# skip connection: input -> right -> up -> into the ⊕ box
xr = 7.6
plot!(plt, [CX+W/2-0.4, xr], [1.0, 1.0]; lw = 2.2, c = :crimson)            # right off input
plot!(plt, [xr, xr], [1.0, 8.4]; lw = 2.2, c = :crimson)                     # up the side
plot!(plt, [xr, CX+W/2], [8.4, 8.4]; arrow = arrow(:closed), lw = 2.2, c = :crimson)  # into ⊕
annotate!(plt, xr+0.15, 4.7, text("identity\nskip", 9, :crimson, :left))

# bracket label for the residual branch f_θ
annotate!(plt, CX-W/2-0.55, 4.4, text("f", 12, :steelblue, :right))
annotate!(plt, CX-W/2-0.30, 4.4, text("θ", 8, :steelblue, :left))
plot!(plt, [CX-W/2-0.15, CX-W/2-0.15], [3.0-H/2, 5.8+H/2]; lw = 2, c = :steelblue)
annotate!(plt, CX, 12.05, text("Residual block:  hₗ₊₁ = hₗ + f_θ(hₗ)", 10, :black))

savefig(plt, joinpath(@__DIR__, "..", "figures", "resnet_block.png"))
println("wrote figures/resnet_block.png")
