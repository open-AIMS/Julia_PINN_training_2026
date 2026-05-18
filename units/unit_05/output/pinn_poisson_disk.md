::: {.cell-output .cell-output-stdout}
```text
┌ Warning: `ForwardDiff.derivative(f, x)` within Zygote cannot track gradients with respect to `f`,
│ and `f` appears to be a closure, or a struct with fields (according to `issingletontype(typeof(f))`).
│ typeof(f) = var"#1#5"{Chain{@NamedTuple{layer_1::Dense{typeof(tanh), Int64, Int64, Nothing, Nothing, Static.True}, layer_2::Dense{typeof(tanh), Int64, Int64, Nothing, Nothing, Static.True}, layer_3::Dense{typeof(identity), Int64, Int64, Nothing, Nothing, Static.True}}, Nothing}, Float32, @NamedTuple{layer_1::@NamedTuple{weight::Matrix{Float32}, bias::Vector{Float32}}, layer_2::@NamedTuple{weight::Matrix{Float32}, bias::Vector{Float32}}, layer_3::@NamedTuple{weight::Matrix{Float32}, bias::Vector{Float32}}}, @NamedTuple{layer_1::@NamedTuple{}, layer_2::@NamedTuple{}, layer_3::@NamedTuple{}}}
└ @ Zygote ~/.julia/packages/Zygote/55SqB/src/lib/forward.jl:158
┌ Warning: `ForwardDiff.derivative(f, x)` within Zygote cannot track gradients with respect to `f`,
│ and `f` appears to be a closure, or a struct with fields (according to `issingletontype(typeof(f))`).
│ typeof(f) = var"#3#7"{Chain{@NamedTuple{layer_1::Dense{typeof(tanh), Int64, Int64, Nothing, Nothing, Static.True}, layer_2::Dense{typeof(tanh), Int64, Int64, Nothing, Nothing, Static.True}, layer_3::Dense{typeof(identity), Int64, Int64, Nothing, Nothing, Static.True}}, Nothing}, Float32, @NamedTuple{layer_1::@NamedTuple{weight::Matrix{Float32}, bias::Vector{Float32}}, layer_2::@NamedTuple{weight::Matrix{Float32}, bias::Vector{Float32}}, layer_3::@NamedTuple{weight::Matrix{Float32}, bias::Vector{Float32}}}, @NamedTuple{layer_1::@NamedTuple{}, layer_2::@NamedTuple{}, layer_3::@NamedTuple{}}}
└ @ Zygote ~/.julia/packages/Zygote/55SqB/src/lib/forward.jl:158
Poisson on the unit disk — three methods
  -Δu = 1, u|∂Ω = 0, exact u(r) = (1 - r²)/4
  network: 2 → 32 → 32 → 1 tanh, Adam @ 5e-3

[1/3] forward PINN  (17.0s)
      max|u-exact| = 5.672e-01   mean = 1.869e-01

[2/3] inverse PINN  (6.2s)
      max|u-exact| = 1.427e-01   mean = 2.942e-02   ĉ = 0.3193 (true 1.000)

[3/3] data-only MLP (3.6s)
      max|u-exact| = 1.645e-01   mean = 2.809e-02

──────── summary ────────
method            max error    mean error
forward PINN      5.672e-01    1.869e-01
inverse PINN      1.427e-01    2.942e-02   (ĉ recovered: 0.3193)
data-only MLP     1.645e-01    2.809e-02
```
:::
