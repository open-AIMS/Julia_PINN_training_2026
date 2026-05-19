::: {.cell-output .cell-output-stdout}
```text
Precompiling Lux...
    313.6 ms  ✓ ConcreteStructs
   1891.0 ms  ✓ WeightInitializers
   1019.1 ms  ✓ WeightInitializers → ChainRulesCoreExt
   9618.3 ms  ✓ Lux
  4 dependencies successfully precompiled in 13 seconds. 94 already precompiled.
Precompiling ComponentArraysExt...
   1309.8 ms  ✓ Lux → ComponentArraysExt
  1 dependency successfully precompiled in 2 seconds. 104 already precompiled.
Precompiling ZygoteExt...
   1800.2 ms  ✓ Lux → ZygoteExt
  1 dependency successfully precompiled in 2 seconds. 137 already precompiled.
N_DATA = 121 (gauge samples every 6 min × 4 gauges = 484 data pts)
# water cells (interior, excl. boundary): 8515

Warming up loss…
  initial loss = 1.0206e+01   (data=1.531e-01 phys=4.004e+03 ic=2.732e-01 tic=5.783e-10 sm=5.737e+00)
Adam: 8000 steps @ lr=0.004
  step    1   loss = 1.9770e+01   (data=3.345e-01 phys=4.069e+03 ic=5.035e-01 tic=8.849e-10 sm=4.409e+00)   10.3s
  step  250   loss = 2.3564e-02   (data=2.633e-04 phys=1.914e+01 ic=6.191e-05 tic=2.387e-11 sm=8.837e-01)   14.6s
  step  500   loss = 1.6383e-02   (data=2.240e-04 phys=9.386e+00 ic=2.139e-05 tic=1.851e-11 sm=5.601e-01)   18.4s
  step  750   loss = 1.3212e-02   (data=2.080e-04 phys=4.934e+00 ic=2.279e-05 tic=2.112e-11 sm=3.771e-01)   22.3s
  step 1000   loss = 1.2080e-02   (data=1.986e-04 phys=3.561e+00 ic=2.304e-05 tic=2.010e-11 sm=4.019e-01)   26.0s
  step 1250   loss = 1.1929e-02   (data=1.949e-04 phys=3.548e+00 ic=2.963e-05 tic=1.308e-11 sm=4.398e-01)   29.8s
  step 1500   loss = 1.0711e-02   (data=1.878e-04 phys=2.121e+00 ic=1.517e-05 tic=9.701e-12 sm=2.905e-01)   33.6s
  step 1750   loss = 1.0561e-02   (data=1.847e-04 phys=2.189e+00 ic=1.392e-05 tic=1.161e-11 sm=2.560e-01)   37.4s
  step 2000   loss = 1.0367e-02   (data=1.819e-04 phys=2.045e+00 ic=1.479e-05 tic=7.248e-12 sm=2.729e-01)   41.2s
  step 2250   loss = 1.0555e-02   (data=1.876e-04 phys=1.844e+00 ic=3.027e-05 tic=6.972e-12 sm=2.411e-01)   45.0s
  step 2500   loss = 1.9602e-02   (data=3.670e-04 phys=2.019e+00 ic=4.947e-05 tic=6.459e-12 sm=1.771e-01)   48.8s
  step 2750   loss = 9.9837e-03   (data=1.793e-04 phys=1.599e+00 ic=1.750e-05 tic=4.877e-12 sm=2.310e-01)   52.6s
  step 3000   loss = 9.9608e-03   (data=1.804e-04 phys=1.484e+00 ic=2.681e-05 tic=4.979e-12 sm=1.837e-01)   56.4s
  step 3250   loss = 9.6855e-03   (data=1.748e-04 phys=1.610e+00 ic=8.122e-06 tic=4.848e-12 sm=1.524e-01)   60.4s
  step 3500   loss = 9.5328e-03   (data=1.744e-04 phys=1.377e+00 ic=8.189e-06 tic=3.571e-12 sm=1.334e-01)   64.1s
  step 3750   loss = 9.4795e-03   (data=1.738e-04 phys=1.291e+00 ic=1.031e-05 tic=5.184e-12 sm=1.546e-01)   67.8s
  step 4000   loss = 9.4133e-03   (data=1.734e-04 phys=1.129e+00 ic=8.890e-06 tic=5.126e-12 sm=2.003e-01)   71.6s
  step 4250   loss = 9.3133e-03   (data=1.730e-04 phys=1.029e+00 ic=9.240e-06 tic=4.588e-12 sm=1.603e-01)   75.3s
  step 4500   loss = 9.4269e-03   (data=1.729e-04 phys=1.358e+00 ic=8.201e-06 tic=5.004e-12 sm=1.057e-01)   79.1s
  step 4750   loss = 9.5551e-03   (data=1.778e-04 phys=1.089e+00 ic=7.743e-06 tic=4.341e-12 sm=1.298e-01)   82.7s
  step 5000   loss = 9.3065e-03   (data=1.726e-04 phys=1.133e+00 ic=8.388e-06 tic=4.572e-12 sm=1.147e-01)   86.3s
  step 5250   loss = 9.2591e-03   (data=1.723e-04 phys=1.072e+00 ic=6.224e-06 tic=3.663e-12 sm=1.204e-01)   89.8s
  step 5500   loss = 1.1669e-02   (data=2.212e-04 phys=9.174e-01 ic=1.730e-05 tic=4.887e-12 sm=1.434e-01)   93.5s
  step 5750   loss = 9.0946e-03   (data=1.708e-04 phys=9.499e-01 ic=8.907e-06 tic=5.357e-12 sm=8.025e-02)   97.1s
  step 6000   loss = 9.0020e-03   (data=1.706e-04 phys=7.840e-01 ic=5.671e-06 tic=3.865e-12 sm=8.620e-02)   100.6s
  step 6250   loss = 9.0907e-03   (data=1.705e-04 phys=9.931e-01 ic=7.585e-06 tic=3.958e-12 sm=6.839e-02)   104.2s
  step 6500   loss = 8.9695e-03   (data=1.704e-04 phys=7.232e-01 ic=6.246e-06 tic=2.927e-12 sm=9.515e-02)   107.8s
  step 6750   loss = 8.9302e-03   (data=1.701e-04 phys=7.137e-01 ic=5.737e-06 tic=3.566e-12 sm=7.304e-02)   111.3s
  step 7000   loss = 8.9407e-03   (data=1.700e-04 phys=7.136e-01 ic=9.110e-06 tic=4.692e-12 sm=8.076e-02)   114.9s
  step 7250   loss = 8.9432e-03   (data=1.700e-04 phys=7.596e-01 ic=6.048e-06 tic=3.524e-12 sm=6.561e-02)   119.0s
  step 7500   loss = 9.3424e-03   (data=1.781e-04 phys=6.958e-01 ic=1.155e-05 tic=4.703e-12 sm=8.540e-02)   123.2s
  step 7750   loss = 8.9873e-03   (data=1.707e-04 phys=7.689e-01 ic=1.039e-05 tic=5.190e-12 sm=5.587e-02)   127.4s
  step 8000   loss = 8.9468e-03   (data=1.701e-04 phys=7.578e-01 ic=7.932e-06 tic=4.986e-12 sm=6.135e-02)   131.6s
Adam phase done (131.6 s)

ψ recovery:
  L2 err = 0.1373 m   (relative: 83.6 %)
  truth peak = +0.450 m,  recovered peak = +0.074 m

Wrote data/psi_recovered.csv and data/gauges_pinn_pred.csv
```
:::
