::: {.cell-output .cell-output-stdout}
```text
N_DATA = 121 (gauge samples every 6 min × 4 gauges = 484 data pts)
# water cells (interior, excl. boundary): 8515

Warming up loss…
  initial loss = 1.0398e+01   (data=1.586e-01 phys=3.851e+03 ic=2.689e-01 tic=6.124e-10 sm=4.546e+00)
Adam: 8000 steps @ lr=0.004
  step    1   loss = 1.9134e+01   (data=3.253e-01 phys=3.772e+03 ic=4.912e-01 tic=6.940e-10 sm=3.593e+00)   10.1s
  step  250   loss = 4.3735e-02   (data=4.694e-04 phys=1.538e+01 ic=8.867e-05 tic=1.469e-11 sm=1.550e+01)   14.5s
  step  500   loss = 3.8884e-02   (data=3.714e-04 phys=1.013e+01 ic=4.157e-05 tic=1.921e-11 sm=1.896e+01)   18.1s
  step  750   loss = 3.3602e-02   (data=3.335e-04 phys=7.867e+00 ic=1.657e-05 tic=2.545e-11 sm=1.620e+01)   21.7s
  step 1000   loss = 3.3225e-02   (data=3.286e-04 phys=6.250e+00 ic=1.580e-05 tic=2.828e-11 sm=1.705e+01)   25.3s
  step 1250   loss = 3.1261e-02   (data=2.864e-04 phys=7.113e+00 ic=1.035e-05 tic=3.197e-11 sm=1.671e+01)   29.0s
  step 1500   loss = 3.1669e-02   (data=3.232e-04 phys=7.246e+00 ic=2.899e-05 tic=3.339e-11 sm=1.479e+01)   32.9s
  step 1750   loss = 3.4914e-02   (data=4.559e-04 phys=6.438e+00 ic=1.152e-05 tic=2.848e-11 sm=1.110e+01)   37.0s
  step 2000   loss = 3.0396e-02   (data=2.590e-04 phys=6.586e+00 ic=1.590e-05 tic=2.360e-11 sm=1.765e+01)   40.8s
  step 2250   loss = 2.9527e-02   (data=3.257e-04 phys=7.682e+00 ic=1.082e-04 tic=3.008e-11 sm=1.148e+01)   45.0s
  step 2500   loss = 3.6585e-02   (data=4.755e-04 phys=6.080e+00 ic=5.782e-05 tic=3.182e-11 sm=1.207e+01)   48.8s
  step 2750   loss = 2.8050e-02   (data=2.961e-04 phys=6.456e+00 ic=6.031e-05 tic=3.551e-11 sm=1.237e+01)   52.7s
  step 3000   loss = 2.6139e-02   (data=2.803e-04 phys=7.390e+00 ic=4.466e-05 tic=2.172e-11 sm=1.042e+01)   56.8s
  step 3250   loss = 2.2529e-02   (data=2.331e-04 phys=6.850e+00 ic=1.113e-05 tic=2.642e-11 sm=9.280e+00)   60.4s
  step 3500   loss = 2.4835e-02   (data=2.523e-04 phys=8.402e+00 ic=1.374e-05 tic=2.643e-11 sm=9.991e+00)   64.0s
  step 3750   loss = 2.4990e-02   (data=2.356e-04 phys=7.163e+00 ic=7.005e-06 tic=2.820e-11 sm=1.202e+01)   67.7s
  step 4000   loss = 2.1949e-02   (data=2.272e-04 phys=7.609e+00 ic=7.087e-06 tic=2.762e-11 sm=8.464e+00)   71.3s
  step 4250   loss = 2.1249e-02   (data=2.433e-04 phys=5.735e+00 ic=1.137e-05 tic=2.727e-11 sm=7.740e+00)   75.0s
  step 4500   loss = 2.2997e-02   (data=2.445e-04 phys=5.862e+00 ic=1.902e-05 tic=2.456e-11 sm=9.756e+00)   78.7s
  step 4750   loss = 2.2298e-02   (data=2.355e-04 phys=6.733e+00 ic=3.270e-05 tic=3.090e-11 sm=8.861e+00)   82.5s
  step 5000   loss = 2.0736e-02   (data=2.455e-04 phys=5.633e+00 ic=1.195e-05 tic=2.000e-11 sm=7.027e+00)   86.3s
  step 5250   loss = 2.2926e-02   (data=2.800e-04 phys=6.240e+00 ic=2.838e-05 tic=2.883e-11 sm=7.185e+00)   90.0s
  step 5500   loss = 1.8452e-02   (data=2.103e-04 phys=4.959e+00 ic=1.780e-05 tic=3.334e-11 sm=6.778e+00)   93.7s
  step 5750   loss = 1.9383e-02   (data=2.099e-04 phys=5.336e+00 ic=1.259e-05 tic=2.927e-11 sm=7.741e+00)   97.4s
  step 6000   loss = 1.7696e-02   (data=2.203e-04 phys=5.418e+00 ic=8.706e-06 tic=3.251e-11 sm=4.941e+00)   101.0s
  step 6250   loss = 1.7035e-02   (data=1.879e-04 phys=6.253e+00 ic=1.013e-05 tic=2.819e-11 sm=5.619e+00)   104.7s
  step 6500   loss = 1.7277e-02   (data=1.995e-04 phys=5.271e+00 ic=1.376e-05 tic=4.733e-11 sm=5.797e+00)   108.3s
  step 6750   loss = 1.6574e-02   (data=1.944e-04 phys=5.032e+00 ic=9.385e-06 tic=3.604e-11 sm=5.397e+00)   111.9s
  step 7000   loss = 1.7191e-02   (data=2.001e-04 phys=6.318e+00 ic=8.082e-06 tic=2.614e-11 sm=5.011e+00)   115.5s
  step 7250   loss = 1.6643e-02   (data=2.075e-04 phys=5.435e+00 ic=9.906e-06 tic=3.918e-11 sm=4.412e+00)   119.2s
  step 7500   loss = 1.7169e-02   (data=1.931e-04 phys=7.589e+00 ic=2.065e-05 tic=2.979e-11 sm=4.595e+00)   122.9s
  step 7750   loss = 1.6209e-02   (data=1.983e-04 phys=5.728e+00 ic=2.125e-05 tic=2.883e-11 sm=4.233e+00)   126.4s
  step 8000   loss = 1.6317e-02   (data=1.897e-04 phys=6.149e+00 ic=1.429e-05 tic=4.500e-11 sm=4.664e+00)   130.2s
Adam phase done (130.2 s)

ψ recovery:
  L2 err = 0.0876 m   (relative: 53.3 %)
  truth peak = +0.450 m,  recovered peak = +0.240 m

Wrote data/psi_recovered.csv and data/gauges_pinn_pred.csv
```
:::
