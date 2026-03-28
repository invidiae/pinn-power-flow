## Power Flow PINN — IEEE 14-Bus

Train a Physics-Informed Neural Network (PINN) to predict AC power flow solutions faster than Newton-Raphson. This is a work in progress – I am learning along the way.


## Goal

Replace Newton-Raphson solver (pandapower `runpp`) with a neural network that maps load and generator conditions to bus voltages.


## Steps

### 1. Data Generation

Generate a dataset of (load, voltage) pairs by sampling random operating points around the IEEE 14-bus baseline and running Newton-Raphson for targets.

- Load the IEEE 14-bus network
- For each sample, perturb all load and generator P and Q values by a random scaling factor
- Solve the power flow
- Save inputs and targets to a CSV file
- Discard samples where the solver fails to converge
- Compute Ybus for calculating physics loss
- Todo: Get a better understanding what non-convergence means


### 2. Dataset Preprocessing

Prepare the raw DataFrame for training. Run `preprocessing.py` directly to generate all artifacts:

This will:
- Split columns into inputs `X` (load P/Q) and targets `y` (bus V/θ)
- Split into train/validation/test sets
- Scale the data
- Save the scaled arrays to `data/processed.npz`
- Save the fitted scalers to `data/scaler_X.pkl` and `data/scaler_Y.pkl`


### 3. Baseline: Pure Data-Driven Neural Network

Before adding physics constraints, establish a plain MLP baseline.

- Architecture: fully-connected network, 2 hidden layers, 128 units each, ReLU activations
- Loss: MSE on predicted vs. true voltages and angles
- Train, evaluate, and record prediction error (in the literature, MAE seems to be standard)
- Thats what the PINN should beat

### 4. Physics Loss

Augment the data loss with a residual that penalises violations of the AC power flow equations.

The power balance equations at each bus `i` are:

```
P_i = V_i * Σ_j [ V_j * (G_ij cos(θ_i - θ_j) + B_ij sin(θ_i - θ_j)) ]
Q_i = V_i * Σ_j [ V_j * (G_ij sin(θ_i - θ_j) - B_ij cos(θ_i - θ_j)) ]
```

where `G + iB = Y_bus` (admittance matrix, derivable from the network).

**Implementation:**
- Extract `Y_bus` from pandapower
- Implement the residuals as differentiable PyTorch operations
- Total loss: `L = α * L_data + β * L_physics`
- Tune `α`, `β` – a common schedule starts with data loss only, then gradually increases `β`

preliminary results 
für 100 epochs, 200 samples, pertubation in interval x0,8-x1,2, bs=32, adam optimizer lr=1e-3, beta=0,5\, no early stopping
PINN MAE voltage magnitude: 0.003682 pu\
PINN MAE voltage angle:     0.050539 deg\
NN MAE voltage magnitude: 0.007656 pu\
NN MAE voltage angle:     0.078240 deg

for pertubation x0,6-x1,4, the magnitudes are same same, but voltage angle approx. are significantly worse:\
PINN MAE voltage magnitude: 0.003670 pu\
PINN MAE voltage angle:     0.099308 deg\
NN MAE voltage magnitude: 0.008621 pu\
NN MAE voltage angle:     0.161284 deg

this trend continues, pertubation x0-5 for loads and x0-3 for gen:\
PINN MAE voltage magnitude: 0.006393 pu\
PINN MAE voltage angle:     0.817888 deg\
NN MAE voltage magnitude: 0.010739 pu\
NN MAE voltage angle:     1.182836 deg


At this point, the voltage angles are so far off, such an approximation seems questionable for real use.

Todo: Look at inference time and time Newton-Raphson.

### 5. Evaluation

Compare the trained PINN against Newton-Raphson on a held-out test set.

| Metric | Target |
|---|---|
| MAE `vm_pu` | < 0.005 pu |
| MAE `va_degree` | < 0.1° |
| Inference time (single sample) | < 1 ms (vs ~5–50 ms NR) |
| Physics residual norm | Near zero |

Also test on out-of-distribution operating points (e.g. ±40% load) to assess generalisation.