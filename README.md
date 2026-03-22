# Power Flow PINN — IEEE 14-Bus

Train a Physics-Informed Neural Network (PINN) to predict AC power flow solutions faster than Newton-Raphson.

---

## Goal

Replace the iterative Newton-Raphson solver (pandapower `runpp`) with a neural network that maps load conditions → bus voltages in a single forward pass.

**Input:** Active and reactive load per load bus (P, Q) → flattened vector of length `2 × n_loads`
**Output:** Voltage magnitude (pu) and angle (degrees) per bus → flattened vector of length `2 × n_buses`

---

## Steps

### 1. Data Generation

Generate a dataset of (load, voltage) pairs by sampling random operating points around the IEEE 14-bus baseline and running Newton-Raphson for ground truth.

- Load the IEEE 14-bus network via `pandapower.networks.case14()`
- For each sample, perturb all load and generator P and Q values by a random scaling factor (e.g. ±20%)
- Run `pp.runpp(net)` to solve the power flow
- Save inputs (`net.load[['p_mw', 'q_mvar']]`) and targets (`net.res_bus[['vm_pu', 'va_degree']]`) to a CSV/numpy file
- Discard samples where the solver fails to converge

See [tutorial.py](tutorial.py) for a working implementation of this step.

**Considerations:**
- Generate enough samples to cover the operating space (start with ~5 000–10 000)
- Widen the perturbation range gradually; too wide causes non-convergence
- Store the baseline load values so perturbations are always relative to them (currently [tutorial.py](tutorial.py) modifies `net.load` in-place — this needs fixing before scaling up)

---

### 2. Dataset Preprocessing

Prepare the raw DataFrame for training.

- Split columns back into inputs `X` and targets `y`
- Normalize/standardize both `X` and `y` (z-score or min-max) — voltage magnitudes live near 1 pu, angles near 0°, but loads vary widely
- Train/validation/test split (e.g. 70/15/15)
- Save scalers so they can be inverted at inference time

---

### 3. Baseline: Pure Data-Driven Neural Network

Before adding physics constraints, establish a plain MLP baseline.

- Architecture: fully-connected network, 3–5 hidden layers, 128–256 units each, ReLU activations
- Loss: MSE on predicted vs. true voltages
- Train, evaluate, and record prediction error (MAE on `vm_pu`, `va_degree`)
- This sets the accuracy floor that the PINN should meet or beat

---

### 4. Physics Loss (the PINN part)

Augment the data loss with a residual that penalises violations of the AC power flow equations.

The power balance equations at each bus `i` are:

```
P_i = V_i * Σ_j [ V_j * (G_ij cos(θ_i - θ_j) + B_ij sin(θ_i - θ_j)) ]
Q_i = V_i * Σ_j [ V_j * (G_ij sin(θ_i - θ_j) - B_ij cos(θ_i - θ_j)) ]
```

where `G + jB = Y_bus` (admittance matrix, derivable from the network).

**Implementation:**
- Extract `Y_bus` from pandapower (`net._ppc["internal"]["Ybus"]` or build from `net.line`/`net.trafo`)
- Implement the residuals as differentiable PyTorch/JAX operations
- Total loss: `L = α * L_data + β * L_physics`
- Tune `α`, `β` — a common schedule starts with data loss only, then gradually increases `β`

---

### 5. Training

- Framework: PyTorch (or JAX/Flax)
- Optimiser: Adam, learning rate ~1e-3 with cosine decay
- Batch size: 256–512
- Early stopping on validation loss
- Log: total loss, data loss, physics residual norm separately

---

### 6. Evaluation

Compare the trained PINN against Newton-Raphson on a held-out test set.

| Metric | Target |
|---|---|
| MAE `vm_pu` | < 0.005 pu |
| MAE `va_degree` | < 0.1° |
| Inference time (single sample) | < 1 ms (vs ~5–50 ms NR) |
| Physics residual norm | Near zero |

Also test on out-of-distribution operating points (e.g. ±40% load) to assess generalisation.

---

### 7. Iteration

- If accuracy is insufficient: more training data, wider network, tighter physics weight
- If physics residual is high: check `Y_bus` construction, verify angle convention (radians vs degrees)
- If generalisation is poor: add noise to training inputs, use dropout, or extend the sampling range

---

## File Structure (planned)

```
competition/
├── tutorial.py          # data generation prototype
├── main.py              # entry point
├── data/
│   └── ieee14_samples.csv
├── models/
│   ├── baseline_mlp.py
│   └── pinn.py
├── train.py
├── evaluate.py
└── README.md
```

---

## Dependencies

- `pandapower` — network model and Newton-Raphson solver
- `numpy`, `pandas` — data handling
- `torch` — model and training
- `scikit-learn` — preprocessing, metrics
- `tqdm` — progress bars
