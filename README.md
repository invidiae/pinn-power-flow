# Power Flow PINN — IEEE 14-Bus

Train a Physics-Informed Neural Network (PINN) to predict AC power flow solutions faster than Newton-Raphson.

---

## Goal

Replace the iterative Newton-Raphson solver (pandapower `runpp`) with a neural network that maps load conditions → bus voltages in a single forward pass.

**Input:** Active and reactive load per load bus (P, Q) → flattened vector of length `2 × (n_loads+n_generators)`
**Output:** Voltage magnitude (pu) and angle (degrees) per bus → flattened vector of length `2 × n_buses`

---

## Steps

### 1. Data Generation

Generate a dataset of (load, voltage) pairs by sampling random operating points around the IEEE 14-bus baseline and running Newton-Raphson for targets.

- Load the IEEE 14-bus network via `pandapower.networks.case14()`
- For each sample, perturb all load and generator P and Q values by a random scaling factor
- Run `pp.runpp(net)` to solve the power flow
- Save inputs (`net.load[['p_mw', 'q_mvar']]`) and targets (`net.res_bus[['vm_pu', 'va_degree']]`) to a CSV file
- Discard samples where the solver fails to converge

**Considerations:**
- Generate enough samples to cover the operating space
- Widen the perturbation range gradually; too wide causes non-convergence (#TODO was bedeutet das für das Netz?)
---

### 2. Dataset Preprocessing

Prepare the raw DataFrame for training. Run `preprocessing.py` directly to generate all artifacts:

```
python preprocessing.py
```

This will:
- Split columns into inputs `X` (load P/Q) and targets `y` (bus V/θ)
- Split into train/validation/test sets (70/15/15, `random_state=42`)
- Fit a `StandardScaler` on the training set only (mean=0, std=1) and apply to both splits
- Save the scaled arrays to `data/processed.npz`
- Save the fitted scalers to `data/scaler_X.pkl` and `data/scaler_Y.pkl`

To load the preprocessed data in other modules:

```python
from preprocessing import load_processed

X_train, X_val, X_test, y_train, y_val, y_test, scaler_X, scaler_y = load_processed()

# Invert scaling on model predictions before evaluation
y_pred_original = scaler_y.inverse_transform(y_pred_scaled)
```

> `processed.npz` and `*.pkl` files are gitignored — they are derived from the CSV and should be regenerated locally.

---

### 3. Baseline: Pure Data-Driven Neural Network

Before adding physics constraints, establish a plain MLP baseline.

- Architecture: fully-connected network, 2 hidden layers, 128 units each, ReLU activations
- Loss: MSE on predicted vs. true voltages and angles
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

- Framework: PyTorch
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
├── preprocessing.py     # scaling, train/test split, artifact export
├── main.py              # entry point
├── data/
│   ├── ieee_14.csv      # raw samples (committed)
│   ├── processed.npz    # scaled arrays (gitignored, regenerate locally)
│   ├── scaler_X.pkl     # fitted input scaler (gitignored)
│   └── scaler_Y.pkl     # fitted target scaler (gitignored)
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
