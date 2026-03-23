### load model
### apply to test set
### compute metrics (MSE, MAE, R2, % error) for both voltage magnitude and angle
import numpy as np
import torch

from net import n
from preprocessing import load_processed

_, _, X_test, _, _, y_test, scaler_X, scaler_y = load_processed()

n.load_state_dict(torch.load("data/best_model.pt"))
n.eval()

y_pred = n(torch.from_numpy(X_test)).detach().numpy()

y_pred_unscaled = scaler_y.inverse_transform(y_pred)
y_test_unscaled = scaler_y.inverse_transform(y_test)

n_buses = y_test_unscaled.shape[1] // 2
def mape(true, pred):
    with np.errstate(divide="ignore", invalid="ignore"):
        pct = np.where(true == 0, np.nan, np.abs(true - pred) / np.abs(true))
    return np.nanmean(pct) * 100

vm_mae = mape(y_test_unscaled[:, :n_buses], y_pred_unscaled[:, :n_buses])
va_mae = mape(y_test_unscaled[:, n_buses:], y_pred_unscaled[:, n_buses:])

print(f"MAE voltage magnitude: {vm_mae:.6f} pu")
print(f"MAE voltage angle:     {va_mae:.6f} deg")
