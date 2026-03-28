import numpy as np
import torch

from preprocessing import load_processed
from run_pinn import net

_, _, X_test, _, _, y_test, _, _, _, _, scaler_X, scaler_y = load_processed()

n_pi = net(input_dim=X_test.shape[1], output_dim=y_test.shape[1])
n_pi.load_state_dict(torch.load("data/best_model_pi.pt"))
n_pi.eval()

y_pred = n_pi(torch.from_numpy(X_test)).detach().numpy()

y_pred_unscaled = scaler_y.inverse_transform(y_pred)
y_test_unscaled = scaler_y.inverse_transform(y_test)

n_buses = y_test_unscaled.shape[1] // 2

va_mae = np.mean(np.abs(y_test_unscaled[:, n_buses:] - y_pred_unscaled[:, n_buses:]))
vm_mae = np.mean(np.abs(y_test_unscaled[:, :n_buses] - y_pred_unscaled[:, :n_buses]))


print(f"PINN MAE voltage magnitude: {vm_mae:.6f} pu")
print(f"PINN MAE voltage angle:     {va_mae:.6f} deg")


#### evaluate normal net

from net import net as net_normal

nnn = net_normal(input_dim=X_test.shape[1], output_dim=y_test.shape[1])
nnn.load_state_dict(torch.load("data/best_model.pt"))
nnn.eval()

y_pred_normal = nnn(torch.from_numpy(X_test)).detach().numpy()

y_pred_normal_unscaled = scaler_y.inverse_transform(y_pred_normal)


va_mae_n = np.mean(np.abs(y_test_unscaled[:, n_buses:] - y_pred_normal_unscaled[:, n_buses:]))
vm_mae_n = np.mean(np.abs(y_test_unscaled[:, :n_buses] - y_pred_normal_unscaled[:, :n_buses]))

print(f"NN MAE voltage magnitude: {vm_mae_n:.6f} pu")
print(f"NN MAE voltage angle:     {va_mae_n:.6f} deg")