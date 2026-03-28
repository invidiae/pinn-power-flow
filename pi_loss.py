import torch

"""
Aufpassen: 
- einen batch reinladen
- theta und V dürfen nicht skaliert sein (sollte passen)
- theta muss in rad sein (sollte passen)
- V muss in pu sein (sollte passen)
- Wo wendet man am besten die skalierung an?
- Ich habe P_bus and Q_bus implementiert, jetzt muss
    aber auch das slicing in preprocessing angepasst werden!
    Reihenfolge: load p, load q, gen p, bus V, bus theta, bus p, bus q
"""

## load in one batch of predictions and targets


def calculate_p_q(V_pred, Theta_pred, G, B):
    """
    V_pred: (batch_size, num_buses)
    Theta_pred: (batch_size, num_buses) -> RADIANS!!
    """
    # 1. Compute theta_ij (matrix of differences)
    # This uses broadcasting: (N, 1) - (1, N) = (N, N)
    # result[i, j] = theta[i] - theta[j] ToDo: understand this better!

    theta_ij = Theta_pred.unsqueeze(2) - Theta_pred.unsqueeze(1)

    # 2. Calculate the term inside the parentheses
    # (G_ij * cos(theta_ij) + B_ij * sin(theta_ij))
    term_p = G * torch.cos(theta_ij) + B * torch.sin(theta_ij)

    # 3. Multiply by V_j and sum over j
    # V_pred.unsqueeze(1) makes V a row vector for matrix multiplication
    summation_p = torch.sum(V_pred.unsqueeze(1) * term_p, dim=2)

    # 4. Multiply by V_i
    P_calc = V_pred * summation_p

    term_q = G * torch.sin(theta_ij) - B * torch.cos(theta_ij)
    summation_q = torch.sum(V_pred.unsqueeze(1) * term_q, dim=2)

    Q_calc = V_pred * summation_q
    return P_calc, Q_calc


def physics_loss(y_pred, p_actual, q_actual, G, B, mean_y, std_y):
    y_unscaled = y_pred * std_y + mean_y

    V_pred = y_unscaled[:, :14]
    theta_pred = y_unscaled[:, 14:] * (torch.pi / 180)  # Convert degrees to radians

    p_calc, q_calc = calculate_p_q(V_pred, theta_pred, G, B)

    # Convert MW/Mvar → pu and negate: res_bus uses load convention (positive=consuming),
    # power flow equations use generation convention (positive=injecting)
    p_actual_pu = -p_actual / 100.0
    q_actual_pu = -q_actual / 100.0

    res_p = torch.mean((p_calc - p_actual_pu) ** 2)
    res_q = torch.mean((q_calc - q_actual_pu) ** 2)
    return res_p + res_q


def loss_fn_pi(
    y_pred, y_true, p_actual, q_actual, G, B, mean_y, std_y, alpha=1.0, beta=1.0
):
    data_loss = torch.mean((y_pred - y_true) ** 2)
    phys_loss = physics_loss(y_pred, p_actual, q_actual, G, B, mean_y, std_y)
    total_loss = alpha * data_loss + beta * phys_loss
    return total_loss  ## evtl phys_loss und data_loss auch returnen, für analysis



### sanity check: compute physics loss for perfect predictions (should be zero)
if __name__ == "__main__":
    # Dummy data for testing
    import pandas as pd
    import scipy.sparse

    columns_names = pd.read_csv("data/ieee_14.csv").columns.tolist()
    raw_df = pd.read_csv("data/ieee_14.csv").head(30).to_numpy()
    ## load X – erste 24 columns
    x = torch.tensor(raw_df[:, :26])
    y = torch.tensor(raw_df[:, 26:54])
    mean_y = y.mean()
    std_y = y.var()
    y = (y - mean_y ) / std_y
    p_actual = torch.tensor(raw_df[:, 54:68])
    q_actual = torch.tensor(raw_df[:, 68:82])
    Ybus = scipy.sparse.load_npz("data/Ybus.npz")
    G     = torch.tensor(Ybus.real.toarray(), dtype=torch.float32)
    B     = torch.tensor(Ybus.imag.toarray(), dtype=torch.float32)
    
    loss = physics_loss(y, p_actual, q_actual, G, B, mean_y, std_y)
    print(f"Physics loss for perfect predictions: {loss.item():.6f} (should be close to 0)")