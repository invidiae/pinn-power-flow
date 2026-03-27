import scipy.sparse
import torch

Ybus = scipy.sparse.load_npz("data/Ybus.npz")

G = torch.tensor(Ybus.real, dtype=torch.float32)  # Conductance matrix
B = torch.tensor(Ybus.imag, dtype=torch.float32)  # Susceptance matrix

"""
Aufpassen: 
- theta und V dürfen nicht skaliert sein
- theta muss in rad sein
- V muss in pu sein
- Wo wendet man am besten die skalierung an?
""" 

def calculate_p(V_pred, Theta_pred):
    """
    V_pred: (batch_size, num_buses)
    Theta_pred: (batch_size, num_buses) -> RADIANS!!
    """
    # 1. Compute theta_ij (matrix of differences)
    # This uses broadcasting: (N, 1) - (1, N) = (N, N)
    # result[i, j] = theta[i] - theta[j]
    theta_ij = Theta_pred.unsqueeze(2) - Theta_pred.unsqueeze(1)

    # 2. Calculate the term inside the parentheses
    # (G_ij * cos(theta_ij) + B_ij * sin(theta_ij))
    term = G * torch.cos(theta_ij) + B * torch.sin(theta_ij)

    # 3. Multiply by V_j and sum over j
    # V_pred.unsqueeze(1) makes V a row vector for matrix multiplication
    summation = torch.sum(V_pred.unsqueeze(1) * term, dim=2)

    # 4. Multiply by V_i
    P_calc = V_pred * summation
    
    return P_calc # This is your "Physics-based P"

def physics_loss(v_pred, theta_pred, p_actual, q_actual, Y_bus):
    # 1. Use v_pred and theta_pred to calculate P_calc and Q_calc 
    #    using the Power Balance Equations (matrix multiplication with Y_bus)
    p_calc = calculate_p(v_pred, theta_pred, Y_bus)
    q_calc = calculate_q(v_pred, theta_pred, Y_bus)
    
    # 2. Compute the residual (how far off the physics are)
    res_p = torch.mean((p_calc - p_actual)**2)
    res_q = torch.mean((q_calc - q_actual)**2)
    
    return res_p + res_q