import matplotlib.pyplot as plt
import scipy.sparse
import torch
import tqdm
from torch import nn
from torch.utils.data import DataLoader

from pi_loss import loss_fn_pi
from preprocessing import load_processed


class net(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
            nn.Linear(64, output_dim),
        )

    def reset_parameters(self):
        for layer in self.net:
            if hasattr(layer, "reset_parameters"):
                layer.reset_parameters()

    def forward(self, x):
        return self.net(x)

def train(train_dl, model, loss_fn, optimizer, G, B, mean_y, std_y, alpha, beta):
    model.train()
    for batch, (X, y, p, q) in tqdm.tqdm(enumerate(train_dl)):
        # Compute prediction error
        pred = model(X)
        loss = loss_fn(pred, y, p, q, G, B, mean_y, std_y, alpha, beta)

        # Backpropagation
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()


def final_evaluate(dl, model):
    mae = nn.L1Loss()
    num_batches = len(dl)
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for X, y in dl:
            total_loss += mae(model(X), y).item()
    return total_loss / num_batches


def evaluate(dl, model, loss_fn, G, B, mean_y, std_y, alpha, beta):
    num_batches = len(dl)
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for X, y, p, q in dl:
            total_loss += loss_fn(model(X), y, p, q, G, B, mean_y, std_y, alpha, beta).item()
    return total_loss / num_batches


def run_training(train_dl, model, loss_fn, optimizer, G, B, mean_y, std_y, n_epochs=10, patience=None, alpha=1.0, beta=1.0):
    if patience is None:
        patience = n_epochs // 10

    best_val_loss = float("inf")
    epochs_without_improvement = 0
    train_losses = []
    val_losses = []

    for epoch in tqdm.tqdm(range(n_epochs)):
        train(train_dl, model, loss_fn, optimizer, G, B, mean_y, std_y, alpha, beta)
        train_loss = evaluate(train_dl, model, loss_fn, G, B, mean_y, std_y, alpha, beta)
        val_loss = evaluate(val_dl, model, loss_fn, G, B, mean_y, std_y, alpha, beta)
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        #print(
        #    f"Epoch {epoch + 1:>3d} — train loss: {train_loss:.6f}  val loss: {val_loss:.6f}"
        #)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_without_improvement = 0
            torch.save(model.state_dict(), "data/best_model_pi.pt")
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= patience:
                print(f"Early stopping at epoch {epoch + 1}")
                break

    model.load_state_dict(torch.load("data/best_model_pi.pt"))
    test_loss = final_evaluate(test_dl, model)
    return test_loss, train_losses, val_losses


if __name__ == "__main__":
    X_train, X_val, X_test, y_train, y_val, y_test, p_train, p_val, q_train, q_val, scaler_X, scaler_y = load_processed()

    Ybus = scipy.sparse.load_npz("data/Ybus.npz")
    G     = torch.tensor(Ybus.real.toarray(), dtype=torch.float32)
    B     = torch.tensor(Ybus.imag.toarray(), dtype=torch.float32)
    mean_y = torch.tensor(scaler_y.mean_,  dtype=torch.float32)
    std_y  = torch.tensor(scaler_y.scale_, dtype=torch.float32)

    train_dl = DataLoader(list(zip(X_train, y_train, p_train, q_train)), batch_size=32, shuffle=True)
    val_dl = DataLoader(list(zip(X_val, y_val, p_val, q_val)), batch_size=32, shuffle=False)
    test_dl = DataLoader(list(zip(X_test, y_test)), batch_size=32, shuffle=False)

    n_pi = net(input_dim=X_train.shape[1], output_dim=y_train.shape[1])
    n_pi.reset_parameters()
    optimizer = torch.optim.Adam(n_pi.parameters(), lr=1e-3)
    loss_fn = loss_fn_pi
    test_loss, train_losses, val_losses = run_training(train_dl, n_pi, loss_fn, optimizer, G, B, mean_y, std_y, n_epochs=100, beta=0.5, patience=80)
    plt.plot(val_losses[:80], label="Val Loss PINN")
    plt.xlabel("Epoch")
    plt.ylabel("MSE Loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig("data/loss_curve.png", dpi=150)
    plt.show()