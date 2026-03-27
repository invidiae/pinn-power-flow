import matplotlib.pyplot as plt
import torch
import tqdm
from torch import nn
from torch.utils.data import DataLoader

from preprocessing import load_processed

X_train, X_val, X_test, y_train, y_val, y_test, scaler_X, scaler_y = load_processed()


loss_fn = nn.MSELoss()

train_dl = DataLoader(list(zip(X_train, y_train)), batch_size=32, shuffle=True)
val_dl = DataLoader(list(zip(X_val, y_val)), batch_size=32, shuffle=False)
test_dl = DataLoader(list(zip(X_test, y_test)), batch_size=32, shuffle=False)


class net(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, output_dim),
        )

    def reset_parameters(self):
        for layer in self.net:
            if hasattr(layer, "reset_parameters"):
                layer.reset_parameters()

    def forward(self, x):
        return self.net(x)


n = net(input_dim=X_train.shape[1], output_dim=y_train.shape[1])

n.reset_parameters()

optimizer = torch.optim.Adam(n.parameters(), lr=1e-3)


## train
def train(train_dl, model, loss_fn, optimizer):
    model.train()
    for batch, (X, y) in tqdm.tqdm(enumerate(train_dl)):
        # Compute prediction error
        pred = model(X)
        loss = loss_fn(pred, y)

        # Backpropagation
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()


def evaluate(dl, model, loss_fn):
    num_batches = len(dl)
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for X, y in dl:
            total_loss += loss_fn(model(X), y).item()
    return total_loss / num_batches


def run_training(train_dl, model, loss_fn, optimizer, n_epochs=10, patience=None):
    if patience is None:
        patience = n_epochs // 10
    
    best_val_loss = float("inf")
    epochs_without_improvement = 0
    train_losses = []
    val_losses = []

    for epoch in tqdm.tqdm(range(n_epochs)):
        train(train_dl, n, loss_fn, optimizer)
        train_loss = evaluate(train_dl, n, loss_fn)
        val_loss = evaluate(val_dl, n, loss_fn)
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        print(
            f"Epoch {epoch + 1:>3d} — train loss: {train_loss:.6f}  val loss: {val_loss:.6f}"
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_without_improvement = 0
            torch.save(n.state_dict(), "data/best_model.pt")
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= patience:
                print(f"Early stopping at epoch {epoch + 1}")
                break

    model.load_state_dict(torch.load("data/best_model.pt"))
    test_loss = evaluate(test_dl, n, loss_fn)
    return test_loss, train_losses, val_losses

if __name__ == "__main__":
    n.reset_parameters()
    test_loss, train_losses, val_losses = run_training(train_dl, n, loss_fn, optimizer, n_epochs=30, patience=3)

    print(f"Test loss: {test_loss:.6f}")

    plt.plot(train_losses, label="train")
    plt.plot(val_losses, label="val")
    plt.xlabel("Epoch")
    plt.ylabel("MSE Loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig("data/loss_curve.png", dpi=150)
    plt.show()