import matplotlib.pyplot as plt
import numpy as np
import torch
import tqdm
from torch import nn
from torch.utils.data import DataLoader

from preprocessing import load_processed

X_train, X_val, X_test, y_train, y_val, y_test, scaler_X, scaler_y = load_processed()

## Naive Model – predict mean of training targets
y_train_mean = y_train.mean(axis=0)
y_pred_naive = np.tile(y_train_mean, (y_test.shape[0], 1))

## define loss function

loss_fn = nn.MSELoss()

## loss naive model
naive_loss = loss_fn(torch.from_numpy(y_pred_naive), torch.from_numpy(y_test))


train_dl = DataLoader(list(zip(X_train, y_train)), batch_size=32, shuffle=True)
val_dl = DataLoader(list(zip(X_val, y_val)), batch_size=32, shuffle=False)
test_dl = DataLoader(list(zip(X_test, y_test)), batch_size=32, shuffle=False)


# for X, y in train_dl:
#     print(f"Shape of X [N, C]: {X.shape}")
#     print(f"Shape of y [N, C]: {y.shape} {y.dtype}")
#     break ## just look at one batch, looks as expected

class net(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, output_dim)
        )
        
    def forward(self, x):
        return self.net(x)

n = net(input_dim=X_train.shape[1], output_dim=y_train.shape[1])
## define optimizer
optimizer = torch.optim.Adam(n.parameters(), lr=1e-3)


## train
def train(train_dl, model, loss_fn, optimizer):
    size = len(train_dl.dataset)
    model.train()
    for batch, (X, y) in enumerate(train_dl):

        # Compute prediction error
        pred = model(X)
        loss = loss_fn(pred, y)

        # Backpropagation
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        if batch % 100 == 0:
            loss, current = loss.item(), (batch + 1) * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")

def evaluate(dl, model, loss_fn):
    num_batches = len(dl)
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for X, y in dl:
            total_loss += loss_fn(model(X), y).item()
    return total_loss / num_batches


EPOCHS = 100
PATIENCE = 10

best_val_loss = float("inf")
epochs_without_improvement = 0
train_losses = []
val_losses = []

for epoch in tqdm.tqdm(range(EPOCHS)):
    train(train_dl, n, loss_fn, optimizer)
    train_loss = evaluate(train_dl, n, loss_fn)
    val_loss = evaluate(val_dl, n, loss_fn)
    train_losses.append(train_loss)
    val_losses.append(val_loss)
    print(f"Epoch {epoch+1:>3d} — train loss: {train_loss:.6f}  val loss: {val_loss:.6f}")

    if val_loss < best_val_loss:
        best_val_loss = val_loss
        epochs_without_improvement = 0
        torch.save(n.state_dict(), "data/best_model.pt")
    else:
        epochs_without_improvement += 1
        if epochs_without_improvement >= PATIENCE:
            print(f"Early stopping at epoch {epoch+1}")
            break

n.load_state_dict(torch.load("data/best_model.pt"))
test_loss = evaluate(test_dl, n, loss_fn)
print(f"Test loss: {test_loss:.6f}")

plt.plot(train_losses, label="train")
plt.plot(val_losses, label="val")
plt.axhline(y=naive_loss.item(), color="gray", linestyle="--", label="naive baseline")
plt.xlabel("Epoch")
plt.ylabel("MSE Loss")
plt.legend()
plt.tight_layout()
plt.savefig("data/loss_curve.png", dpi=150)
plt.show()