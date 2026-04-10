import time
import numpy as np
import torch
import torch.nn as nn

class SurrogateNN(nn.Module):   # n_features = number of input features, 
                                # hidden_size = number of neurons in the single hidden layer

    def __init__(self, n_features: int, hidden_size: int = 64):
        super().__init__()
        self.hidden = nn.Linear(n_features, hidden_size)
        self.relu   = nn.ReLU()
        self.output = nn.Linear(hidden_size, 1)
        self.sigmoid = nn.Sigmoid()

        # Xavier init for stable gradients
        nn.init.xavier_uniform_(self.hidden.weight)
        nn.init.zeros_(self.hidden.bias)
        nn.init.xavier_uniform_(self.output.weight)
        nn.init.zeros_(self.output.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.relu(self.hidden(x))
        x = self.sigmoid(self.output(x))
        return x.squeeze(-1)

def train_surrogate(
    X:           np.ndarray,    # Perturbed samples (N, F)
    y:           np.ndarray,    # Black-box model predictions (N,)
    w:           np.ndarray,    # Kernel weights (N,)
    hidden_size: int   = 64,    # Hidden layer size
    lr:          float = 1e-3,  # Adam learning rate
    n_epochs:    int   = 500,   # Number of full passes over the data
    batch_size:  int   = 256,   # Mini-batch size
    device:      str   = "cpu", # 'cpu' or 'cuda'
    verbose:     bool  = True,  # Print loss every 100 epochs
) -> tuple:
    device = torch.device(device)

    X_t = torch.tensor(X, dtype=torch.float32).to(device)
    y_t = torch.tensor(y, dtype=torch.float32).to(device)
    w_t = torch.tensor(w, dtype=torch.float32).to(device)

    # Normalise weights so the maximum importance = 1.0
    w_t = w_t / (w_t.max() + 1e-8)

    n_samples, n_features = X.shape
    model     = SurrogateNN(n_features, hidden_size).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=n_epochs)
    criterion = nn.BCELoss(reduction="none")   # We apply weights manually

    # Build mini-batch index tensor
    indices = torch.arange(n_samples, device=device)

    loss_history = []

    for epoch in range(n_epochs):
        model.train()

        # Shuffle indices each epoch
        perm = torch.randperm(n_samples, device=device)

        epoch_loss = 0.0
        n_batches  = 0

        for start in range(0, n_samples, batch_size):
            idx       = perm[start : start + batch_size]
            X_batch   = X_t[idx]
            y_batch   = y_t[idx]
            w_batch   = w_t[idx]

            optimizer.zero_grad()
            preds        = model(X_batch)
            loss_per_s   = criterion(preds, y_batch)
            weighted_loss = (w_batch * loss_per_s).mean()
            weighted_loss.backward()
            optimizer.step()

            epoch_loss += weighted_loss.item()
            n_batches  += 1

        scheduler.step()
        avg_loss = epoch_loss / n_batches
        loss_history.append(avg_loss)

        if verbose and (epoch + 1) % 100 == 0:
            print(f"  Epoch {epoch+1:4d}/{n_epochs}  |  "
                  f"Weighted BCE Loss: {avg_loss:.6f}  |  "
                  f"LR: {scheduler.get_last_lr()[0]:.2e}")

    return model, loss_history

# Measures mean training time over n_repeats runs and returns a dict with mean and std
def benchmark_surrogate(
    X:           np.ndarray,
    y:           np.ndarray,
    w:           np.ndarray,
    hidden_size: int = 64,
    n_epochs:    int = 500,
    n_repeats:   int = 3,
    device:      str = "cpu",
) -> dict:
    times = []
    for _ in range(n_repeats):
        t0 = time.perf_counter()
        train_surrogate(
            X, y, w,
            hidden_size=hidden_size,
            n_epochs=n_epochs,
            device=device,
            verbose=False,
        )
        times.append((time.perf_counter() - t0) * 1000)

    return {
        "mean_ms": float(np.mean(times)),
        "std_ms":  float(np.std(times)),
    }
