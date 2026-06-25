"""
Train both learned scorers (QuantRegressor + RAMDemoter) from smart-kv cache data.

Usage:
  python train_learned_score.py <dataset.bin> [output_dir]

Dataset format (from learned-score.h ls_export_dataset):
  [header: 2 floats: n_samples, n_features=17]
  [X:       n_samples * 17 floats]
  [y_quant: n_samples * 1 float]
  [y_ram:   n_samples * 1 float]
  [mem_pressure: n_samples * 1 float]

Output:
  quant_weights.bin  — QuantRegressor (17→16→1, 289 floats)
  ram_weights.bin    — RAMDemoter (18→8→1, 161 floats)
"""

import sys
import struct
import os
import numpy as np
import torch
import torch.nn as nn

# ── Architectures (matching learned-score.h) ──────────────────────────

QR_IN_DIM  = 21
QR_HID_DIM = 16
QR_OUT_DIM = 1

RD_IN_DIM  = 22  # 21 base + memory_pressure
RD_HID_DIM = 8
RD_OUT_DIM = 1

class QuantRegressor(nn.Module):
    """Ordinal regression for quant tier (1-5).
    Input: 21 features → hidden 16 → sigmoid(1) → score 0-1
    """
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(QR_IN_DIM, QR_HID_DIM),
            nn.ReLU(),
            nn.Linear(QR_HID_DIM, QR_OUT_DIM),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return self.net(x)


class RAMDemoter(nn.Module):
    """Binary classifier for RAM eviction (tier 6).
    Input: 22 features (21 base + memory_pressure) → hidden 8 → sigmoid(1)
    """
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(RD_IN_DIM, RD_HID_DIM),
            nn.ReLU(),
            nn.Linear(RD_HID_DIM, RD_OUT_DIM),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return self.net(x)


# ── Load binary dataset ────────────────────────────────────────────────

def load_dataset(path):
    with open(path, "rb") as f:
        header = struct.unpack("ff", f.read(8))
        n_samples = int(header[0])
        n_features = int(header[1])
        assert n_features == QR_IN_DIM, f"expected {QR_IN_DIM} features, got {n_features}"

        # X: features
        X = np.frombuffer(f.read(n_samples * n_features * 4), dtype=np.float32
                         ).reshape(n_samples, n_features)
        # y_quant: quant labels
        y_quant = np.frombuffer(f.read(n_samples * 4), dtype=np.float32)
        # y_ram: RAM labels
        y_ram = np.frombuffer(f.read(n_samples * 4), dtype=np.float32)
        # mem_pressure
        mem_pressure = np.frombuffer(f.read(n_samples * 4), dtype=np.float32)

    return X, y_quant, y_ram, mem_pressure


# ── Export weights to binary (C-compatible) ────────────────────────────

def export_quant_weights(model, path):
    w1 = model.net[0].weight.data.cpu().numpy().T   # [21, 16]
    b1 = model.net[0].bias.data.cpu().numpy()        # [16]
    w2 = model.net[2].weight.data.cpu().numpy().T    # [16, 1]
    b2 = model.net[2].bias.data.cpu().numpy()         # [1]

    with open(path, "wb") as f:
        f.write(w1.astype(np.float32).tobytes())
        f.write(b1.astype(np.float32).tobytes())
        f.write(w2.astype(np.float32).tobytes())
        f.write(b2.astype(np.float32).tobytes())

    total = w1.size + b1.size + w2.size + b2.size
    print(f"  exported {path} ({total} floats)")


def export_ram_weights(model, path):
    w1 = model.net[0].weight.data.cpu().numpy().T   # [22, 8]
    b1 = model.net[0].bias.data.cpu().numpy()        # [8]
    w2 = model.net[2].weight.data.cpu().numpy().T    # [8, 1]
    b2 = model.net[2].bias.data.cpu().numpy()         # [1]

    with open(path, "wb") as f:
        f.write(w1.astype(np.float32).tobytes())
        f.write(b1.astype(np.float32).tobytes())
        f.write(w2.astype(np.float32).tobytes())
        f.write(b2.astype(np.float32).tobytes())

    total = w1.size + b1.size + w2.size + b2.size
    print(f"  exported {path} ({total} floats)")


# ── Training ────────────────────────────────────────────────────────────

def train_quant(X, y_quant, device):
    """Train QuantRegressor (MSE loss on normalized tier 0-1)."""
    print(f"\n{'='*60}")
    print(f"  QuantRegressor: {X.shape[0]} samples, {X.shape[1]} features")
    print(f"{'='*60}")

    n = X.shape[0]
    split = int(n * 0.8)
    X_train = torch.from_numpy(X[:split])
    y_train = torch.from_numpy(y_quant[:split]).unsqueeze(1)
    X_val   = torch.from_numpy(X[split:])
    y_val   = torch.from_numpy(y_quant[split:]).unsqueeze(1)

    print(f"  train: {len(X_train)}, val: {len(X_val)}")

    model = QuantRegressor().to(device)
    loss_fn = nn.MSELoss()
    lr = 0.01
    epochs = 150

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, epochs)
    batch_size = min(256, len(X_train))
    dataset = torch.utils.data.TensorDataset(X_train, y_train)
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

    best_val = float("inf")
    best_params = None

    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        for bx, by in loader:
            bx, by = bx.to(device), by.to(device)
            optimizer.zero_grad()
            pred = model(bx)
            loss = loss_fn(pred, by)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            train_loss += loss.item()

        model.eval()
        with torch.no_grad():
            val_loss = loss_fn(model(X_val.to(device)), y_val.to(device)).item()

        scheduler.step()

        if (epoch + 1) % 20 == 0 or epoch == 0:
            mae = torch.abs(model(X_val.to(device)) - y_val.to(device)).mean().item()
            print(f"  epoch {epoch+1:3d} | train_loss={train_loss/len(loader):.4f} val_loss={val_loss:.4f} val_mae={mae:.4f}")

        if val_loss < best_val:
            best_val = val_loss
            best_params = [p.detach().cpu().clone() for p in model.parameters()]

    if best_params:
        for p, best_p in zip(model.parameters(), best_params):
            p.data.copy_(best_p)

    # Final evaluation
    model.eval()
    with torch.no_grad():
        pred = model(torch.from_numpy(X).to(device))
        mae = torch.abs(pred - torch.from_numpy(y_quant).unsqueeze(1).to(device)).mean().item()
        print(f"\n  Best val loss: {best_val:.4f}")
        print(f"  Final MAE: {mae:.4f}")

    return model


def train_ram(X, y_ram, device):
    """Train RAMDemoter (weighted BCE — FP=3×, FN=1×)."""
    print(f"\n{'='*60}")
    print(f"  RAMDemoter: {X.shape[0]} samples, {X.shape[1]} features")
    print(f"{'='*60}")

    n = X.shape[0]
    split = int(n * 0.8)
    X_train = torch.from_numpy(X[:split])
    y_train = torch.from_numpy(y_ram[:split]).unsqueeze(1)
    X_val   = torch.from_numpy(X[split:])
    y_val   = torch.from_numpy(y_ram[split:]).unsqueeze(1)

    print(f"  train: {len(X_train)}, val: {len(X_val)}")
    pos_rate = y_train.mean().item()
    print(f"  RAM eviction rate (train): {pos_rate:.3f}")

    model = RAMDemoter().to(device)
    loss_fn = nn.BCELoss(reduction='none')
    lr = 0.005
    epochs = 150

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, epochs)
    batch_size = min(256, len(X_train))
    dataset = torch.utils.data.TensorDataset(X_train, y_train)
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

    best_val = float("inf")
    best_params = None

    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        for bx, by in loader:
            bx, by = bx.to(device), by.to(device)
            optimizer.zero_grad()
            pred = model(bx)
            # Weighted loss: FP (predict=1, truth=0) = 3×, FN (predict=0, truth=1) = 1×
            loss = loss_fn(pred, by)
            weights = torch.where(
                (by > 0.5) & (pred < 0.5), 1.0,  # FN
                torch.where((by < 0.5) & (pred > 0.5), 3.0, 1.0)  # FP, TP, TN
            )
            loss = (loss * weights).mean()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            train_loss += loss.item()

        model.eval()
        with torch.no_grad():
            val_pred = model(X_val.to(device))
            val_loss = loss_fn(val_pred, y_val.to(device))
            weights_val = torch.where(
                (y_val.to(device) > 0.5) & (val_pred < 0.5), 1.0,
                torch.where((y_val.to(device) < 0.5) & (val_pred > 0.5), 3.0, 1.0)
            )
            val_loss = (val_loss * weights_val).mean().item()

        scheduler.step()

        if (epoch + 1) % 20 == 0 or epoch == 0:
            # Metrics: precision, recall, F1 for RAM class
            pred_bin = (model(X_val.to(device)) > 0.5).float()
            tp = ((pred_bin > 0.5) & (y_val.to(device) > 0.5)).sum().item()
            fp = ((pred_bin > 0.5) & (y_val.to(device) < 0.5)).sum().item()
            fn = ((pred_bin < 0.5) & (y_val.to(device) > 0.5)).sum().item()
            precision = tp / (tp + fp + 1e-8)
            recall = tp / (tp + fn + 1e-8)
            f1 = 2 * precision * recall / (precision + recall + 1e-8)
            print(f"  epoch {epoch+1:3d} | val_loss={val_loss:.4f} prec={precision:.3f} rec={recall:.3f} F1={f1:.3f}")

        if val_loss < best_val:
            best_val = val_loss
            best_params = [p.detach().cpu().clone() for p in model.parameters()]

    if best_params:
        for p, best_p in zip(model.parameters(), best_params):
            p.data.copy_(best_p)

    # Final evaluation
    model.eval()
    with torch.no_grad():
        pred_all = model(torch.from_numpy(X).to(device))
        pred_bin = (pred_all > 0.5).float()
        y_t = torch.from_numpy(y_ram).unsqueeze(1).to(device)
        tp = ((pred_bin > 0.5) & (y_t > 0.5)).sum().item()
        fp = ((pred_bin > 0.5) & (y_t < 0.5)).sum().item()
        fn = ((pred_bin < 0.5) & (y_t > 0.5)).sum().item()
        tn = ((pred_bin < 0.5) & (y_t < 0.5)).sum().item()
        precision = tp / (tp + fp + 1e-8)
        recall = tp / (tp + fn + 1e-8)
        f1 = 2 * precision * recall / (precision + recall + 1e-8)
        acc = (tp + tn) / (tp + tn + fp + fn)
        print(f"\n  Best val loss: {best_val:.4f}")
        print(f"  Final: acc={acc:.3f} prec={precision:.3f} rec={recall:.3f} F1={f1:.3f}")
        print(f"  Confusion: TP={tp} FP={fp} FN={fn} TN={tn}")

    return model


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)

    dataset_path = sys.argv[1]
    output_dir = sys.argv[2] if len(sys.argv) > 2 else os.path.dirname(dataset_path)

    print(f"Loading dataset: {dataset_path}")
    X, y_quant, y_ram, mem_pressure = load_dataset(dataset_path)
    n = X.shape[0]
    print(f"  {n} samples, {X.shape[1]} features")
    print(f"  quant labels: mean={y_quant.mean():.3f}, range=[{y_quant.min():.3f}, {y_quant.max():.3f}]")
    print(f"  ram labels:   pos_rate={y_ram.mean():.3f} ({int(y_ram.sum())}/{n})")
    print(f"  mem_pressure: mean={mem_pressure.mean():.3f}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"  device: {device}")

    # ── Train QuantRegressor ──
    quant_model = train_quant(X, y_quant, device)

    # ── Train RAMDemoter (X + memory_pressure as 18th feature) ──
    X_ram = np.column_stack([X, mem_pressure])
    ram_model = train_ram(X_ram, y_ram, device)

    # ── Export ──
    os.makedirs(output_dir, exist_ok=True)
    qpath = os.path.join(output_dir, "quant_weights.bin")
    rpath = os.path.join(output_dir, "ram_weights.bin")

    print(f"\n{'='*60}")
    print(f"  EXPORTING")
    print(f"{'='*60}")
    export_quant_weights(quant_model, qpath)
    export_ram_weights(ram_model, rpath)

    print(f"\nDone. Copy both files to the plugin directory:")
    print(f"  {qpath}")
    print(f"  {rpath}")


if __name__ == "__main__":
    main()
