from __future__ import annotations

import os
import json
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

from cnn.models import PixelMLP
from cnn.data import NPZPixelDataset


def main():
    # Match how RF/SVM load the dataset (see random_forest/model_rf.py and SVM/model_svm_linear.py)
    data_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "wetland_dataset_1.5M_4Training.npz")
    data = np.load(data_path)
    X = data["X"]          # (N, 64)
    y = data["y"]          # (N,)
    class_weights = data["class_weights"]  # (6,)
    data.close()

    # Train/val/test split
    X_train, X_tmp, y_train, y_tmp = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
    X_val, X_test, y_val, y_test = train_test_split(X_tmp, y_tmp, test_size=0.5, random_state=42, stratify=y_tmp)

    # Normalize using train stats only
    mean = X_train.mean(axis=0)
    std = X_train.std(axis=0)

    train_ds = NPZPixelDataset(X_train, y_train, mean=mean, std=std)
    val_ds = NPZPixelDataset(X_val, y_val, mean=mean, std=std)
    test_ds = NPZPixelDataset(X_test, y_test, mean=mean, std=std)

    train_loader = DataLoader(train_ds, batch_size=4096, shuffle=True, num_workers=2, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=4096, shuffle=False, num_workers=2, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=4096, shuffle=False, num_workers=2, pin_memory=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = PixelMLP(in_features=64, num_classes=6, hidden=256, dropout=0.2).to(device)

    # Use provided class weights (see Helper Files/training_dataset_validation.txt)
    cw = torch.tensor(class_weights, dtype=torch.float32, device=device)
    criterion = nn.CrossEntropyLoss(weight=cw)

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)

    best_val_acc = -1.0
    best_state = None

    for epoch in range(1, 11):
        model.train()
        total_loss = 0.0

        for xb, yb in train_loader:
            xb = xb.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()

            total_loss += float(loss.item()) * xb.size(0)

        train_loss = total_loss / len(train_ds)

        # Validate
        model.eval()
        y_pred = []
        y_true = []
        with torch.no_grad():
            for xb, yb in val_loader:
                xb = xb.to(device, non_blocking=True)
                logits = model(xb)
                pred = torch.argmax(logits, dim=1).cpu().numpy()
                y_pred.append(pred)
                y_true.append(yb.numpy())

        y_pred = np.concatenate(y_pred)
        y_true = np.concatenate(y_true)
        val_acc = accuracy_score(y_true, y_pred)

        print(f"Epoch {epoch:02d} | train_loss={train_loss:.4f} | val_acc={val_acc:.4f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    # Test using best checkpoint
    if best_state is not None:
        model.load_state_dict(best_state)

    model.eval()
    y_pred = []
    y_true = []
    with torch.no_grad():
        for xb, yb in test_loader:
            xb = xb.to(device, non_blocking=True)
            logits = model(xb)
            pred = torch.argmax(logits, dim=1).cpu().numpy()
            y_pred.append(pred)
            y_true.append(yb.numpy())

    y_pred = np.concatenate(y_pred)
    y_true = np.concatenate(y_true)

    acc = accuracy_score(y_true, y_pred)
    cm = confusion_matrix(y_true, y_pred)
    report = classification_report(y_true, y_pred, digits=4)

    print("\nTEST ACC:", acc)
    print("\nCONFUSION MATRIX:\n", cm)
    print("\nREPORT:\n", report)

    # Save model + metadata (similar pattern to random_forest/model_rf.py)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_model = os.path.join(os.path.dirname(__file__), f"pixel_mlp_v1_{timestamp}.pt")
    out_meta = os.path.join(os.path.dirname(__file__), f"pixel_mlp_v1_{timestamp}_metadata.json")

    torch.save(
        {
            "state_dict": model.state_dict(),
            "mean": mean.astype(np.float32),
            "std": std.astype(np.float32),
            "class_weights": class_weights.astype(np.float32),
        },
        out_model,
    )

    meta = {
        "timestamp": timestamp,
        "model_type": "PixelMLP",
        "input_features": 64,
        "num_classes": 6,
        "best_val_acc": float(best_val_acc),
        "test_acc": float(acc),
        "dataset": {"source": "../wetland_dataset_1.5M_4Training.npz"},
    }
    with open(out_meta, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    print(f"\nSaved: {out_model}")
    print(f"Saved: {out_meta}")


if __name__ == "__main__":
    main()