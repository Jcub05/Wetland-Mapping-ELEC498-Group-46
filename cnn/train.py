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

# Import the new ResNet transfer learning model
from cnn.models import ResNet18Wetland
from cnn.data import NPZPatchDataset


def main():
    # Load the new geographically split 15x15 CNN dataset
    data_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "wetland_cnn_dataset_15x15.npz")
    
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Could not find dataset at: {data_path}")
        
    print(f"Loading data from: {data_path}")
    data = np.load(data_path)
    
    # 1. Direct Assignment (No more training leakage!)
    X_train = data["X_train"] 
    y_train = data["y_train"]
    class_weights = data["class_weights"]
    
    # 2. Split the validation tiles into Val and Test sets
    X_val_tiles = data["X_val"]
    y_val_tiles = data["y_val"]
    
    data.close()
    
    X_val, X_test, y_val, y_test = train_test_split(
        X_val_tiles, y_val_tiles, test_size=0.50, random_state=42, stratify=y_val_tiles
    )

    print(f"Loaded X_train: {X_train.shape} | y_train: {y_train.shape}")
    print(f"Loaded X_val:   {X_val.shape} | y_val:   {y_val.shape}")
    print(f"Loaded X_test:  {X_test.shape} | y_test:  {y_test.shape}")

    # Normalize per-channel using train stats only
    mean = X_train.mean(axis=(0, 2, 3))
    std = X_train.std(axis=(0, 2, 3))

    train_ds = NPZPatchDataset(X_train, y_train, mean=mean, std=std, is_train=True)
    val_ds = NPZPatchDataset(X_val, y_val, mean=mean, std=std)
    test_ds = NPZPatchDataset(X_test, y_test, mean=mean, std=std)

    # 15x15 patches are memory-light, allowing for much larger batch sizes
    batch_size = 256 
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Training on device: {device}")

    # Use the 15x15 optimized ResNet-18 Transfer Learning model
    model = ResNet18Wetland(in_channels=64, num_classes=6, dropout=0.3).to(device)

    cw = torch.tensor(class_weights, dtype=torch.float32, device=device)
    criterion = nn.CrossEntropyLoss(weight=cw)
    
    # Optimizer and Learning Rate Scheduler
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    
    epochs = 15
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    best_val_acc = -1.0
    best_state = None

    print(f"Starting training for {epochs} epochs...")
    
    for epoch in range(1, epochs + 1):
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
        y_pred, y_true = [], []
        with torch.no_grad():
            for xb, yb in val_loader:
                xb = xb.to(device, non_blocking=True)
                logits = model(xb)
                pred = torch.argmax(logits, dim=1).cpu().numpy()
                y_pred.append(pred)
                y_true.append(yb.numpy())

        val_acc = accuracy_score(np.concatenate(y_true), np.concatenate(y_pred))

        current_lr = scheduler.get_last_lr()[0]
        print(f"Epoch {epoch:02d} | train_loss={train_loss:.4f} | val_acc={val_acc:.4f} | lr={current_lr:.2e}")

        # Step the learning rate scheduler
        scheduler.step()

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    # Test using best checkpoint
    print("\nEvaluating on Test Set...")
    if best_state is not None:
        model.load_state_dict(best_state)

    model.eval()
    y_pred, y_true = [], []
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

    # Save model + metadata
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_model = os.path.join(os.path.dirname(__file__), f"wetland_cnn_v3_resnet_{timestamp}.pt")
    out_meta = os.path.join(os.path.dirname(__file__), f"wetland_cnn_v3_resnet_{timestamp}_metadata.json")

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
        "model_type": "ResNet18Wetland",
        "input_channels": 64,
        "patch_size": 15,
        "num_classes": 6,
        "best_val_acc": float(best_val_acc),
        "test_acc": float(acc),
        "optimizer": "AdamW_CosineAnnealingLR",
        "dataset": {"source": "wetland_cnn_dataset_15x15.npz"},
    }
    with open(out_meta, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    print(f"\nSaved: {out_model}")
    print(f"Saved: {out_meta}")

if __name__ == "__main__":
    main()