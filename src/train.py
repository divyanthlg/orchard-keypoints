import os

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.config import (
    BATCH_SIZE, CHECKPOINT_DIR, DEVICE, EPOCHS, LR, NUM_WORKERS, WEIGHT_DECAY,
)
from src.dataset import build_train_val_datasets
from src.losses import CombinedKeypointLoss
from src.model import KeypointHeatmapNet


def run_epoch(model, loader, criterion, optimizer, device, train=True):
    model.train() if train else model.eval()
    total_loss = 0.0
    breakdown_sum = {}

    context = torch.enable_grad() if train else torch.no_grad()
    with context:
        for images, heatmaps in tqdm(loader, leave=False):
            images = images.to(device)
            heatmaps = heatmaps.to(device)

            preds = model(images)
            loss, breakdown = criterion(preds, heatmaps)

            if train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            total_loss += loss.item() * images.size(0)
            for k, v in breakdown.items():
                breakdown_sum[k] = breakdown_sum.get(k, 0.0) + v * images.size(0)

    n = len(loader.dataset)
    avg_breakdown = {k: v / n for k, v in breakdown_sum.items()}
    return total_loss / n, avg_breakdown


def main():
    device = DEVICE if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    train_ds, val_ds = build_train_val_datasets()
    train_loader = DataLoader(
        train_ds, batch_size=BATCH_SIZE, shuffle=True,
        num_workers=NUM_WORKERS, pin_memory=(device == "cuda"),
    )
    val_loader = DataLoader(
        val_ds, batch_size=BATCH_SIZE, shuffle=False,
        num_workers=NUM_WORKERS, pin_memory=(device == "cuda"),
    )

    model = KeypointHeatmapNet(pretrained=True).to(device)
    criterion = CombinedKeypointLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=5
    )

    best_val_loss = float("inf")

    for epoch in range(1, EPOCHS + 1):
        train_loss, train_breakdown = run_epoch(model, train_loader, criterion, optimizer, device, train=True)
        val_loss, val_breakdown = run_epoch(model, val_loader, criterion, optimizer, device, train=False)
        scheduler.step(val_loss)

        current_lr = optimizer.param_groups[0]["lr"]
        print(f"Epoch {epoch:3d}/{EPOCHS} | train_loss {train_loss:.4f} | "
              f"val_loss {val_loss:.4f} | lr {current_lr:.2e}")
        print(f"  val breakdown: " +
              ", ".join(f"{k}={v:.4f}" for k, v in val_breakdown.items()))

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            path = os.path.join(CHECKPOINT_DIR, "best.pt")
            torch.save({"model_state": model.state_dict(), "epoch": epoch,
                        "val_loss": val_loss}, path)
            print(f"  -> saved new best checkpoint ({path})")

    final_path = os.path.join(CHECKPOINT_DIR, "last.pt")
    torch.save({"model_state": model.state_dict(), "epoch": EPOCHS}, final_path)
    print(f"Training done. Final checkpoint: {final_path}")


if __name__ == "__main__":
    main()
