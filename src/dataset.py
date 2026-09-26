import os
import random

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from src.config import (
    CLASS_NAMES, GAUSSIAN_SIGMA, IMAGES_DIR, INPUT_SIZE, JSON_PATH,
    OUTPUT_STRIDE, SEED, VAL_FRACTION,
)
from src.hasty_parser import load_hasty_json

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def _draw_gaussian(heatmap, center_x, center_y, sigma):
    """Draw a 2D gaussian centered at (center_x, center_y) into heatmap,
    taking the pixel-wise max with whatever's already there."""
    h, w = heatmap.shape
    tmp_size = int(sigma * 3)
    x0, y0 = int(round(center_x)), int(round(center_y))

    left, right = max(0, x0 - tmp_size), min(w, x0 + tmp_size + 1)
    top, bottom = max(0, y0 - tmp_size), min(h, y0 + tmp_size + 1)
    if left >= right or top >= bottom:
        return heatmap

    xs = np.arange(left, right)
    ys = np.arange(top, bottom)[:, None]
    gauss = np.exp(-((xs - center_x) ** 2 + (ys - center_y) ** 2) / (2 * sigma ** 2))

    heatmap[top:bottom, left:right] = np.maximum(
        heatmap[top:bottom, left:right], gauss
    )
    return heatmap


class OrchardKeypointDataset(Dataset):
    def __init__(self, records, augment=False):
        self.records = records
        self.augment = augment
        self.out_size = INPUT_SIZE // OUTPUT_STRIDE

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        rec = self.records[idx]
        img_path = os.path.join(IMAGES_DIR, rec["image_name"])
        image = cv2.imread(img_path)
        if image is None:
            raise FileNotFoundError(f"Could not read image: {img_path}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        orig_h, orig_w = image.shape[:2]
        points = rec["points"]

        # blur + color jitter only, matching CropFollow++ Ch.4, minus
        # horizontal flip (left/right stay fixed, no class-swap needed)
        if self.augment:
            image = self._color_jitter(image)
            if random.random() < 0.3:
                image = cv2.GaussianBlur(image, (5, 5), sigmaX=random.uniform(0.5, 1.5))

        image = cv2.resize(image, (INPUT_SIZE, INPUT_SIZE))
        scale_x = INPUT_SIZE / orig_w
        scale_y = INPUT_SIZE / orig_h

        heatmaps = np.zeros((len(CLASS_NAMES), self.out_size, self.out_size), dtype=np.float32)
        for ch, class_name in enumerate(CLASS_NAMES):
            for (x, y) in points[class_name]:
                hx = x * scale_x / OUTPUT_STRIDE
                hy = y * scale_y / OUTPUT_STRIDE
                if 0 <= hx < self.out_size and 0 <= hy < self.out_size:
                    _draw_gaussian(heatmaps[ch], hx, hy, GAUSSIAN_SIGMA)

        image_norm = (image.astype(np.float32) / 255.0 - IMAGENET_MEAN) / IMAGENET_STD
        image_tensor = torch.from_numpy(image_norm.transpose(2, 0, 1)).float()
        heatmap_tensor = torch.from_numpy(heatmaps).float()

        return image_tensor, heatmap_tensor

    @staticmethod
    def _color_jitter(image):
        alpha = random.uniform(0.8, 1.2)
        beta = random.uniform(-20, 20)
        jittered = image.astype(np.float32) * alpha + beta
        return np.clip(jittered, 0, 255).astype(np.uint8)


def build_train_val_datasets():
    records = load_hasty_json(JSON_PATH)
    records = [r for r in records if any(len(v) > 0 for v in r["points"].values())]

    random.Random(SEED).shuffle(records)
    n_val = max(1, int(len(records) * VAL_FRACTION))
    val_records = records[:n_val]
    train_records = records[n_val:]

    print(f"Total annotated images: {len(records)}  "
          f"(train: {len(train_records)}, val: {len(val_records)})")

    train_ds = OrchardKeypointDataset(train_records, augment=True)
    val_ds = OrchardKeypointDataset(val_records, augment=False)
    return train_ds, val_ds


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    train_ds, val_ds = build_train_val_datasets()
    n_show = min(3, len(train_ds))
    for i in range(n_show):
        image_tensor, heatmap_tensor = train_ds[i]
        image = image_tensor.permute(1, 2, 0).numpy()
        image = (image * IMAGENET_STD + IMAGENET_MEAN).clip(0, 1)

        combined_heatmap = heatmap_tensor.numpy().max(axis=0)
        combined_heatmap_resized = cv2.resize(combined_heatmap, (INPUT_SIZE, INPUT_SIZE))

        fig, axes = plt.subplots(1, 2, figsize=(10, 5))
        axes[0].imshow(image)
        axes[0].set_title("Input image")
        axes[1].imshow(image)
        axes[1].imshow(combined_heatmap_resized, cmap="jet", alpha=0.5)
        axes[1].set_title("Combined heatmap overlay (all 4 channels)")
        for ax in axes:
            ax.axis("off")
        plt.tight_layout()
        plt.savefig(f"sanity_check_{i}.png", dpi=120)
        plt.close(fig)
        print(f"Saved sanity_check_{i}.png")
