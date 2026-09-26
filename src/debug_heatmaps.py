"""
Diagnostic tool: shows the RAW sigmoid heatmap for every channel, before
any confidence threshold is applied. Use this when decode_heatmaps() is
returning empty lists for some classes -- it tells you whether the model
produced a real-but-low-confidence peak (fixable with a lower threshold /
more training) or genuinely nothing at all (a training problem).

Usage:
    python3 debug_heatmaps.py path/to/image.png --checkpoint checkpoints/best.pt
"""
import argparse

import cv2
import numpy as np
import torch

from src.config import CLASS_NAMES, DEVICE, INPUT_SIZE
from src.dataset import IMAGENET_MEAN, IMAGENET_STD
from src.model import KeypointHeatmapNet


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("image_path")
    parser.add_argument("--checkpoint", default="checkpoints/best.pt")
    args = parser.parse_args()

    device = DEVICE if torch.cuda.is_available() else "cpu"
    model = KeypointHeatmapNet(pretrained=False).to(device)
    ckpt = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    print(f"Loaded checkpoint from epoch {ckpt.get('epoch', '?')}, "
          f"val_loss {ckpt.get('val_loss', '?')}")

    image_bgr = cv2.imread(args.image_path)
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    resized = cv2.resize(image_rgb, (INPUT_SIZE, INPUT_SIZE))
    norm = (resized.astype(np.float32) / 255.0 - IMAGENET_MEAN) / IMAGENET_STD
    tensor = torch.from_numpy(norm.transpose(2, 0, 1)).float().unsqueeze(0).to(device)

    with torch.no_grad():
        logits = model(tensor)[0]
        probs = torch.sigmoid(logits)

    print("\nPer-channel raw confidence (post-sigmoid, before any threshold):")
    print(f"{'class':20s} {'max':>8s} {'mean':>10s}   interpretation")
    for ch, name in enumerate(CLASS_NAMES):
        channel = probs[ch]
        max_val = channel.max().item()
        mean_val = channel.mean().item()
        if max_val < 0.05:
            note = "dead - model never learned this class"
        elif max_val < 0.3:
            note = "weak signal - present but below default threshold"
        else:
            note = "strong peak - should be detected"
        print(f"{name:20s} {max_val:8.4f} {mean_val:10.6f}   {note}")

    # save each channel as its own grayscale heatmap image for visual inspection
    for ch, name in enumerate(CLASS_NAMES):
        heat = (probs[ch].cpu().numpy() * 255).astype(np.uint8)
        heat_resized = cv2.resize(heat, (INPUT_SIZE, INPUT_SIZE))
        heat_color = cv2.applyColorMap(heat_resized, cv2.COLORMAP_JET)
        overlay = cv2.addWeighted(cv2.resize(image_bgr, (INPUT_SIZE, INPUT_SIZE)),
                                   0.6, heat_color, 0.4, 0)
        out_path = f"debug_heatmap_{name}.png"
        cv2.imwrite(out_path, overlay)
        print(f"Saved {out_path}")


if __name__ == "__main__":
    main()
