"""
Runs the trained keypoint model + the calibrated CropFollow++ state
estimation pipeline (Eq. 22-24, src/camera_geometry.py) on every image in
a folder, saving annotated output images and a summary CSV into a new
folder.

Usage:
    python3 -m src.batch_infer /path/to/images_folder
    python3 -m src.batch_infer /path/to/images_folder --output /path/to/out
    python3 -m src.batch_infer /path/to/images_folder --checkpoint checkpoints/best.pt
"""
import argparse
import csv
import os
import sys

import cv2
import torch
from tqdm import tqdm

from src.config import CLASS_NAMES, DEVICE, ROW_SPACING_M
from src.camera_geometry import estimate_state_with_steps
from src.infer import draw_row_lines, draw_state_text, get_camera_params, infer_image, load_model

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


def find_images(folder):
    files = []
    for name in sorted(os.listdir(folder)):
        ext = os.path.splitext(name)[1].lower()
        if ext in IMAGE_EXTENSIONS:
            files.append(os.path.join(folder, name))
    return files


def process_folder(images_folder, output_folder, checkpoint_path, device):
    os.makedirs(output_folder, exist_ok=True)

    image_paths = find_images(images_folder)
    if not image_paths:
        print(f"No images found in {images_folder} "
              f"(looked for extensions: {sorted(IMAGE_EXTENSIONS)})")
        return

    print(f"Found {len(image_paths)} images in {images_folder}")
    print(f"Loading model from {checkpoint_path} onto {device}...")
    model = load_model(checkpoint_path, device)

    csv_path = os.path.join(output_folder, "batch_results.csv")
    csv_fields = (
        ["image_name", "pitch_deg", "heading_deg", "d_ratio",
         "distance_left_m", "distance_right_m", "state_error"]
        + [f"{c}_x" for c in CLASS_NAMES] + [f"{c}_y" for c in CLASS_NAMES]
        + [f"{c}_confidence" for c in CLASS_NAMES] + [f"{c}_outlier" for c in CLASS_NAMES]
    )

    n_ok, n_failed, n_unusable = 0, 0, 0
    warned_about_focal_length = False

    with open(csv_path, "w", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=csv_fields)
        writer.writeheader()

        for image_path in tqdm(image_paths, desc="Processing"):
            image_name = os.path.basename(image_path)
            row = {"image_name": image_name}

            try:
                image_bgr, points, confidences = infer_image(model, image_path, device)
            except Exception as e:
                print(f"\n[FAILED] {image_name}: {e}", file=sys.stderr)
                n_failed += 1
                row["state_error"] = f"inference failed: {e}"
                writer.writerow(row)
                continue

            h, w = image_bgr.shape[:2]
            focal_length_px, principal_point, roll_rad, used_placeholder = get_camera_params(w, h)
            if used_placeholder and not warned_about_focal_length:
                print(f"\n[WARNING] FOCAL_LENGTH_PX not set in config.py -- using "
                      f"image width as a placeholder for this whole batch. See "
                      f"config.py's comment on FOCAL_LENGTH_PX for what to do "
                      f"on a real robot.")
                warned_about_focal_length = True

            state = estimate_state_with_steps(
                points, focal_length_px, principal_point, roll_rad, ROW_SPACING_M,
                image_height=h, image_width=w
            )

            for class_name in CLASS_NAMES:
                pts = points[class_name]
                row[f"{class_name}_x"] = pts[0][0] if pts else ""
                row[f"{class_name}_y"] = pts[0][1] if pts else ""
                row[f"{class_name}_confidence"] = f"{confidences[class_name]['confidence']:.4f}"
                row[f"{class_name}_outlier"] = confidences[class_name]["is_outlier"]

            if "error" in state:
                n_unusable += 1
                row["state_error"] = state["error"]
            else:
                row["pitch_deg"] = f"{state['pitch_deg']:.3f}"
                row["heading_deg"] = f"{state['heading_deg']:.3f}"
                row["d_ratio"] = f"{state['d_ratio']:.4f}"
                if "distance_left_m" in state:
                    row["distance_left_m"] = f"{state['distance_left_m']:.3f}"
                    row["distance_right_m"] = f"{state['distance_right_m']:.3f}"
                row["state_error"] = ""

            annotated = draw_row_lines(image_bgr, points)
            annotated = draw_state_text(annotated, state)
            out_path = os.path.join(output_folder, image_name)
            cv2.imwrite(out_path, annotated)

            writer.writerow(row)
            n_ok += 1

    print(f"\nDone. {n_ok} images processed, {n_unusable} unusable "
          f"(missing keypoints or [could not recognize]), {n_failed} failed to load/run.")
    print(f"Annotated images saved to: {output_folder}")
    print(f"Per-image summary CSV saved to: {csv_path}")
    if ROW_SPACING_M is None:
        print("Note: ROW_SPACING_M is not set in config.py, so distance_left_m/"
              "distance_right_m are blank in the CSV -- only the dimensionless "
              "d_ratio is available. Set ROW_SPACING_M to your field's real row "
              "spacing to get metric distances.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("images_folder", help="Folder of input images to process")
    parser.add_argument("--output", default=None,
                         help="Output folder (default: <images_folder>_results, "
                              "created as a sibling of the input folder)")
    parser.add_argument("--checkpoint", default="checkpoints/best.pt")
    args = parser.parse_args()

    images_folder = os.path.abspath(args.images_folder)
    if not os.path.isdir(images_folder):
        print(f"Not a folder: {images_folder}")
        sys.exit(1)

    if args.output:
        output_folder = os.path.abspath(args.output)
    else:
        parent = os.path.dirname(images_folder.rstrip("/"))
        base_name = os.path.basename(images_folder.rstrip("/"))
        output_folder = os.path.join(parent, f"{base_name}_results")

    device = DEVICE if torch.cuda.is_available() else "cpu"
    process_folder(images_folder, output_folder, args.checkpoint, device)


if __name__ == "__main__":
    main()
