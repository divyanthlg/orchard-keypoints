import argparse
import math
import os

import cv2
import numpy as np
import torch

from src.camera_geometry import estimate_state_with_steps
from src.config import (
    CAMERA_ROLL_DEG, CLASS_NAMES, DEVICE, FOCAL_LENGTH_PX, INPUT_SIZE,
    PRINCIPAL_POINT, ROW_SPACING_M,
)
from src.confidence import keypoint_confidence
from src.dataset import IMAGENET_MEAN, IMAGENET_STD
from src.decode import decode_heatmaps
from src.model import KeypointHeatmapNet
from src.visualize_steps import make_step_figure

# BGR (OpenCV order):
# green line = vanishing_point (left)        -> left_intercept
# blue line  = vanishing_point_right (right) -> right_intercept
# red        = keypoint markers (all 4)
COLOR_LEFT_LINE = (0, 200, 0)
COLOR_RIGHT_LINE = (255, 0, 0)
COLOR_POINT = (0, 0, 255)


def load_model(checkpoint_path, device):
    model = KeypointHeatmapNet(pretrained=False).to(device)
    ckpt = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    return model


def infer_image(model, image_path, device):
    image_bgr = cv2.imread(image_path)
    if image_bgr is None:
        raise FileNotFoundError(image_path)
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    orig_h, orig_w = image_rgb.shape[:2]

    resized = cv2.resize(image_rgb, (INPUT_SIZE, INPUT_SIZE))
    norm = (resized.astype(np.float32) / 255.0 - IMAGENET_MEAN) / IMAGENET_STD
    tensor = torch.from_numpy(norm.transpose(2, 0, 1)).float().unsqueeze(0).to(device)

    with torch.no_grad():
        heatmap_logits = model(tensor)[0]  # drop batch dim -> (C, H, W)

    points_resized_space = decode_heatmaps(heatmap_logits)

    # confidence check (CropFollow++ Eq. 25), per channel -- see
    # src/confidence.py docstring for caveats and calibration guidance.
    confidences = {}
    for ch, class_name in enumerate(CLASS_NAMES):
        conf, sigma_vp, is_outlier = keypoint_confidence(heatmap_logits[ch])
        confidences[class_name] = {"confidence": conf, "sigma_vp": sigma_vp, "is_outlier": is_outlier}

    scale_x = orig_w / INPUT_SIZE
    scale_y = orig_h / INPUT_SIZE
    points_original_space = {}
    for class_name, pts in points_resized_space.items():
        points_original_space[class_name] = [(x * scale_x, y * scale_y) for (x, y) in pts]

    return image_bgr, points_original_space, confidences


def get_camera_params(image_width, image_height):
    """Resolves the calibration constants from config.py, applying the
    documented placeholder fallbacks and returning whether a placeholder
    was used (so callers can warn about it). See config.py's comments for
    exactly what to replace each of these with on a real robot."""
    used_placeholder_focal_length = FOCAL_LENGTH_PX is None
    focal_length_px = FOCAL_LENGTH_PX if FOCAL_LENGTH_PX is not None else image_width
    principal_point = PRINCIPAL_POINT if PRINCIPAL_POINT is not None else (image_width / 2, image_height / 2)
    roll_rad = math.radians(CAMERA_ROLL_DEG)
    return focal_length_px, principal_point, roll_rad, used_placeholder_focal_length


def draw_row_lines(image_bgr, points):
    """vanishing_point -> left_intercept (green),
    vanishing_point_right -> right_intercept (blue), red markers at all
    four points. Purely a visualization of the raw detections -- no
    calibration involved."""
    out = image_bgr.copy()
    vp_left = points.get("vanishing_point", [])
    vp_right = points.get("vanishing_point_right", [])
    left = points.get("left_intercept", [])
    right = points.get("right_intercept", [])

    if vp_left and left:
        p1 = (int(vp_left[0][0]), int(vp_left[0][1]))
        p2 = (int(left[0][0]), int(left[0][1]))
        cv2.line(out, p1, p2, COLOR_LEFT_LINE, 2, cv2.LINE_AA)
    if vp_right and right:
        p1 = (int(vp_right[0][0]), int(vp_right[0][1]))
        p2 = (int(right[0][0]), int(right[0][1]))
        cv2.line(out, p1, p2, COLOR_RIGHT_LINE, 2, cv2.LINE_AA)

    for pts in (vp_left, vp_right, left, right):
        for (x, y) in pts:
            cv2.circle(out, (int(x), int(y)), 4, COLOR_POINT, -1)

    return out


def draw_state_text(image_bgr, calibrated_state):
    """Overlays the calibrated pipeline's output (Eq. 22-24) on the image.
    Text values are computed via the camera_geometry homography pipeline,
    not raw pixel measurements -- see console output / camera_geometry.py
    for the full breakdown."""
    out = image_bgr.copy()
    if "error" in calibrated_state:
        text = f"state unavailable: {calibrated_state['error']}"
        cv2.putText(out, text, (8, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                    (0, 0, 255), 1, cv2.LINE_AA)
        return out

    lines = [
        f"pitch: {calibrated_state['pitch_deg']:+.1f} deg",
        f"heading: {calibrated_state['heading_deg']:+.1f} deg",
    ]
    if "distance_left_m" in calibrated_state:
        lines.append(f"d_L: {calibrated_state['distance_left_m']:.2f} m")
        lines.append(f"d_R: {calibrated_state['distance_right_m']:.2f} m")
    else:
        lines.append(f"d ratio: {calibrated_state['d_ratio']:.3f} (set ROW_SPACING_M for meters)")

    for i, line in enumerate(lines):
        y = 18 + i * 18
        cv2.putText(out, line, (8, y), cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                    (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(out, line, (8, y), cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                    (0, 0, 0), 1, cv2.LINE_AA)
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("image_path")
    parser.add_argument("--checkpoint", default="checkpoints/best.pt")
    args = parser.parse_args()

    device = DEVICE if torch.cuda.is_available() else "cpu"
    model = load_model(args.checkpoint, device)

    image_bgr, points, confidences = infer_image(model, args.image_path, device)

    print("Decoded keypoints (original image pixel coordinates) and confidence:")
    for class_name in CLASS_NAMES:
        c = confidences[class_name]
        flag = " <-- OUTLIER (below 0.5 threshold)" if c["is_outlier"] else ""
        print(f"  {class_name}: {points[class_name]}  "
              f"confidence={c['confidence']:.4f}  sigma_vp={c['sigma_vp']:.3e}{flag}")

    h, w = image_bgr.shape[:2]
    focal_length_px, principal_point, roll_rad, used_placeholder = get_camera_params(w, h)
    if used_placeholder:
        print(f"\n[WARNING] FOCAL_LENGTH_PX not set in config.py -- using "
              f"image width ({w}px) as a rough placeholder. This is NOT a "
              f"real calibration; see config.py's comment on FOCAL_LENGTH_PX "
              f"for what to do on a real robot.")

    calibrated_state = estimate_state_with_steps(
        points, focal_length_px, principal_point, roll_rad, ROW_SPACING_M,
        image_height=h, image_width=w
    )

    print("\nRobot state (CropFollow++ Eq. 22-24 pipeline):")
    if "error" in calibrated_state:
        print(f"  {calibrated_state['error']}")
    else:
        print(f"  pitch angle:   {calibrated_state['pitch_deg']:+.2f} deg")
        print(f"  heading angle: {calibrated_state['heading_deg']:+.2f} deg "
              f"(positive = tilted toward right row)")
        print(f"  distance ratio d = d_L/(d_L+d_R): {calibrated_state['d_ratio']:.4f}")
        if "distance_left_m" in calibrated_state:
            print(f"  distance to left row:  {calibrated_state['distance_left_m']:.2f} m")
            print(f"  distance to right row: {calibrated_state['distance_right_m']:.2f} m")
        else:
            print(f"  (ROW_SPACING_M is not set in config.py -- set it to your "
                  f"field's real row spacing to get metric distances)")

    annotated = draw_row_lines(image_bgr, points)
    annotated = draw_state_text(annotated, calibrated_state)
    out_path = os.path.splitext(args.image_path)[0] + "_keypoints.png"
    cv2.imwrite(out_path, annotated)
    print(f"\nSaved annotated image to {out_path}")

    if "error" not in calibrated_state:
        steps_path = os.path.splitext(args.image_path)[0] + "_steps.png"
        make_step_figure(image_bgr, calibrated_state, principal_point, steps_path)
        print(f"Saved step-by-step derivation figure to {steps_path}")
    else:
        print("Skipped step-by-step figure (calibrated state unavailable).")


if __name__ == "__main__":
    main()
