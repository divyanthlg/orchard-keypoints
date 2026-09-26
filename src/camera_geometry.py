"""
Implements the CropFollow++ Ch.4 pipeline (Eq. 22-24) with a refinement for
using two independently-detected vanishing points (left + right row lines,
rather than assuming they share one annotated point):

  1. Fit LLS = the line through left_intercept (LI) and vanishing_point (LV).
     Fit RLS = the line through right_intercept (RI) and vanishing_point_right (RV).
  2. AVP ("actual vanishing point") = intersection of LLS and RLS. This is
     a geometrically-derived single vanishing point (unlike just averaging
     the two annotated VP positions) -- it's exactly where the two detected
     row lines would truly converge, which is what a single physical camera
     orientation implies (pitch/heading are properties of the ONE real
     camera, not one-per-row-line).
  3. ALVP / ARVP ("actual left/right vanishing point" -- despite the name,
     these are boundary-crossing points, not vanishing points at infinity):
     starting AT AVP and travelling in the direction of LI (resp. RI) and
     CONTINUING PAST IT, find where that ray exits the image rectangle.
     Only this direction is searched (never back past AVP), so the top
     edge can never be the exit -- by construction, AVP sits "above" LI/RI
     in the image, so continuing away from AVP only ever moves toward the
     left, right, or bottom edges.
  4. Validity check: LLS's exit must be the LEFT or BOTTOM edge; RLS's exit
     must be the RIGHT or BOTTOM edge. Any other outcome (including a
     degenerate near-parallel-lines case where AVP itself is undefined)
     means the frame is flagged "[could not recognize]" and is not used
     for state estimation.
  5. AVP (from the RAW, uncorrected detections) then drives the SAME
     roll -> pitch -> heading homography correction sequence as before.
     ALVP/ARVP are used ONLY for the validity check in step 4 above --
     the actual distance ratio is computed by projecting the (now fully
     corrected) LLS/RLS lines onto the bottom row, same as the original
     Eq. 24 mechanics, so both row lines are compared at the same image
     depth.

REQUIRES CAMERA CALIBRATION to be meaningful:
  - focal_length_px: focal length in PIXELS (not mm). If you don't have a
    proper calibration yet, a rough placeholder is image_width_px (this
    corresponds to roughly a 53-degree horizontal field of view -- common
    but not necessarily YOUR camera). Get a real value via a checkerboard
    calibration (OpenCV's cv2.calibrateCamera) as soon as you can; every
    angle computed here scales with how correct this number is.
  - principal_point: usually close to (image_width/2, image_height/2)
    unless your calibration says otherwise.
  - roll_rad: from your IMU. Defaults to 0.0 (no IMU wired in yet) --
    the roll-correction step is then a no-op.
  - row_spacing_m: the known real-world perpendicular spacing between crop
    rows. Required to convert the dimensionless distance ratio into an
    actual metric distance (Eq. 24's final step).

Sign conventions for the roll/pitch/heading corrections were verified
numerically (see chat): a round-trip test simulating a pitched + yawed
camera and running it through this exact correction sequence returns the
vanishing point to the principal point.
"""
import math

import numpy as np


def K_matrix(f, cx, cy):
    return np.array([[f, 0, cx], [0, f, cy], [0, 0, 1]], dtype=float)


def Rx(theta):
    c, s = math.cos(theta), math.sin(theta)
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])


def Ry(theta):
    c, s = math.cos(theta), math.sin(theta)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])


def Rz(theta):
    c, s = math.cos(theta), math.sin(theta)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])


def homography_from_rotation(K, R):
    return K @ R @ np.linalg.inv(K)


def apply_homography(H, points):
    """points: list of (x, y). Returns transformed list of (x, y)."""
    out = []
    for (x, y) in points:
        v = H @ np.array([x, y, 1.0])
        out.append((v[0] / v[2], v[1] / v[2]))
    return out


def line_intersection(p1, p2, p3, p4):
    """Intersection of the infinite line through p1,p2 and the infinite
    line through p3,p4. Returns (x, y), or None if the lines are parallel
    (or numerically indistinguishable from parallel)."""
    x1, y1 = p1
    x2, y2 = p2
    x3, y3 = p3
    x4, y4 = p4
    denom = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
    if abs(denom) < 1e-9:
        return None
    px = ((x1 * y2 - y1 * x2) * (x3 - x4) - (x1 - x2) * (x3 * y4 - y3 * x4)) / denom
    py = ((x1 * y2 - y1 * x2) * (y3 - y4) - (y1 - y2) * (x3 * y4 - y3 * x4)) / denom
    return (px, py)


def ray_box_exit(origin, through_point, image_width, image_height):
    """Ray starting at `origin`, travelling in the direction of
    `through_point` and CONTINUING PAST IT (never searching the opposite
    direction, back through origin). Finds where this ray exits the
    image rectangle [0, image_width-1] x [0, image_height-1].

    Returns ((exit_x, exit_y), edge_name) where edge_name is one of
    'left', 'right', 'top', 'bottom', or (None, None) if no valid forward
    exit exists (e.g. the ray never re-enters the box, or the direction
    is degenerate)."""
    ax, ay = origin
    dx, dy = through_point[0] - ax, through_point[1] - ay

    def slab(a, d, lo, hi):
        if abs(d) < 1e-12:
            return (-math.inf, math.inf) if lo <= a <= hi else None
        t1, t2 = (lo - a) / d, (hi - a) / d
        return (min(t1, t2), max(t1, t2))

    sx = slab(ax, dx, 0, image_width - 1)
    sy = slab(ay, dy, 0, image_height - 1)
    if sx is None or sy is None:
        return None, None

    t_enter, t_exit = max(sx[0], sy[0]), min(sx[1], sy[1])
    if t_enter > t_exit or t_exit <= 0:
        return None, None

    exit_pt = (ax + dx * t_exit, ay + dy * t_exit)

    edge = None
    if abs(t_exit - sx[1]) < 1e-6:
        edge = "right" if dx > 0 else "left"
    if abs(t_exit - sy[1]) < 1e-6 and edge is None:
        edge = "bottom" if dy > 0 else "top"
    return exit_pt, edge


def estimate_state_with_steps(points, focal_length_px, principal_point,
                               roll_rad=0.0, row_spacing_m=None,
                               image_height=None, image_width=None):
    """points: dict with 'vanishing_point' (LV), 'vanishing_point_right'
    (RV), 'left_intercept' (LI), 'right_intercept' (RI), each a list of
    (x, y) (uses the first / only entry). image_width/image_height are
    the RAW input image dimensions (required for the AVP/ALVP/ARVP
    validity check, which runs on the raw, uncorrected detections).

    Returns a dict with every intermediate step (for visualization) plus
    the final heading/distance estimate, or {"error": "[could not
    recognize] ..."} if the AVP/ALVP/ARVP validity check fails.
    """
    required = ["vanishing_point", "vanishing_point_right", "left_intercept", "right_intercept"]
    missing = [k for k in required if not points.get(k)]
    if missing:
        return {"error": f"missing detection(s): {missing}"}
    if image_width is None or image_height is None:
        return {"error": "image_width and image_height are required"}

    LV = points["vanishing_point"][0]
    RV = points["vanishing_point_right"][0]
    LI = points["left_intercept"][0]
    RI = points["right_intercept"][0]

    # --- AVP / ALVP / ARVP, computed on RAW (uncorrected) detections ---
    avp = line_intersection(LI, LV, RI, RV)
    if avp is None:
        return {"error": "[could not recognize] left and right row lines are "
                          "parallel (or nearly so) -- no well-defined vanishing point"}

    alvp, alvp_edge = ray_box_exit(avp, LI, image_width, image_height)
    if alvp is None or alvp_edge not in ("left", "bottom"):
        return {"error": f"[could not recognize] left row line (LLS) exits via "
                          f"the '{alvp_edge}' edge -- expected 'left' or 'bottom'"}

    arvp, arvp_edge = ray_box_exit(avp, RI, image_width, image_height)
    if arvp is None or arvp_edge not in ("right", "bottom"):
        return {"error": f"[could not recognize] right row line (RLS) exits via "
                          f"the '{arvp_edge}' edge -- expected 'right' or 'bottom'"}

    f = focal_length_px
    cx, cy = principal_point
    K = K_matrix(f, cx, cy)

    # AVP is carried through the SAME roll->pitch->heading correction
    # sequence as before, replacing the earlier midpoint-averaging
    # approach with this geometrically-derived intersection point. LV/RV
    # are also carried through purely for visualization; LI/RI/AVP are
    # what the actual angle/distance math needs.
    all_pts_raw = {"vp_left": LV, "vp_right": RV, "left": LI, "right": RI, "avp": avp}

    # --- Step 1: roll correction. Roll is rotation about the camera's
    # optical (Z) axis, which is exactly a 2D pixel rotation about the
    # principal point.
    H_roll = homography_from_rotation(K, Rz(-roll_rad))
    pts_after_roll = {k: apply_homography(H_roll, [v])[0] for k, v in all_pts_raw.items()}

    # --- Step 2: pitch angle from vertical offset of AVP (Eq. 22), then
    # pitch-correction homography (rotation about camera X).
    avp_for_pitch = pts_after_roll["avp"]
    pitch_rad = math.atan((avp_for_pitch[1] - cy) / f)
    H_pitch = homography_from_rotation(K, Rx(pitch_rad))
    pts_after_pitch = {k: apply_homography(H_pitch, [v])[0] for k, v in pts_after_roll.items()}

    # --- Step 3: heading angle from horizontal offset of the
    # pitch-corrected AVP (Eq. 23), then heading-correction homography
    # (rotation about camera Y).
    avp_for_heading = pts_after_pitch["avp"]
    heading_rad = -math.atan((avp_for_heading[0] - cx) / f)
    H_heading = homography_from_rotation(K, Ry(heading_rad))
    pts_after_heading = {k: apply_homography(H_heading, [v])[0] for k, v in pts_after_pitch.items()}

    # --- Step 4: distance ratio (Eq. 24). ALVP/ARVP were only used for the
    # validity check above; the actual ratio compares both row lines at
    # the SAME depth (the bottom row) after full correction, using the
    # (now-transformed) LLS/RLS lines directly -- same mechanics as the
    # original Eq. 24, just driven by AVP instead of a per-side vp.
    bottom_y = image_height - 1
    x_left_bottom = _extrapolate_to_y(pts_after_heading["vp_left"], pts_after_heading["left"], bottom_y)
    x_right_bottom = _extrapolate_to_y(pts_after_heading["vp_right"], pts_after_heading["right"], bottom_y)

    d_L_px = cx - x_left_bottom
    d_R_px = x_right_bottom - cx
    d_ratio = d_L_px / (d_L_px + d_R_px) if (d_L_px + d_R_px) != 0 else None

    result = {
        "pts_raw": all_pts_raw,
        "pts_after_roll": pts_after_roll,
        "pts_after_pitch": pts_after_pitch,
        "pts_after_heading": pts_after_heading,
        "avp": avp,
        "alvp": alvp,
        "arvp": arvp,
        "roll_rad": roll_rad,
        "pitch_rad": pitch_rad,
        "heading_rad": heading_rad,
        "roll_deg": math.degrees(roll_rad),
        "pitch_deg": math.degrees(pitch_rad),
        "heading_deg": math.degrees(heading_rad),
        "H_roll": H_roll,
        "H_pitch": H_pitch,
        "H_heading": H_heading,
        "x_left_bottom": x_left_bottom,
        "x_right_bottom": x_right_bottom,
        "bottom_y": bottom_y,
        "d_L_px": d_L_px,
        "d_R_px": d_R_px,
        "d_ratio": d_ratio,
    }

    if row_spacing_m is not None and d_ratio is not None:
        result["distance_left_m"] = d_ratio * row_spacing_m
        result["distance_right_m"] = (1 - d_ratio) * row_spacing_m

    return result


def _extrapolate_to_y(p1, p2, target_y):
    x1, y1 = p1
    x2, y2 = p2
    if y2 == y1:
        return x1
    t = (target_y - y1) / (y2 - y1)
    return x1 + t * (x2 - x1)
