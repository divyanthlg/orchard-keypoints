"""
Builds a Figure-16-style, multi-panel step-by-step image showing how the
robot state was derived from the 4 detected keypoints: raw detections ->
roll correction -> pitch correction -> heading correction -> distance
ratio at the bottom row. Uses src/camera_geometry.py's actual homography
pipeline (Eq. 22-24).

Visual style matches CropFollow++ Fig. 16: PP (principal point) marked,
AVP (the intersection of the two detected row lines) marked, dashed offset
brackets showing the y-offset (drives pitch) and x-offset (drives heading)
used at each stage, and cyan vertical lines marking dL/dR at the bottom
row in the final panel.
"""
import cv2
import matplotlib.pyplot as plt
import numpy as np

COLOR_LEFT_LINE = (0, 200, 0)     # BGR
COLOR_RIGHT_LINE = (255, 0, 0)
COLOR_POINT = (0, 0, 255)
COLOR_PP = (0, 0, 255)
COLOR_BRACKET = (255, 255, 255)
COLOR_DIST_LINE = (255, 255, 0)   # cyan in BGR


def _clip(pt, w, h):
    return (int(np.clip(pt[0], -w, 2 * w)), int(np.clip(pt[1], -h, 2 * h)))


def _draw_overlay(image_bgr, pts):
    """pts: dict with vp_left, vp_right, left, right (x,y) tuples."""
    out = image_bgr.copy()
    h, w = out.shape[:2]

    cv2.line(out, _clip(pts["vp_left"], w, h), _clip(pts["left"], w, h),
              COLOR_LEFT_LINE, 2, cv2.LINE_AA)
    cv2.line(out, _clip(pts["vp_right"], w, h), _clip(pts["right"], w, h),
              COLOR_RIGHT_LINE, 2, cv2.LINE_AA)
    for key in ("vp_left", "vp_right", "left", "right"):
        p = _clip(pts[key], w, h)
        if 0 <= p[0] < w and 0 <= p[1] < h:
            cv2.circle(out, p, 5, COLOR_POINT, -1)
    return out


def _draw_pp_marker(image_bgr, principal_point):
    out = image_bgr.copy()
    pp = (int(principal_point[0]), int(principal_point[1]))
    cv2.drawMarker(out, pp, COLOR_PP, cv2.MARKER_CROSS, 14, 2)
    cv2.putText(out, "PP", (pp[0] + 8, pp[1] - 6), cv2.FONT_HERSHEY_SIMPLEX,
                0.5, COLOR_PP, 1, cv2.LINE_AA)
    return out


def _draw_avp_marker(image_bgr, avp):
    """Marks AVP -- the geometrically-derived intersection of the left and
    right row lines -- which is what actually drives the pitch/heading
    calculation (see camera_geometry.py)."""
    out = image_bgr.copy()
    p = (int(avp[0]), int(avp[1]))
    h, w = out.shape[:2]
    if 0 <= p[0] < w and 0 <= p[1] < h:
        cv2.drawMarker(out, p, (0, 255, 255), cv2.MARKER_DIAMOND, 10, 2)
    return out


def _draw_offset_bracket(image_bgr, vp, principal_point, axis):
    """Dashed bracket showing the y-offset (axis='y', drives pitch) or
    x-offset (axis='x', drives heading) between vp and the principal point,
    matching Fig. 16's callout style."""
    out = image_bgr.copy()
    pp = (int(principal_point[0]), int(principal_point[1]))
    vp_pt = (int(vp[0]), int(vp[1]))

    def dashed_line(p1, p2, color, dash_len=6):
        p1, p2 = np.array(p1, dtype=float), np.array(p2, dtype=float)
        dist = np.linalg.norm(p2 - p1)
        if dist < 1e-3:
            return
        n_dashes = max(1, int(dist // (dash_len * 2)))
        for i in range(n_dashes):
            s = p1 + (p2 - p1) * (i * 2 * dash_len) / dist
            e = p1 + (p2 - p1) * min((i * 2 + 1) * dash_len / dist, 1.0)
            cv2.line(out, tuple(s.astype(int)), tuple(e.astype(int)), color, 1, cv2.LINE_AA)

    if axis == "y":
        corner = (pp[0], vp_pt[1])
        dashed_line(vp_pt, corner, COLOR_BRACKET)
        dashed_line(corner, pp, COLOR_BRACKET)
        cv2.putText(out, "y", (corner[0] - 18, (corner[1] + vp_pt[1]) // 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, COLOR_BRACKET, 1, cv2.LINE_AA)
    else:  # 'x'
        corner = (vp_pt[0], pp[1])
        dashed_line(vp_pt, corner, COLOR_BRACKET)
        dashed_line(corner, pp, COLOR_BRACKET)
        cv2.putText(out, "x", ((corner[0] + vp_pt[0]) // 2, corner[1] - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, COLOR_BRACKET, 1, cv2.LINE_AA)

    return _draw_pp_marker(out, principal_point)


def make_step_figure(image_bgr, state, principal_point, save_path):
    """image_bgr: the original input image.
    state: the dict returned by camera_geometry.estimate_state_with_steps().
    Saves a 5-panel figure to save_path, styled to match CropFollow++ Fig. 16."""
    if "error" in state:
        raise ValueError(f"Cannot visualize -- {state['error']}")

    h, w = image_bgr.shape[:2]

    H_roll = state["H_roll"]
    H_pitch = state["H_pitch"]
    H_heading = state["H_heading"]

    H_cum_roll = H_roll
    H_cum_pitch = H_pitch @ H_roll
    H_cum_heading = H_heading @ H_pitch @ H_roll

    img_raw = image_bgr
    img_roll = cv2.warpPerspective(image_bgr, H_cum_roll, (w, h))
    img_pitch = cv2.warpPerspective(image_bgr, H_cum_pitch, (w, h))
    img_heading = cv2.warpPerspective(image_bgr, H_cum_heading, (w, h))

    panel1 = _draw_overlay(img_raw, state["pts_raw"])
    panel1 = _draw_pp_marker(panel1, principal_point)
    panel1 = _draw_avp_marker(panel1, state["pts_raw"]["avp"])

    # panel 2 shows the y-offset bracket (drives the NEXT step's pitch calc)
    panel2 = _draw_overlay(img_roll, state["pts_after_roll"])
    panel2 = _draw_offset_bracket(panel2, state["pts_after_roll"]["avp"], principal_point, axis="y")
    panel2 = _draw_avp_marker(panel2, state["pts_after_roll"]["avp"])

    # panel 3 shows the x-offset bracket (drives the NEXT step's heading calc)
    panel3 = _draw_overlay(img_pitch, state["pts_after_pitch"])
    panel3 = _draw_offset_bracket(panel3, state["pts_after_pitch"]["avp"], principal_point, axis="x")
    panel3 = _draw_avp_marker(panel3, state["pts_after_pitch"]["avp"])

    panel4 = _draw_overlay(img_heading, state["pts_after_heading"])
    panel4 = _draw_pp_marker(panel4, principal_point)

    # panel 5: final corrected view with cyan dL/dR lines at the bottom row
    panel5 = panel4.copy()
    bottom_y = int(state["bottom_y"])
    cv2.line(panel5, (0, bottom_y), (w, bottom_y), (200, 200, 200), 1, cv2.LINE_AA)
    cx = int(principal_point[0])
    xl, xr = int(state["x_left_bottom"]), int(state["x_right_bottom"])
    for x in (xl, xr):
        if 0 <= x < w:
            cv2.line(panel5, (x, max(0, bottom_y - 40)), (x, min(h - 1, bottom_y + 10)),
                      COLOR_DIST_LINE, 2, cv2.LINE_AA)
    if 0 <= cx < w:
        cv2.line(panel5, (cx, max(0, bottom_y - 40)), (cx, min(h - 1, bottom_y + 10)),
                  (0, 255, 255), 1, cv2.LINE_AA)

    panels = [
        (panel1, "1. Raw detections"),
        (panel2, f"2. After roll correction\n(roll={state['roll_deg']:.1f} deg)"),
        (panel3, f"3. After pitch correction\n(theta_pitch={state['pitch_deg']:.1f} deg)"),
        (panel4, f"4. After heading correction\n(theta_heading={state['heading_deg']:+.1f} deg)"),
        (panel5, f"5. Distance ratio at bottom row\nd={state['d_ratio']:.3f}"
                 + (f"  |  L={state.get('distance_left_m', float('nan')):.2f}m "
                    f"R={state.get('distance_right_m', float('nan')):.2f}m"
                    if "distance_left_m" in state else "")),
    ]

    fig, axes = plt.subplots(1, 5, figsize=(22, 5))
    for ax, (panel, title) in zip(axes, panels):
        ax.imshow(cv2.cvtColor(panel, cv2.COLOR_BGR2RGB))
        ax.set_title(title, fontsize=10)
        ax.axis("off")
    plt.tight_layout()
    plt.savefig(save_path, dpi=130)
    plt.close(fig)
    return save_path
