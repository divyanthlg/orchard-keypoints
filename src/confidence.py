"""
Confidence check from CropFollow++ (Ch.4, Eq. 25):

    confidence = sigmoid( 3 * tanh( 5 * (1 - eps / (5 * sigma_vp)) ) )

where sigma_vp is the variance of the (spatial-softmax-normalized)
vanishing point heatmap, and eps = 1e-5. Threshold at 0.5: confidence < 0.5
is classified as an outlier / untrustworthy prediction.

NOTE ON TRANSCRIPTION: this formula was read off a rendered PDF page of a
thesis, from a small printed fraction. I'm confident about the overall
shape (sigmoid of a scaled tanh of a variance-based term), but equations
transcribed this way carry some risk of a numerator/denominator mixup.
Before trusting this in your pipeline, sanity-check it empirically per the
instructions printed by plot_confidence_curve() below: run it on one
clearly-good (fully visible row) frame and one clearly-bad (heavily
occluded / off-row) frame, and confirm confidence comes out HIGH for the
good one and LOW for the bad one. If it's backwards, invert the fraction
(sigma_vp / (5 * eps) in place of eps / (5 * sigma_vp)) -- everything else
in this module still works unchanged.
"""
import numpy as np
import torch

EPSILON = 1e-5


def heatmap_variance(logits_channel):
    """logits_channel: (H, W) RAW logits for one heatmap channel.
    Computes the spatial variance of the softmax-normalized distribution:
    sigma_vp = Var(x) + Var(y), the total spread of probability mass
    around its own expected location. This is my best-effort reading of
    "the variance of the vanishing point heatmap" -- the thesis doesn't
    spell out whether it's summed, averaged, or a covariance determinant,
    so summed per-axis variance (a single scalar, larger = more spread out)
    is the simplest faithful choice.
    """
    h, w = logits_channel.shape
    probs = torch.softmax(logits_channel.flatten(), dim=0)

    ys = torch.arange(h, device=logits_channel.device).float()
    xs = torch.arange(w, device=logits_channel.device).float()
    grid_y, grid_x = torch.meshgrid(ys, xs, indexing="ij")
    grid_x = grid_x.flatten()
    grid_y = grid_y.flatten()

    mean_x = (probs * grid_x).sum()
    mean_y = (probs * grid_y).sum()
    var_x = (probs * (grid_x - mean_x) ** 2).sum()
    var_y = (probs * (grid_y - mean_y) ** 2).sum()

    return (var_x + var_y).item()


def confidence_from_variance(sigma_vp, epsilon=EPSILON):
    """Eq. 25 from CropFollow++ Ch.4, as printed."""
    inner = 5.0 * (1.0 - epsilon / (5.0 * sigma_vp))
    return 1.0 / (1.0 + np.exp(-3.0 * np.tanh(inner)))


def keypoint_confidence(logits_channel, epsilon=EPSILON):
    """Convenience wrapper: raw logits -> (confidence, sigma_vp, is_outlier)."""
    sigma_vp = heatmap_variance(logits_channel)
    conf = confidence_from_variance(sigma_vp, epsilon)
    return conf, sigma_vp, conf < 0.5


def heatmap_covariance_2x2(logits_channel):
    """logits_channel: (H, W) RAW logits for one heatmap channel.
    Returns the full 2x2 spatial covariance matrix [[Var(x), Cov(x,y)],
    [Cov(x,y), Var(y)]] of the softmax-normalized distribution, in
    heatmap-pixel units. This is the quantity you actually want for EKF
    fusion -- propagate it through your pixel-to-state Jacobian to get a
    proper measurement noise covariance R_vision, rather than collapsing
    it into a single confidence scalar. See module docstring / chat
    discussion for why a bounded [0,1] score is the wrong currency for a
    Kalman filter update.
    """
    h, w = logits_channel.shape
    probs = torch.softmax(logits_channel.flatten(), dim=0)

    ys = torch.arange(h, device=logits_channel.device).float()
    xs = torch.arange(w, device=logits_channel.device).float()
    grid_y, grid_x = torch.meshgrid(ys, xs, indexing="ij")
    grid_x = grid_x.flatten()
    grid_y = grid_y.flatten()

    mean_x = (probs * grid_x).sum()
    mean_y = (probs * grid_y).sum()
    dx = grid_x - mean_x
    dy = grid_y - mean_y

    var_x = (probs * dx ** 2).sum()
    var_y = (probs * dy ** 2).sum()
    cov_xy = (probs * dx * dy).sum()

    cov = torch.stack([
        torch.stack([var_x, cov_xy]),
        torch.stack([cov_xy, var_y]),
    ])
    return cov  # (2, 2) tensor, in heatmap-pixel^2 units


def calibrated_confidence(sigma_vp, sigma_ref, power=2.0):
    """A smoothly-graded, data-calibrated alternative to Eq. 25 for cases
    where you want a single bounded [0,1] score (e.g. logging, gating,
    or a soft blend weight in a non-covariance-aware fusion scheme) --
    NOT as a substitute for feeding real covariance into an EKF's R.

    sigma_ref: the reference "typical good detection" variance, estimated
    empirically as the MEDIAN sigma_vp over a validation set of visually
    confirmed good detections. This replaces Eq. 25's fixed epsilon=1e-5
    with a number actually derived from your model's real output scale.
    power: steepness of the transition around sigma_ref (try 2.0).

    confidence -> 1 as sigma_vp -> 0 (sharper than typical)
    confidence = 0.5 exactly at sigma_vp == sigma_ref
    confidence -> 0 as sigma_vp -> inf (more spread than typical)
    """
    return 1.0 / (1.0 + (sigma_vp / sigma_ref) ** power)


def calibrate_sigma_ref(list_of_sigma_vp_from_good_detections):
    """Run this once, offline, on sigma_vp values you've collected from a
    batch of visually-confirmed GOOD detections (clear row, no occlusion).
    Returns the sigma_ref to plug into calibrated_confidence()."""
    import statistics
    return statistics.median(list_of_sigma_vp_from_good_detections)
    """Illustrates the shape of Eq. 25 across a range of heatmap variances,
    so you can see where the 0.5 threshold actually falls before trusting
    it on real predictions."""
    import matplotlib.pyplot as plt

    variances = np.logspace(-7, 2, 500)  # 1e-7 to 100, log-spaced
    confidences = [confidence_from_variance(v) for v in variances]

    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(variances, confidences, color="#3C3489", linewidth=2)
    ax.axhline(0.5, color="#D85A30", linestyle="--", linewidth=1, label="0.5 threshold")
    ax.set_xscale("log")
    ax.set_xlabel("sigma_vp (heatmap variance)")
    ax.set_ylabel("confidence")
    ax.set_title("CropFollow++ Eq. 25: confidence vs. vanishing-point heatmap variance")
    ax.legend()
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=120)
    plt.close(fig)
    print(f"Saved {save_path}")
    print("Look at where the curve crosses 0.5 on the x-axis (sigma_vp value) "
          "-- that's the actual variance threshold this formula enforces. "
          "Compare that number against sigma_vp values you observe on real "
          "good vs. bad predictions with debug_heatmaps.py / this module, "
          "to confirm the formula is behaving the way you expect.")


if __name__ == "__main__":
    plot_confidence_curve()
