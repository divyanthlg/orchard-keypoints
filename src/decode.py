import torch
import torch.nn.functional as F

from src.config import CLASS_NAMES, OUTPUT_STRIDE, SINGLE_INSTANCE_CLASSES


def soft_argmax_2d(logits_channel):
    """logits_channel: (H, W) RAW logits (pre-sigmoid/softmax) for a
    single-instance channel. Applies spatial softmax (matching the
    SpatialSoftmaxKLLoss training objective) then takes the expectation
    over pixel coordinates -- the "spatial softargmax" described in
    CropFollow++ Ch.4. Returns (x, y) sub-pixel coordinate in heatmap
    space, plus the peak probability mass (as a rough confidence proxy)."""
    h, w = logits_channel.shape
    flat_logits = logits_channel.flatten()
    probs = torch.softmax(flat_logits, dim=0)

    ys = torch.arange(h, device=logits_channel.device).float()
    xs = torch.arange(w, device=logits_channel.device).float()
    grid_y, grid_x = torch.meshgrid(ys, xs, indexing="ij")

    x = (probs * grid_x.flatten()).sum()
    y = (probs * grid_y.flatten()).sum()
    confidence = probs.max().item() * h * w  # peak-vs-uniform ratio, not a probability
    return x.item(), y.item(), confidence


def nms_peaks(heatmap, kernel=3, threshold=0.3, max_points=30):
    """heatmap: (H, W) tensor of probabilities (post-sigmoid).
    Standard CenterNet-style max-pool NMS: a pixel survives only if it
    equals the max in its local neighborhood. Returns list of (x, y, conf)."""
    heatmap = heatmap.unsqueeze(0).unsqueeze(0)  # (1, 1, H, W)
    pad = (kernel - 1) // 2
    hmax = F.max_pool2d(heatmap, kernel, stride=1, padding=pad)
    keep = (hmax == heatmap).float()
    filtered = heatmap * keep
    filtered = filtered.squeeze(0).squeeze(0)

    ys, xs = torch.where(filtered > threshold)
    confs = filtered[ys, xs]

    # sort by confidence, cap number of points
    order = torch.argsort(confs, descending=True)[:max_points]
    points = [(xs[i].item(), ys[i].item(), confs[i].item()) for i in order]
    return points


def decode_heatmaps(heatmap_logits, single_conf_threshold=1.5, trunk_conf_threshold=0.3):
    """heatmap_logits: (num_classes, H, W) raw model output (pre-activation),
    for a SINGLE image (no batch dim).

    Returns a dict: class_name -> list of (x, y) in *original input image*
    pixel coordinates (i.e. already multiplied by OUTPUT_STRIDE).

    single_conf_threshold is compared against soft_argmax_2d's peak-vs-uniform
    ratio (1.0 = perfectly uniform/no signal; higher = more peaked). Tune by
    running debug_heatmaps.py and looking at real values for your model.
    """
    results = {}

    for ch, class_name in enumerate(CLASS_NAMES):
        channel_logits = heatmap_logits[ch]

        if class_name in SINGLE_INSTANCE_CLASSES:
            x, y, conf = soft_argmax_2d(channel_logits)
            points = [] if conf < single_conf_threshold else [(x, y)]
        else:
            channel_probs = torch.sigmoid(channel_logits)
            peaks = nms_peaks(channel_probs, threshold=trunk_conf_threshold)
            points = [(x, y) for (x, y, conf) in peaks]

        # scale from heatmap space back to input-image pixel space
        points = [(x * OUTPUT_STRIDE, y * OUTPUT_STRIDE) for (x, y) in points]
        results[class_name] = points

    return results
