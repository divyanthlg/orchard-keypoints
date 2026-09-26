import torch
import torch.nn as nn
import torch.nn.functional as F

from src.config import CLASS_NAMES, SINGLE_INSTANCE_CLASSES


class SpatialSoftmaxKLLoss(nn.Module):
    """KL-divergence loss on spatially-softmax-normalized heatmaps, exactly
    as used for the 3 core keypoints in CropFollow++ (Ch.4): the predicted
    heatmap logits are normalized into a probability distribution over all
    pixels (spatial softmax), and compared against a target Gaussian that's
    also normalized to sum to 1, via KL divergence.

    Only meaningful for SINGLE-INSTANCE channels -- a spatial softmax forces
    all probability mass into one distribution, so it cannot represent
    multiple simultaneous peaks (see HeatmapFocalLoss for the multi-instance
    trunk channels instead).
    """

    def forward(self, pred_logits_channel, target_heatmap_channel):
        # pred_logits_channel, target_heatmap_channel: (B, H, W)
        b = pred_logits_channel.shape[0]
        pred_flat = pred_logits_channel.reshape(b, -1)
        target_flat = target_heatmap_channel.reshape(b, -1)

        pred_log_prob = F.log_softmax(pred_flat, dim=1)
        target_prob = target_flat / (target_flat.sum(dim=1, keepdim=True) + 1e-8)

        # F.kl_div expects (log-probabilities, probabilities), reduction 'batchmean'
        return F.kl_div(pred_log_prob, target_prob, reduction="batchmean")


class HeatmapFocalLoss(nn.Module):
    """Modified focal loss from CornerNet/CenterNet, used here only for the
    multi-instance trunk channels (not covered by the paper). Ground-truth
    heatmaps contain soft gaussian values in [0, 1], 1 exactly at annotated
    point centers."""

    def __init__(self, alpha=2.0, beta=4.0):
        super().__init__()
        self.alpha = alpha
        self.beta = beta

    def forward(self, pred_logits, target_heatmap):
        pred = torch.sigmoid(pred_logits).clamp(min=1e-4, max=1 - 1e-4)

        pos_mask = target_heatmap.eq(1).float()
        neg_mask = target_heatmap.lt(1).float()
        neg_weights = torch.pow(1 - target_heatmap, self.beta)

        pos_loss = torch.log(pred) * torch.pow(1 - pred, self.alpha) * pos_mask
        neg_loss = (
            torch.log(1 - pred) * torch.pow(pred, self.alpha) * neg_weights * neg_mask
        )

        num_pos = pos_mask.sum()
        pos_loss = pos_loss.sum()
        neg_loss = neg_loss.sum()

        if num_pos == 0:
            return -neg_loss
        return -(pos_loss + neg_loss) / num_pos


class CombinedKeypointLoss(nn.Module):
    """Splits the 5 output channels by type:
      - single-instance channels (vanishing_point, left_intercept,
        right_intercept) -> SpatialSoftmaxKLLoss, matching CropFollow++ Ch.4
      - multi-instance channels (left_trunk, right_trunk) -> HeatmapFocalLoss

    Returns the sum, weighted equally by default. If the trunk loss
    dominates or lags too much during training, adjust trunk_weight.
    """

    def __init__(self, trunk_weight=1.0):
        super().__init__()
        self.kl_loss = SpatialSoftmaxKLLoss()
        self.focal_loss = HeatmapFocalLoss()
        self.trunk_weight = trunk_weight

        self.single_idx = [i for i, n in enumerate(CLASS_NAMES) if n in SINGLE_INSTANCE_CLASSES]
        self.multi_idx = [i for i, n in enumerate(CLASS_NAMES) if n not in SINGLE_INSTANCE_CLASSES]

    def forward(self, pred_logits, target_heatmaps):
        # pred_logits, target_heatmaps: (B, C, H, W)
        loss = 0.0
        loss_breakdown = {}

        for ch in self.single_idx:
            l = self.kl_loss(pred_logits[:, ch], target_heatmaps[:, ch])
            loss = loss + l
            loss_breakdown[CLASS_NAMES[ch]] = l.item()

        if self.multi_idx:
            multi_pred = pred_logits[:, self.multi_idx]
            multi_target = target_heatmaps[:, self.multi_idx]
            l = self.focal_loss(multi_pred, multi_target) * self.trunk_weight
            loss = loss + l
            loss_breakdown["trunk_channels_combined"] = l.item()

        return loss, loss_breakdown
