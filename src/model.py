import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

from src.config import NUM_CLASSES


class UpBlock(nn.Module):
    """Bilinear upsample + concat with encoder skip connection + conv layers.
    This matches CropFollow++ Ch.4: 'decoder with bilinear upsampling and
    convolutional layers' in a U-Net-like architecture (skip connections
    from the ResNet18 encoder stages)."""

    def __init__(self, in_ch, skip_ch, out_ch):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch + skip_ch, out_ch, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_ch)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x, skip):
        x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        x = torch.cat([x, skip], dim=1)
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.bn2(self.conv2(x)))
        return x


class KeypointHeatmapNet(nn.Module):
    """ResNet18 encoder (ImageNet pretrained) + U-Net-style decoder with
    bilinear upsampling and skip connections, per CropFollow++ (Ch.4).

    Encoder strides: stem+layer1 -> /4 (64ch), layer2 -> /8 (128ch),
    layer3 -> /16 (256ch), layer4 -> /32 (512ch).
    Decoder upsamples back to /4 (matching OUTPUT_STRIDE), fusing skip
    connections at each stage. Output: NUM_CLASSES heatmap channels at /4
    resolution, exactly as in the paper (1/4 the input image dimensions).
    """

    def __init__(self, num_classes=NUM_CLASSES, pretrained=True):
        super().__init__()
        backbone = models.resnet18(
            weights=models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
        )
        self.stem = nn.Sequential(
            backbone.conv1, backbone.bn1, backbone.relu, backbone.maxpool
        )
        self.layer1 = backbone.layer1  # -> 64ch,  stride 4
        self.layer2 = backbone.layer2  # -> 128ch, stride 8
        self.layer3 = backbone.layer3  # -> 256ch, stride 16
        self.layer4 = backbone.layer4  # -> 512ch, stride 32

        self.up1 = UpBlock(in_ch=512, skip_ch=256, out_ch=256)  # /32 -> /16
        self.up2 = UpBlock(in_ch=256, skip_ch=128, out_ch=128)  # /16 -> /8
        self.up3 = UpBlock(in_ch=128, skip_ch=64, out_ch=64)    # /8  -> /4

        self.heatmap_head = nn.Conv2d(64, num_classes, kernel_size=1)
        nn.init.constant_(self.heatmap_head.bias, 0.0)

    def forward(self, x):
        s1 = self.stem(x)
        f1 = self.layer1(s1)   # /4,  64ch  -- skip
        f2 = self.layer2(f1)   # /8,  128ch -- skip
        f3 = self.layer3(f2)   # /16, 256ch -- skip
        f4 = self.layer4(f3)   # /32, 512ch -- bottleneck

        d = self.up1(f4, f3)   # -> /16
        d = self.up2(d, f2)    # -> /8
        d = self.up3(d, f1)    # -> /4

        heatmaps = self.heatmap_head(d)
        return heatmaps  # raw logits, shape (B, num_classes, H/4, W/4)


if __name__ == "__main__":
    model = KeypointHeatmapNet()
    dummy = torch.randn(2, 3, 512, 512)
    out = model(dummy)
    print("Output shape:", out.shape)  # expect (2, 4, 128, 128)
