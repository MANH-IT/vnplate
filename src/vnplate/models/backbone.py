from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models


class ResNetBackbone(nn.Module):
    def __init__(self, name: str = "resnet18", pretrained: bool = True, out_channels: int = 256, freeze_bn: bool = False):
        super().__init__()
        weights = "DEFAULT" if pretrained else None
        try:
            if name == "resnet18":
                net = models.resnet18(weights=weights)
            elif name == "resnet34":
                net = models.resnet34(weights=weights)
            elif name == "resnet50":
                net = models.resnet50(weights=weights)
            else:
                raise ValueError(f"unsupported backbone '{name}'")
        except Exception:
            if not pretrained:
                raise
            net = getattr(models, name)(weights=None)

        self.stem = nn.Sequential(net.conv1, net.bn1, net.relu, net.maxpool)
        self.layer1 = net.layer1
        self.layer2 = net.layer2
        self.layer3 = net.layer3
        self.layer4 = net.layer4
        self.num_channels = 512 if name in ("resnet18", "resnet34") else 2048
        self.proj = nn.Conv2d(self.num_channels, out_channels, kernel_size=1)
        self.freeze_bn = freeze_bn

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.freeze_bn:
            for module in self.modules():
                if isinstance(module, nn.BatchNorm2d):
                    module.eval()
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        return self.proj(x)


class TinyBackbone(nn.Module):
    def __init__(self, out_channels: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            _block(3, 32),
            _block(32, 64),
            _block(64, 128),
            _block(128, out_channels),
        )

    def forward(self, x):
        return self.net(x)


def _block(cin, cout):
    return nn.Sequential(
        nn.Conv2d(cin, cout, 3, stride=2, padding=1, bias=False),
        nn.BatchNorm2d(cout),
        nn.ReLU(inplace=True),
        nn.Conv2d(cout, cout, 3, padding=1, bias=False),
        nn.BatchNorm2d(cout),
        nn.ReLU(inplace=True),
    )


def build_backbone(name: str = "resnet18", pretrained: bool = True, out_channels: int = 256) -> nn.Module:
    if name == "tiny":
        return TinyBackbone(out_channels)
    return ResNetBackbone(name=name, pretrained=pretrained, out_channels=out_channels)
