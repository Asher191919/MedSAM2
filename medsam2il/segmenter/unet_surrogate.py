"""轻量点击条件 U-Net —— MedSAM2 的 CPU 可训练"替身"。

输入 4 通道: [图像, 正点击累计高斯图, 负点击累计高斯图, 框选指示图]
输出: 单通道分割 logit。
它扮演 SAM/MedSAM2 的角色: 策略网络学的"怎么点"与后端解耦,
有 GPU + 权重时用 medsam2_adapter.MedSAM2Segmenter 平滑替换。
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

from medsam2il.segmenter.base import InteractiveSegmenter, box_map, click_map


def conv_block(cin: int, cout: int) -> nn.Module:
    return nn.Sequential(
        nn.Conv2d(cin, cout, 3, padding=1), nn.BatchNorm2d(cout), nn.ReLU(inplace=True),
        nn.Conv2d(cout, cout, 3, padding=1), nn.BatchNorm2d(cout), nn.ReLU(inplace=True))


class ClickUNet(nn.Module):
    def __init__(self, base: int = 16):
        super().__init__()
        b = base
        self.enc1 = conv_block(4, b)
        self.enc2 = conv_block(b, b * 2)
        self.enc3 = conv_block(b * 2, b * 4)
        self.enc4 = conv_block(b * 4, b * 8)
        self.pool = nn.MaxPool2d(2)
        self.mid = conv_block(b * 8, b * 16)
        self.dec4 = conv_block(b * 16 + b * 8, b * 8)
        self.dec3 = conv_block(b * 8 + b * 4, b * 4)
        self.dec2 = conv_block(b * 4 + b * 2, b * 2)
        self.dec1 = conv_block(b * 2 + b, b)
        self.head = nn.Conv2d(b, 1, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))
        m = self.mid(self.pool(e4))
        d4 = self.dec4(torch.cat([nn.functional.interpolate(m, scale_factor=2), e4], dim=1))
        d3 = self.dec3(torch.cat([nn.functional.interpolate(d4, scale_factor=2), e3], dim=1))
        d2 = self.dec2(torch.cat([nn.functional.interpolate(d3, scale_factor=2), e2], dim=1))
        d1 = self.dec1(torch.cat([nn.functional.interpolate(d2, scale_factor=2), e1], dim=1))
        return self.head(d1)


class UNetSegmenter(InteractiveSegmenter):
    """把 ClickUNet 包装成交互式分割器。"""

    def __init__(self, model: ClickUNet, size: int, sigma: float, device: str = "cpu",
                 threshold: float = 0.5):
        self.model = model.to(device).eval()
        self.size = size
        self.sigma = sigma
        self.device = device
        self.threshold = threshold
        self._img = None
        self._pos = np.zeros((size, size), dtype=np.float32)
        self._neg = np.zeros((size, size), dtype=np.float32)
        self._box = np.zeros((size, size), dtype=np.float32)
        self._prob = np.zeros((size, size), dtype=np.float32)

    def reset(self, image: np.ndarray) -> None:
        self._img = image.astype(np.float32)
        self._pos = np.zeros((self.size, self.size), dtype=np.float32)
        self._neg = np.zeros((self.size, self.size), dtype=np.float32)
        self._box = np.zeros((self.size, self.size), dtype=np.float32)
        self._prob = np.zeros((self.size, self.size), dtype=np.float32)

    def _forward(self) -> None:
        x = np.stack([self._img, self._pos, self._neg, self._box])[None]
        with torch.no_grad():
            logit = self.model(torch.from_numpy(x).to(self.device))
        self._prob = torch.sigmoid(logit)[0, 0].cpu().numpy()

    def add_click(self, y: int, x: int, positive: bool) -> np.ndarray:
        m = click_map(self.size, (y, x), self.sigma)
        if positive:
            self._pos = np.maximum(self._pos, m)
        else:
            self._neg = np.maximum(self._neg, m)
        self._forward()
        return self.mask()

    def add_box(self, y0: int, x0: int, y1: int, x1: int) -> np.ndarray:
        self._box = box_map(self.size, (y0, x0, y1, x1))
        self._forward()
        return self.mask()

    @property
    def prob(self) -> np.ndarray:
        return self._prob

    def predict_batch(self, images: np.ndarray, pos: np.ndarray, neg: np.ndarray,
                      box: np.ndarray) -> np.ndarray:
        """批推理 (轨迹收集/评测提速用): 返回 sigmoid 概率 [N,size,size]。"""
        x = np.stack([images, pos, neg, box], axis=1)
        out = np.empty((len(x), self.size, self.size), dtype=np.float32)
        with torch.no_grad():
            for i in range(0, len(x), 64):
                xb = torch.from_numpy(x[i:i + 64]).to(self.device)
                out[i:i + 64] = torch.sigmoid(self.model(xb))[:, 0].cpu().numpy()
        return out
