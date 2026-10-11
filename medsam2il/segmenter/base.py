"""交互式分割器统一接口 + 通用工具 (点击图编码 / Dice)。

任何后端 (轻量 U-Net 替身, 或真 MedSAM2/SAM2) 都实现该接口,
策略网络与评测代码对后端无感知 —— 换模型不改管线。
"""
from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np


def click_map(size: int, xy: tuple[float, float] | None, sigma: float) -> np.ndarray:
    """点击位置 -> 高斯热图 [size,size]。xy=None 返回全零。"""
    m = np.zeros((size, size), dtype=np.float32)
    if xy is None:
        return m
    yy, xx = np.mgrid[0:size, 0:size].astype(np.float32)
    y, x = float(xy[0]), float(xy[1])
    m = np.exp(-((yy - y) ** 2 + (xx - x) ** 2) / (2 * sigma ** 2))
    return m


def box_map(size: int, box: tuple[int, int, int, int] | None) -> np.ndarray:
    """框选 (y0,x0,y1,x1) -> 矩形指示图。"""
    m = np.zeros((size, size), dtype=np.float32)
    if box is None:
        return m
    y0, x0, y1, x1 = box
    m[max(y0, 0):y1 + 1, max(x0, 0):x1 + 1] = 1.0
    return m


def dice_binary(pred: np.ndarray, gt: np.ndarray, eps: float = 1e-6) -> float:
    p = pred.astype(bool)
    g = gt.astype(bool)
    if p.sum() == 0 and g.sum() == 0:
        return 1.0
    return float(2.0 * (p & g).sum() / max(p.sum() + g.sum(), eps))


def largest_component_centroid(mask: np.ndarray,
                              rng: np.random.Generator | None = None,
                              jitter: float = 0.0,
                              suboptimal: bool = False
                              ) -> tuple[int, int] | None:
    """取 mask 最大连通域的质心 (可选抖动/边缘次优点), 用于医生点击与误差定位。

    无 scipy 时用 BFS 求连通域 (mask 尺寸小, 足够快)。
    """
    if mask.sum() == 0:
        return None
    lab = np.zeros(mask.shape, dtype=np.int32)
    cur = 0
    sizes: dict[int, int] = {}
    H, W = mask.shape
    ys, xs = np.nonzero(mask)
    for y0, x0 in zip(ys, xs):
        if lab[y0, x0]:
            continue
        cur += 1
        stack = [(int(y0), int(x0))]
        lab[y0, x0] = cur
        n = 0
        while stack:
            y, x = stack.pop()
            n += 1
            for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                ny, nx = y + dy, x + dx
                if 0 <= ny < H and 0 <= nx < W and mask[ny, nx] and not lab[ny, nx]:
                    lab[ny, nx] = cur
                    stack.append((ny, nx))
        sizes[cur] = n
    best = max(sizes, key=sizes.get)
    cys, cxs = np.nonzero(lab == best)
    cy, cx = float(cys.mean()), float(cxs.mean())
    if suboptimal:
        # 次优: 在该连通域内随机取一点 (远离质心, 模拟真实医生手抖)
        idx = np.random.default_rng(0 if rng is None else rng.integers(2 ** 31))
        pick = idx.integers(0, len(cys))
        cy, cx = float(cys[pick]), float(cxs[pick])
    if rng is not None and jitter > 0:
        cy += float(rng.normal(0, jitter))
        cx += float(rng.normal(0, jitter))
    return int(np.clip(cy, 0, mask.shape[0] - 1)), int(np.clip(cx, 0, mask.shape[1] - 1))


class InteractiveSegmenter(ABC):
    """统一接口: 给定图像与历史点击/框, 输出当前分割。"""

    size: int
    supports_rollback = False   # 是否支持 try_click/pop_click (测试时候选搜索)

    @abstractmethod
    def reset(self, image: np.ndarray) -> None: ...

    @abstractmethod
    def add_click(self, y: int, x: int, positive: bool) -> np.ndarray:
        """加入一次点击, 返回当前二值 mask。"""

    def add_box(self, y0: int, x0: int, y1: int, x1: int) -> np.ndarray:
        raise NotImplementedError("该后端不支持框选")

    def try_click(self, y: int, x: int, positive: bool) -> float:
        """试一次可回滚的点击: 应用后返回分割器自信度, 调用方随后 pop_click 还原。

        测试时候选搜索用: 策略提多个候选, 逐个试, 用分割器自己的 predicted-IoU 选优
        —— 不依赖 GT, 纯部署时技巧。
        """
        self.add_click(y, x, positive)
        return self.confidence

    def pop_click(self) -> None:
        """回滚最近一次 try_click (交互式后端支持; 批处理替身不支持)。"""
        raise NotImplementedError("该后端不支持点击回滚")

    @property
    @abstractmethod
    def prob(self) -> np.ndarray:
        """当前概率图 (策略网络的"当前分割结果"输入)。"""

    @property
    def confidence(self) -> float:
        """分割器对当前 mask 的自信度 (0-1), 策略停手决策的关键输入。

        默认用峰值概率; MedSAM2 后端覆盖为官方 predicted-IoU 分数。
        """
        p = self.prob
        return float(p.max()) if p.size else 0.0

    def hypotheses(self) -> tuple[np.ndarray, np.ndarray] | None:
        """当前提示的多个候选 (masks [K,H,W] float, scores [K]); 无则 None。

        SAM2 multimask 每次返回 3 个候选 —— 候选间分歧是不依赖 GT 的
        不确定性信号, 供策略定位疑似误差区域。
        """
        return None

    def embedding(self) -> np.ndarray | None:
        """图像嵌入 (SAM2 image encoder 输出), 供策略网络做视觉特征。

        返回 [C, grid, grid] float16 或 None (后端不支持, 如轻量替身)。
        每张图只算一次 (set_image 时), 之后所有交互步复用。
        """
        return None

    def mask(self) -> np.ndarray:
        return (self.prob > 0.5).astype(np.uint8)
