"""策略网络的状态构建。

按方案定义: 状态 = 当前图像 + 当前分割结果 + 历史点击。
通道 [image(灰度1/彩色3), 分割概率, 不确定度, 图像边缘, 分割边界, 正点击累计图,
       负点击累计图, 交互进度, 分割器自信度, 多假设分歧图, 分数标准差]
= 灰度 11 通道 / 彩色 13 通道 (皮肤镜颜色是强线索, 灰度管线把它丢掉了)。
多假设分歧: SAM2 每次提示返回多个候选 mask, 候选间的不一致区域即"最可能切错处"
(推理时免费获得, 不依赖 GT)。

关键设计:
  - "图像边缘 vs 分割边界错位"是医生判断"哪里切错了"的核心视觉线索;
  - "分割器自信度" (MedSAM2 的 predicted-IoU / 替身的峰值概率) 是"该不该停"的
    最强信号, 且推理时完全可得;
  - 不含 GT 误差图 —— 策略只使用推理时可得信息。
"""
from __future__ import annotations

import numpy as np

N_STATE_CH = 11          # 灰度输入 (兼容旧 checkpoint/缓存)
N_STATE_CH_COLOR = 13    # RGB 输入


def n_state_ch(image: np.ndarray | None) -> int:
    """状态通道数: 彩色 [H,W,3] -> 13, 灰度 [H,W] -> 11。"""
    return N_STATE_CH_COLOR if (image is not None
                                and np.asarray(image).ndim == 3) else N_STATE_CH


def _luma(image: np.ndarray) -> np.ndarray:
    """亮度 (灰度图原样返回; 彩色按 Rec.601 加权)。"""
    if image.ndim == 3:
        return (0.299 * image[..., 0] + 0.587 * image[..., 1]
                + 0.114 * image[..., 2]).astype(np.float32)
    return image.astype(np.float32)


def _grad_mag(a: np.ndarray) -> np.ndarray:
    gy, gx = np.gradient(a.astype(np.float32))
    return np.sqrt(gx * gx + gy * gy)


def _box_blur(x: np.ndarray, k: int = 5) -> np.ndarray:
    pad = k // 2
    xp = np.pad(x, pad, mode="edge")
    cs = np.cumsum(np.cumsum(xp, axis=0), axis=1)
    cs = np.pad(cs, ((1, 0), (1, 0)))
    return ((cs[k:, k:] - cs[:-k, k:] - cs[k:, :-k] + cs[:-k, :-k]) / (k * k)).astype(np.float32)


def build_state(image: np.ndarray, prob: np.ndarray,
                pos_map: np.ndarray, neg_map: np.ndarray,
                step: int, max_clicks: int, conf: float = 0.0,
                hyp_disagree: np.ndarray | None = None,
                score_std: float = 0.0) -> np.ndarray:
    """返回 float32 [C,H,W] (灰度 11 / 彩色 13 通道)。
    conf = 分割器自信度; hyp_disagree = 多候选 mask 的逐像素标准差
    (无多假设时为 0); score_std = 候选分数的标准差。"""
    p = np.clip(prob.astype(np.float32), 0, 1)
    unc = 4.0 * p * (1 - p)                                  # 二值熵
    lum = _luma(image)
    e = _grad_mag(lum)
    scale = np.percentile(e, 95) + 1e-6                      # 逐图归一 (跨域稳健)
    img_edge = np.clip(e / scale, 0, 1)
    mask_edge = np.clip(_box_blur(_grad_mag((p > 0.5).astype(np.float32))), 0, 1)
    prog = np.full(lum.shape, step / max(max_clicks, 1), dtype=np.float32)
    conf_ch = np.full(lum.shape, float(np.clip(conf, 0, 1)), dtype=np.float32)
    dis = (np.zeros(lum.shape, dtype=np.float32) if hyp_disagree is None
           else np.clip(hyp_disagree.astype(np.float32), 0, 1))
    sstd = np.full(lum.shape, float(np.clip(score_std, 0, 1)), dtype=np.float32)
    img_ch = ([image[..., c].astype(np.float32) for c in range(image.shape[2])]
              if image.ndim == 3 else [image.astype(np.float32)])
    return np.stack([*img_ch, p, unc, img_edge, mask_edge,
                     pos_map.astype(np.float32), neg_map.astype(np.float32),
                     prog, conf_ch, dis, sstd])


def pack_state(s: np.ndarray) -> np.ndarray:
    """float [C,H,W] -> uint8 (存储省内存)。"""
    return np.clip(s * 255.0, 0, 255).astype(np.uint8)


def unpack_state(s: np.ndarray) -> np.ndarray:
    return s.astype(np.float32) / 255.0


# ------------------------------------------------------------ 动作空间 ----
# a = (网格单元 y,x) x {正点击, 负点击(擦除)} + 确认(stop), 共 grid*grid*2+1 类
def n_actions(grid: int) -> int:
    return grid * grid * 2 + 1


def action_to_label(grid: int, cell_y: int, cell_x: int, positive: bool) -> int:
    off = 0 if positive else grid * grid
    return off + cell_y * grid + cell_x


def stop_label(grid: int) -> int:
    return grid * grid * 2


def label_to_action(grid: int, label: int) -> tuple[str, int, int, bool] | str:
    if label >= stop_label(grid):
        return "stop"
    positive = label < grid * grid
    off = 0 if positive else grid * grid
    v = label - off
    return "click", int(v // grid), int(v % grid), positive


def cell_of(y: float, x: float, grid: int, size: int) -> tuple[int, int]:
    return (int(np.clip(y, 0, size - 1)) * grid // size,
            int(np.clip(x, 0, size - 1)) * grid // size)


def cell_center(cy: int, cx: int, grid: int, size: int) -> tuple[int, int]:
    return (int((cy + 0.5) * size / grid), int((cx + 0.5) * size / grid))
