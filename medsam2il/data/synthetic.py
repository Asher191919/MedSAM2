"""合成 2D "CT 切片" + 肿瘤 mask 生成器。

用途: 在拿到真实数据 (公开数据集或院内数据) 之前, 让
分割替身训练 -> 医生专家轨迹 -> 模仿学习 -> 评估 的整条闭环可以先跑通。

图像构成:
  - 平滑背景场 (随机低频平面波叠加) + 高斯噪声
  - 若干"器官"椭圆 (强度不同, 边缘平滑过渡)
  - 肿瘤 = 谐波调制的 irregular 团块, 与所在器官有一定强度差 (contrast 可控难度)
真实数据接入: 见 medsam2il/data/external.py, 文件夹格式 images/ + masks/。
"""
from __future__ import annotations

import numpy as np


def _smooth_field(rng: np.random.Generator, size: int, n_waves: int = 24,
                  amp: float = 0.05) -> np.ndarray:
    """低频平面波叠加的平滑背景场。"""
    yy, xx = np.mgrid[0:size, 0:size].astype(np.float32) / size
    field = np.zeros((size, size), dtype=np.float32)
    for _ in range(n_waves):
        wx, wy = rng.normal(0, 6.0, size=2)
        a = rng.uniform(-amp, amp)
        ph = rng.uniform(0, 2 * np.pi)
        field += a * np.sin(wx * xx * 2 * np.pi + wy * yy * 2 * np.pi + ph)
    return field


def _ellipseblob(size: int, cy: float, cx: float, ry: float, rx: float,
                 rot: float, edge: float = 2.0) -> np.ndarray:
    """平滑边缘椭圆 (sigmoid 距离场)。"""
    yy, xx = np.mgrid[0:size, 0:size].astype(np.float32)
    dy, dx = yy - cy, xx - cx
    yr = dy * np.cos(rot) + dx * np.sin(rot)
    xr = -dy * np.sin(rot) + dx * np.cos(rot)
    d = np.sqrt((yr / max(ry, 1e-3)) ** 2 + (xr / max(rx, 1e-3)) ** 2)
    z = np.clip(-(1.0 - d) * max(ry, rx) / edge, -30, 30)
    return 1.0 / (1.0 + np.exp(z))


def _tumor_mask(rng: np.random.Generator, size: int, cy: float, cx: float,
                r: float) -> np.ndarray:
    """谐波调制的不规则肿瘤团块。"""
    theta = np.linspace(0, 2 * np.pi, 720, dtype=np.float32)
    shape = (1.0
             + rng.uniform(0.08, 0.30) * np.sin(3 * theta + rng.uniform(0, 6.28))
             + rng.uniform(0.04, 0.18) * np.sin(5 * theta + rng.uniform(0, 6.28))
             + rng.uniform(0.02, 0.10) * np.sin(8 * theta + rng.uniform(0, 6.28)))
    rr = r * shape
    yy, xx = np.mgrid[0:size, 0:size].astype(np.float32)
    ang = np.arctan2(yy - cy, xx - cx) % (2 * np.pi)
    rad = np.sqrt((yy - cy) ** 2 + (xx - cx) ** 2)
    limit = np.interp(ang, theta, rr)
    return (rad <= limit).astype(np.float32)


def _smooth_2d(x: np.ndarray, k: int = 5) -> np.ndarray:
    """简单盒式模糊 (避免引入 scipy)。"""
    pad = k // 2
    xp = np.pad(x, pad, mode="edge")
    cs = np.cumsum(np.cumsum(xp, axis=0), axis=1)
    cs = np.pad(cs, ((1, 0), (1, 0)))
    n = k * k
    s = (cs[k:, k:] - cs[:-k, k:] - cs[k:, :-k] + cs[:-k, :-k]) / n
    return s.astype(np.float32)


def generate_images(n: int, size: int, seed: int, sy: dict) -> tuple[np.ndarray, np.ndarray, dict]:
    """返回 images [n,size,size] float32 (0-1), masks [n,size,size] uint8, meta。"""
    rng = np.random.default_rng(seed)
    images = np.zeros((n, size, size), dtype=np.float32)
    masks = np.zeros((n, size, size), dtype=np.uint8)
    meta = {"area_frac": [], "contrast": [], "radius": [],
            "names": [f"synthetic_{i:04d}" for i in range(n)]}
    r_lo, r_hi = sy["radius_range"]

    for i in range(n):
        img = 0.22 + _smooth_field(rng, size)
        # 器官: 3-5 个不同强度椭圆
        n_organs = int(rng.integers(3, 6))
        organs = []
        for _ in range(n_organs):
            cy, cx = rng.uniform(size * 0.25, size * 0.75, size=2)
            ry, rx = rng.uniform(size * 0.10, size * 0.28, size=2)
            inten = rng.uniform(0.42, 0.72)
            el = _ellipseblob(size, cy, cx, ry, rx, rng.uniform(0, np.pi))
            img = np.where(el > img, el * inten + (1 - el) * img, img)
            organs.append((cy, cx, min(ry, rx), inten))

        # 肿瘤: 偏向放进某个器官内
        r = float(rng.uniform(r_lo, r_hi))
        if organs and rng.random() < 0.8:
            oy, ox, orad, inten = organs[int(rng.integers(0, len(organs)))]
            ang = rng.uniform(0, 2 * np.pi)
            dist = max(orad - r * 1.4, 0)
            cy = oy + np.sin(ang) * dist * rng.uniform(0.2, 0.9)
            cx = ox + np.cos(ang) * dist * rng.uniform(0.2, 0.9)
            base_inten = inten
        else:
            cy, cx = rng.uniform(r + 6, size - r - 6, size=2)
            base_inten = 0.5
        cy = float(np.clip(cy, r + 4, size - r - 4))
        cx = float(np.clip(cx, r + 4, size - r - 4))

        contrast = float(rng.normal(sy["contrast_mean"], sy["contrast_std"]))
        contrast = float(np.sign(contrast) * max(abs(contrast), 0.05))
        tm = _tumor_mask(rng, size, cy, cx, r)
        tm_soft = np.clip(_smooth_2d(tm, 3), 0, 1)
        tumor_inten = np.clip(base_inten + contrast, 0.05, 0.95)
        img = img * (1 - tm_soft) + tumor_inten * tm_soft

        img = img + rng.normal(0, sy["noise_std"], size=(size, size)).astype(np.float32)
        images[i] = np.clip(img, 0, 1)
        masks[i] = (tm_soft > 0.5).astype(np.uint8)
        meta["area_frac"].append(float(masks[i].mean()))
        meta["contrast"].append(contrast)
        meta["radius"].append(r)

    return images, masks, meta
