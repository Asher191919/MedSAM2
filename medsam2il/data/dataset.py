"""数据获取统一入口: 合成数据生成/缓存, 或加载外部真实数据。"""
from __future__ import annotations

import numpy as np


def get_dataset(cfg, split_seed: int) -> tuple[np.ndarray, np.ndarray, dict]:
    """返回 (images [N,H,W] 灰度 或 [N,H,W,3] 彩色 0-1, masks [N,H,W] 0/1, meta)。
    split_seed 区分三个子集。"""
    ext = cfg.resolve(cfg.raw["data"].get("external_dir"))
    if ext is not None:
        from medsam2il.data.external import load_external
        return load_external(ext, int(cfg.raw["data"]["image_size"]),
                             color=bool(cfg.raw["data"].get("color", False)))
    from medsam2il.data.synthetic import generate_images
    d, sy = cfg.raw["data"], cfg.raw["synthetic"]
    size = int(d["image_size"])
    n = int(d["n_surrogate_train"]) + int(d["n_traj"]) + int(d["n_eval"])
    return generate_images(n, size, cfg.seed + split_seed, sy)
