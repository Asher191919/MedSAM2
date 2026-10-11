"""外部真实数据集加载: images/ + masks/ 两个文件夹 (png/jpg, mask 白=肿瘤)。

公开数据集示例 (均可转为该格式后直接使用):
  - ISIC (皮肤黑色素瘤): isic-archive.com
  - Kvasir-SEG (胃肠息肉): 同名 GitHub
  - LiTS / 3D-IRCADb (肝肿瘤 CT): 导出 2D 切片即可
优先级: config.external_dir 存在则用它, 否则用合成数据。

color=True 时保留 RGB (皮肤镜颜色是强线索; 灰度管线会丢掉它),
图像数组形状变为 [N,H,W,3]; 默认灰度 [N,H,W] (兼容旧缓存)。
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image

IMG_EXTS = {".png", ".jpg", ".jpeg", ".bmp"}


def load_external(root: Path, size: int, color: bool = False
                  ) -> tuple[np.ndarray, np.ndarray, dict]:
    img_dir, msk_dir = root / "images", root / "masks"
    files = sorted(p for p in img_dir.iterdir()
                   if p.suffix.lower() in IMG_EXTS)
    if not files:
        raise FileNotFoundError(f"{img_dir} 中没有图像")
    images, masks = [], []
    for f in files:
        m = msk_dir / f.name
        if not m.exists():  # 常见命名差异
            cand = list(msk_dir.glob(f.stem + ".*"))
            if not cand:
                continue
            m = cand[0]
        img = Image.open(f)
        img = (img.convert("RGB") if color else img.convert("L")).resize((size, size))
        msk = Image.open(m).convert("L").resize((size, size), Image.NEAREST)
        images.append(np.asarray(img, dtype=np.float32) / 255.0)
        masks.append((np.asarray(msk) > 127).astype(np.uint8))
    images = np.stack(images)
    masks = np.stack(masks)
    meta = {"area_frac": masks.mean(axis=(1, 2)).tolist(),
            "contrast": [float("nan")] * len(images),
            "radius": [float("nan")] * len(images),
            "names": [f.name for f in files]}
    return images, masks, meta
