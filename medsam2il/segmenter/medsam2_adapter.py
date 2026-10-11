"""真 MedSAM2 / SAM2 适配器 (需 pip install sam2 + 官方权重)。

- 与 UNetSegmenter 实现同一个 InteractiveSegmenter 接口: 策略/评测代码零改动;
- set_image 只跑一次编码器 (贵), 之后每次点击只跑轻量解码器 —— CPU 也可用;
- 坐标: 管线在 image_size(128) 空间工作, SAM2 直接吃该图, 坐标系天然一致;
- 权重: configs/config_real.yaml -> medsam2.checkpoint
  (官方 wanglab/MedSAM2, 可经 hf-mirror.com 下载, 见 scripts/prepare_real_data.py)。
"""
from __future__ import annotations

import time
from pathlib import Path

import numpy as np

from medsam2il.segmenter.base import InteractiveSegmenter, box_map, click_map


def to_rgb(image: np.ndarray) -> np.ndarray:
    """灰度 [H,W] 或彩色 [H,W,3] (0-1 float) -> uint8 RGB (SAM2 输入)。"""
    a = np.clip(np.asarray(image, dtype=np.float32), 0, 1)
    if a.ndim == 2:
        a = np.repeat(a[..., None], 3, axis=-1)
    return (a * 255.0 + 0.5).astype(np.uint8)


class MedSAM2Segmenter(InteractiveSegmenter):
    supports_rollback = True    # try_click/pop_click: 测试时候选搜索可回滚试错

    def __init__(self, checkpoint: str, config_candidates: list[str], size: int,
                 device: str = "auto", color_input: bool | None = None):
        """color_input: None=跟随输入图像 (灰度/彩色); True/False=强制 (消融用)。"""
        try:
            import torch
            from sam2.build_sam import build_sam2
            from sam2.sam2_image_predictor import SAM2ImagePredictor
        except ImportError as e:
            raise ImportError(
                "未安装 SAM2: pip install sam2  (PyPI 官方包, 1.1.0+)") from e

        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device
        self.size = size
        self.color_input = color_input
        self._points: list[list[float]] = []
        self._labels: list[int] = []          # 1=正 0=负
        self._box: list[float] | None = None
        self._prob = np.zeros((size, size), dtype=np.float32)
        self._has_prompt = False
        self._conf = 0.0

        last_err = None
        for cfg_name in config_candidates:
            try:
                t0 = time.time()
                self.model = build_sam2(cfg_name, checkpoint, device=device)
                print(f"[medsam2] 加载成功: {cfg_name} "
                      f"({time.time() - t0:.0f}s, device={device})")
                break
            except Exception as e:   # 架构不匹配 / hydra 找不到配置
                last_err = e
                print(f"[medsam2] {cfg_name} 失败: {type(e).__name__}: {e}")
        else:
            raise RuntimeError(f"所有配置都无法加载权重 {checkpoint}: {last_err}")
        self.predictor = SAM2ImagePredictor(self.model)

    # ------------------------------------------------------------------
    def reset(self, image: np.ndarray) -> None:
        self._points, self._labels, self._box = [], [], None
        self._has_prompt = False
        self._conf = 0.0
        self._embed = None
        self._hyp = None
        img = np.asarray(image)
        if self.color_input is False and img.ndim == 3:   # 消融: 强制灰度喂 SAM2
            img = (0.299 * img[..., 0] + 0.587 * img[..., 1]
                   + 0.114 * img[..., 2]).astype(np.float32)
        rgb = to_rgb(img)
        self.predictor.set_image(rgb)
        # 缓存图像嵌入 (set_image 只跑一次编码器; 嵌入对所有交互步不变)
        import torch
        f = self.predictor._features
        if f is not None and "image_embed" in f:
            e = f["image_embed"]
            e = torch.nn.functional.adaptive_avg_pool2d(e, (16, 16))
            self._embed = e[0].cpu().numpy().astype(np.float16)

    def hypotheses(self):
        return self._hyp

    def embedding(self) -> np.ndarray | None:
        return self._embed

    def _predict(self) -> None:
        import torch
        if not self._has_prompt:
            self._prob = np.zeros((self.size, self.size), dtype=np.float32)
            return
        pts = np.array(self._points, dtype=np.float32) if self._points else None
        labs = np.array(self._labels, dtype=np.int64) if self._labels else None
        boxes = np.array([self._box], dtype=np.float32) if self._box is not None else None
        with torch.no_grad():
            masks, scores, _ = self.predictor.predict(
                point_coords=pts, point_labels=labs, box=boxes,
                multimask_output=True)      # 恒开: 多假设分歧 = 免费不确定性
        best = int(np.argmax(scores))
        self._prob = masks[best].astype(np.float32)   # SAM2 返回二值 mask, 作概率代理
        self._conf = float(scores[best])              # predicted-IoU: 停手决策的关键信号
        self._hyp = (masks.astype(np.float32), scores.astype(np.float32))

    def add_click(self, y: int, x: int, positive: bool) -> np.ndarray:
        self._points.append([float(x), float(y)])     # SAM2 点坐标序为 (x, y)
        self._labels.append(1 if positive else 0)
        self._has_prompt = True
        self._predict()
        return self.mask()

    def pop_click(self) -> None:
        """回滚一次试点击 (try_click 的配套)。

        只弹出最后一个点; 不回滚框。回滚后 prob/conf 暂不可信 —— 但下一次
        add_click 会用完整提示重新预测, 因此调用方必须先应用正式点击再读状态。
        """
        if not self._points:
            return
        self._points.pop()
        self._labels.pop()
        if not self._points:
            self._has_prompt = self._box is not None

    def add_box(self, y0: int, x0: int, y1: int, x1: int) -> np.ndarray:
        self._box = [float(x0), float(y0), float(x1), float(y1)]
        self._has_prompt = True
        self._predict()
        return self.mask()

    @property
    def prob(self) -> np.ndarray:
        return self._prob

    @property
    def confidence(self) -> float:
        return self._conf

    def hypotheses(self):
        return self._hyp

    def embedding(self) -> np.ndarray | None:
        return self._embed


def make_segmenter(cfg):
    """按配置构造分割后端: 有 MedSAM2 权重用真模型, 否则用轻量替身。"""
    m = cfg.raw.get("medsam2", {})
    ckpt = cfg.resolve(m.get("checkpoint")) if m.get("checkpoint") else None
    if ckpt is not None and Path(ckpt).exists():
        d = cfg.raw["data"]
        cands = m.get("config_candidates") or ["configs/sam2.1/sam2.1_hiera_tiny.yaml"]
        # color_input 缺省跟随数据 (data.color); 显式 true/false 可强制 (消融)
        color_input = m.get("color_input", d.get("color", False))
        return MedSAM2Segmenter(str(ckpt), cands, int(d["image_size"]),
                                m.get("device", "auto"),
                                color_input=bool(color_input))
    if ckpt is not None:
        print(f"[medsam2] 权重不存在: {ckpt}, 回退到轻量替身")
    from medsam2il.segmenter.unet_surrogate import ClickUNet, UNetSegmenter
    import torch
    surrogate_ckpt = cfg.resolve("outputs") / "surrogate_unet.pt"
    if not surrogate_ckpt.exists():
        raise FileNotFoundError("既无 MedSAM2 权重也无替身权重, 请先训练替身或下载权重")
    payload = torch.load(surrogate_ckpt, map_location=cfg.device, weights_only=True)
    net = ClickUNet(base=payload["base"]).to(cfg.device)
    net.load_state_dict(payload["state_dict"])
    d, su = cfg.raw["data"], cfg.raw["surrogate"]
    return UNetSegmenter(net, int(d["image_size"]), float(su["sigma_click"]),
                         cfg.device)
