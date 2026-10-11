"""配置加载 (MedSAM2 子项目, 自包含, 不依赖父目录 huatuo 包)。"""
from __future__ import annotations

from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]


class Config:
    def __init__(self, raw: dict, path: Path | None = None):
        self.raw = raw
        self.path = path

    @classmethod
    def load(cls, path: str | Path | None = None) -> "Config":
        p = Path(path) if path else REPO_ROOT / "configs" / "config.yaml"
        with open(p, "r", encoding="utf-8") as f:
            return cls(yaml.safe_load(f), p)

    def resolve(self, v: str | None) -> Path | None:
        """相对路径 -> 相对本项目根目录。"""
        if v is None:
            return None
        p = Path(v)
        return p if p.is_absolute() else REPO_ROOT / p

    @property
    def seed(self) -> int:
        return int(self.raw["seed"])

    @property
    def device(self) -> str:
        """训练设备: 有 GPU 用 cuda, 否则 cpu (替身模型 CPU 可训)。"""
        return "cuda" if _cuda_ok() else "cpu"


def _cuda_ok() -> bool:
    try:
        import torch
        return torch.cuda.is_available()
    except Exception:
        return False
