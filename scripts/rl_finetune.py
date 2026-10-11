"""对已训练的 BC 策略做 RL 微调 (REINFORCE, Dice 增益奖励), 并评测对比。

用法:
  python scripts/rl_finetune.py --ckpt outputs/bc_policy.pt [--epochs 2] [--images 500]
产出: outputs/bc_policy_rl.pt + 测试集对比数字 (仅报告用, 选择按验证集)。
"""
from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from medsam2il.config import Config  # noqa: E402
from medsam2il.data.dataset import get_dataset  # noqa: E402
from medsam2il.policy.net import InteractionPolicyNet  # noqa: E402
from medsam2il.segmenter.medsam2_adapter import make_segmenter  # noqa: E402
from medsam2il.train.finetune_rl import finetune_rl  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="outputs/bc_policy.pt")
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--images", type=int, default=500)
    ap.add_argument("--config", default="configs/config_real.yaml")
    args = ap.parse_args()

    cfg = Config.load(args.config)
    if args.epochs:
        cfg.raw["policy"]["rl_epochs"] = args.epochs
    if args.images:
        cfg.raw["policy"]["rl_images"] = args.images
    d = cfg.raw["data"]
    images, masks, meta = get_dataset(cfg, split_seed=0)
    n_traj = int(d["n_traj"])
    seg = make_segmenter(cfg)

    import torch
    payload = torch.load(args.ckpt, map_location=cfg.device, weights_only=True)
    net = InteractionPolicyNet(grid=payload["grid"],
                               embed_ch=int(payload.get("embed_ch", 0))
                               ).to(cfg.device)
    net.load_state_dict(payload["state_dict"])
    net.tau = float(payload.get("tau", 0.5))
    net.val_images = payload.get("val_images")
    net.eval()

    embeds = None
    ep = Path("outputs") / "cache_kvasir_seg" / "embeds_traj.npz"
    if ep.exists() and int(payload.get("embed_ch", 0)) > 0:
        embeds = np.load(ep)["embeds"]

    net_rl, best = finetune_rl(cfg, net, seg, images[:n_traj], masks[:n_traj],
                               embeds=embeds)

    out = Path("outputs") / "bc_policy_rl.pt"
    payload["state_dict"] = net_rl.state_dict()
    payload["val_success"] = best
    torch.save(payload, out)
    print(f"[rl] 保存 -> {out} (val 成功率={best:.1%})")


if __name__ == "__main__":
    main()
