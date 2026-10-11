"""评测任意策略 checkpoint (消融用): 同一测试集, 可开关停手护栏。

用法:
  python scripts/eval_checkpoint.py --ckpt outputs/bc_round1.pt --guard
  python scripts/eval_checkpoint.py --ckpt outputs/bc_policy.pt
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from medsam2il.config import Config  # noqa: E402
from medsam2il.data.dataset import get_dataset  # noqa: E402
from medsam2il.eval.report import evaluate_policies  # noqa: E402
from medsam2il.policy.net import InteractionPolicyNet  # noqa: E402
from medsam2il.segmenter.medsam2_adapter import make_segmenter  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--guard", action="store_true", help="启用停手护栏")
    ap.add_argument("--stop_mode", default=None,
                    choices=["classifier", "dice_regression"],
                    help="覆盖 checkpoint 里的停手模式 (消融用)")
    ap.add_argument("--candidates", type=int, default=None,
                    help="测试时候选搜索候选数 (覆盖 config; 1=关)")
    ap.add_argument("--config", default="configs/config_real.yaml")
    args = ap.parse_args()

    cfg = Config.load(args.config)
    if args.guard:
        cfg.raw["eval"]["stop_guard"] = True
    if args.candidates is not None:
        cfg.raw["eval"]["candidates"] = int(args.candidates)
    d = cfg.raw["data"]
    images, masks, meta = get_dataset(cfg, split_seed=0)
    n_eval = int(d["n_eval"])
    seg = make_segmenter(cfg)

    payload = __import__("torch").load(args.ckpt, map_location=cfg.device,
                                       weights_only=True)
    net = InteractionPolicyNet(grid=payload["grid"],
                               embed_ch=int(payload.get("embed_ch", 0)),
                               in_ch=int(payload.get("in_ch", 11))
                               ).to(cfg.device)
    net.load_state_dict(payload["state_dict"])
    net.tau = float(payload.get("tau", 0.5))
    net.stop_mode = args.stop_mode or payload.get("stop_mode", "classifier")
    net.dice_theta = float(payload.get("dice_theta", 0.85))
    net.fine_offset = bool(payload.get("fine_offset", False))
    net.eval()

    # 嵌入缓存按数据集隔离 (outputs/cache_<dataset>/embeds_eval.npz)
    embeds_eval = None
    ds_tag = Path(d["external_dir"]).name if d.get("external_dir") else "synthetic"
    if d.get("run_name"):
        ds_tag += f"_{d['run_name']}"
    for ep in (cfg.resolve("outputs") / f"cache_{ds_tag}" / "embeds_eval.npz",
               cfg.resolve("outputs") / "embeds_eval.npz"):
        if ep.exists() and int(payload.get("embed_ch", 0)) > 0:
            embeds_eval = np.load(ep)["embeds"]
            print(f"[embeds] 载入 {ep}")
            break
    if embeds_eval is None and int(payload.get("embed_ch", 0)) > 0:
        print("[embeds] 警告: 未找到嵌入缓存, BC 将在无嵌入分支下评测 (偏低估)")

    res = evaluate_policies(cfg, seg, images[-n_eval:], masks[-n_eval:],
                            np.array(meta["area_frac"][-n_eval:]), net,
                            embeds_eval=embeds_eval)
    b = res["metrics"]["bc_policy"]
    print(f"\n=== {Path(args.ckpt).name} guard={args.guard} "
          f"candidates={cfg.raw['eval'].get('candidates', 1)} ===")
    print(f"BC: Dice@1={b['dice_curve'][1]:.3f} Dice@3={b['dice_curve'][3]:.3f} "
          f"ATTC={b['attc_mean_censored']:.2f} 成功率={b['success_rate']:.1%}")


if __name__ == "__main__":
    main()
