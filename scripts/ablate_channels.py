"""⑥ 通道消融 (评测时置零): 用数字证明各状态通道的贡献。

对 13 通道状态做留一消融: 图像 RGB (0-2) 作为一组整体置零,
其余 10 个通道逐个置零, 其余不变地跑 BC 评测 (同一批 100 张、同一初始框)。
置零衡量的是**出货策略对各通道的依赖** (推断时可实现, 如传感器缺该信号);
重训式消融是金标准但每组需 ~40 分钟训练, 见 README 说明。

用法: python scripts/ablate_channels.py [--out outputs/ablation_channels.md]
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from medsam2il.config import Config  # noqa: E402
from medsam2il.data.dataset import get_dataset  # noqa: E402
from medsam2il.eval import report as R  # noqa: E402
from medsam2il.eval.rollout import clicks_to_target, dice_curves, run_episodes  # noqa: E402
from medsam2il.policy.net import InteractionPolicyNet  # noqa: E402
from medsam2il.segmenter.medsam2_adapter import make_segmenter  # noqa: E402

# (通道索引元组, 名称, 说明) —— 与 state.build_state 的堆叠顺序一致
CHANNELS = [
    ((0, 1, 2), "image_rgb", "主干视觉输入 (RGB 三通道置零; SAM2 嵌入分支仍在)"),
    ((3,), "prob", "当前分割概率图"),
    ((4,), "uncertainty", "候选 mask 分歧图"),
    ((5,), "img_edge", "图像边缘"),
    ((6,), "mask_edge", "分割边缘"),
    ((7,), "pos_map", "历史正点击"),
    ((8,), "neg_map", "历史负点击"),
    ((9,), "progress", "交互进度"),
    ((10,), "confidence", "分割器自信度 (predicted-IoU)"),
    ((11,), "disagreement", "候选 mask 逐像素分歧 (浮点)"),
    ((12,), "score_std", "候选分数方差"),
]

_build_state_orig = R.build_state
# build_state 在 net/report/rollout 各有一份模块级引用, 置零 patch 必须全覆盖
import medsam2il.policy.net as _net_mod  # noqa: E402
import medsam2il.eval.rollout as _rollout_mod  # noqa: E402
import medsam2il.policy.state as _state_mod  # noqa: E402
_PATCH_SITES = (R, _net_mod, _rollout_mod, _state_mod)


def _patch_build_state(zero_ch):
    if not zero_ch:
        return
    def patched(*a, **k):
        s = _build_state_orig(*a, **k)
        s = s.copy()
        s[list(zero_ch)] = 0.0
        return s
    for m in _PATCH_SITES:
        m.build_state = patched


def _unpatch_build_state():
    for m in _PATCH_SITES:
        m.build_state = _build_state_orig


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/config_real.yaml")
    ap.add_argument("--ckpt", default="outputs/bc_policy.pt")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    cfg = Config.load(args.config)
    d, ev = cfg.raw["data"], cfg.raw["eval"]
    images, masks, meta = get_dataset(cfg, split_seed=0)
    n_eval = int(d["n_eval"])
    images, masks = images[-n_eval:], masks[-n_eval:]
    area = np.array(meta["area_frac"][-n_eval:])

    ds_tag = Path(d["external_dir"]).name if d.get("external_dir") else "synthetic"
    if d.get("run_name"):
        ds_tag += f"_{d['run_name']}"
    embeds = None
    ep = cfg.resolve("outputs") / f"cache_{ds_tag}" / "embeds_eval.npz"
    payload = torch.load(args.ckpt, map_location=cfg.device, weights_only=True)
    net = InteractionPolicyNet(grid=payload["grid"],
                               embed_ch=int(payload.get("embed_ch", 0)),
                               in_ch=int(payload.get("in_ch", 11))).to(cfg.device)
    net.load_state_dict(payload["state_dict"])
    net.tau = float(payload.get("tau", 0.5))
    net.stop_mode = payload.get("stop_mode", "classifier")
    net.dice_theta = float(payload.get("dice_theta", 0.85))
    net.fine_offset = bool(payload.get("fine_offset", False))
    net.eval()
    if int(payload.get("embed_ch", 0)) > 0 and ep.exists():
        embeds = np.load(ep)["embeds"]
        print(f"[embeds] 载入 {ep}")

    seg = make_segmenter(cfg)
    max_clicks = int(ev["max_clicks"])
    init_mode = cfg.raw["doctor"].get("init_mode", "click")
    grid, size = int(d["grid"]), int(d["image_size"])

    def run_bc(zero_ch: tuple[int, ...] = ()) -> tuple[float, float]:
        """置零给定通道后跑 BC 评测, 返回 (成功率@0.85, ATTC@0.85)。"""
        _patch_build_state(zero_ch)
        try:
            def factory(gt, rng, i=0):
                emb = None if embeds is None else embeds[i]
                return R._BCAdapter(net, max_clicks, grid, size, init_mode,
                                    embed=emb,
                                    stop_guard=bool(ev.get("stop_guard", False)))
            eps, _ = run_episodes(seg, images, masks, factory, cfg)
        finally:
            _unpatch_build_state()
        curves = dice_curves(eps, max_clicks)
        k_need, ok = clicks_to_target(curves, 0.85)
        return float(ok.mean()), float(k_need.mean())

    rows, t0 = [], time.time()
    s0, a0 = run_bc()
    rows.append(("-", "baseline (无置零)", s0, a0))
    print(f"[ablate] baseline: 成功@0.85={s0:.1%} ATTC={a0:.2f}", flush=True)
    for chs, name, desc in CHANNELS:
        s, a = run_bc(chs)
        rows.append((str(list(chs)), f"{name} 置零", s, a))
        print(f"[ablate] {name:13s} 置零: 成功@0.85={s:.1%} (Δ{s - s0:+.1%}) "
              f"ATTC={a:.2f} (Δ{a - a0:+.2f})  [{time.time() - t0:.0f}s]",
              flush=True)

    lines = ["# ⑥ 通道消融 (评测时置零, Kvasir RGB 出货策略, 100 张测试图)", "",
             f"> 基线: 成功率@0.85 = {s0:.1%}, ATTC = {a0:.2f}。Δ 为相对基线变化; "
             "置零衡量策略对该通道的依赖 (重训式消融为金标准, 代码 "
             "`scripts/ablate_channels.py`)。", "",
             "| 置零通道 | 成功率@0.85 | Δ成功率 | ATTC | ΔATTC |",
             "|---|---|---|---|---|"]
    for chs, name, s, a in rows:
        if name.startswith("baseline"):
            lines.append(f"| {name} | {s:.1%} | - | {a:.2f} | - |")
        else:
            lines.append(f"| {name} | {s:.1%} | {s - s0:+.1%} | {a:.2f} | {a - a0:+.2f} |")
    report = "\n".join(lines)
    print("\n" + report)
    out = args.out or f"outputs/ablation_channels.json"
    with open(cfg.resolve("outputs") / Path(out).name if not Path(out).is_absolute()
              else out, "w", encoding="utf-8") as f:
        json.dump({"baseline": {"success": s0, "attc": a0},
                   "rows": [{"channels": c, "name": n, "success": s, "attc": a}
                            for c, n, s, a in rows]}, f, ensure_ascii=False, indent=2)
    if args.out:
        Path(args.out).write_text(report, encoding="utf-8")
        print(f"[saved] {args.out}")


if __name__ == "__main__":
    main()
