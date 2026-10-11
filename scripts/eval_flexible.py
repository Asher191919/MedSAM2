"""灵活评测: 任意策略组合在任意配置上跑同一批配对回合, 保存每回合
Dice 曲线与点击记录 (供 McNemar/ATTC 敏感性与"随机医生反常"分析)。

用法:
  # ④ 零样本迁移 (Kvasir 彩色策略 -> ISIC 测试集) + ⑦ 反常数据采集
  python scripts/eval_flexible.py --config configs/config_isic_rgb.yaml \
      --ckpt outputs/bc_policy.pt \
      --policies bc_policy,random_doctor,simulated_doctor \
      --out outputs/zeroshot_isic.json

  # ⑤ 统计检验用每回合数据 (Kvasir RGB)
  python scripts/eval_flexible.py --config configs/config_real.yaml \
      --ckpt outputs/bc_policy.pt \
      --policies bc_policy,single_click,simulated_doctor \
      --out outputs/episodes_kvasir_rgb.json
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
from medsam2il.eval.report import _BCAdapter  # noqa: E402
from medsam2il.eval.rollout import (BOX_KIND, clicks_to_target, dice_curves,  # noqa: E402
                                    run_episodes)
from medsam2il.policy.doctor import (DoctorPolicy, MultiCandidateDoctor,  # noqa: E402
                                     RandomDoctor, SingleClickDoctor)
from medsam2il.policy.net import InteractionPolicyNet  # noqa: E402
from medsam2il.segmenter.medsam2_adapter import make_segmenter  # noqa: E402

TARGETS = (0.80, 0.85, 0.90)   # ⑤ ATTC 敏感性扫描的三档


def _embedding_cache(cfg, seg, images, cache_dir: Path):
    """与 run_all 相同的嵌入缓存逻辑 (编码器昂贵, 每图只算一次)。"""
    path = cache_dir / "embeds_eval.npz"
    if path.exists():
        return np.load(path)["embeds"]
    seg.reset(images[0])
    if seg.embedding() is None:
        return None
    embs, t0 = [], time.time()
    for k, img in enumerate(images):
        seg.reset(img)
        embs.append(seg.embedding())
        if (k + 1) % 50 == 0:
            rate = (time.time() - t0) / (k + 1)
            print(f"    ... {k + 1}/{len(images)} 嵌入 ({rate:.1f}s/图, "
                  f"剩余 {rate * (len(images) - k - 1) / 60:.0f} 分钟)", flush=True)
    arr = np.stack(embs)
    cache_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, embeds=arr)
    print(f"[embeds] eval: {arr.shape} -> {path}", flush=True)
    return arr


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--ckpt", default=None, help="bc_policy 的 checkpoint")
    ap.add_argument("--policies", default="bc_policy",
                    help="逗号分隔: bc_policy,single_click,random_doctor,"
                         "simulated_doctor,multi_doctor")
    ap.add_argument("--multi_cand", type=int, default=None,
                    help="multi_doctor 每步候选数 (默认取 doctor.multi_cand)")
    ap.add_argument("--out", required=True)
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
    cache_dir = cfg.resolve("outputs") / f"cache_{ds_tag}"

    seg = make_segmenter(cfg)
    max_clicks = int(ev["max_clicks"])
    init_mode = cfg.raw["doctor"].get("init_mode", "click")
    grid, size = int(d["grid"]), int(d["image_size"])
    stop_guard = bool(ev.get("stop_guard", False))

    policies = [p.strip() for p in args.policies.split(",") if p.strip()]
    net, embeds_eval = None, None
    if "bc_policy" in policies:
        assert args.ckpt, "含 bc_policy 时必须给 --ckpt"
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
        if int(payload.get("embed_ch", 0)) > 0:
            embeds_eval = _embedding_cache(cfg, seg, images, cache_dir)
            if embeds_eval is None:
                print("[embeds] 警告: 后端无嵌入, BC 将在无嵌入分支下评测 (偏低估)")

    def _factory(name):
        if name == "bc_policy":
            def f(gt, rng, i=0):
                emb = None if embeds_eval is None else embeds_eval[i]
                return _BCAdapter(net, max_clicks, grid, size, init_mode,
                                  embed=emb, stop_guard=stop_guard)
            return f
        if name == "single_click":
            return lambda gt, rng, i=0: SingleClickDoctor(cfg, gt, rng)
        if name == "random_doctor":
            return lambda gt, rng, i=0: RandomDoctor(cfg, gt, rng)
        if name == "simulated_doctor":
            return lambda gt, rng, i=0: DoctorPolicy(cfg, gt, rng)
        if name == "multi_doctor":
            mc = args.multi_cand or int(cfg.raw["doctor"].get("multi_cand", 8))
            return lambda gt, rng, i=0: MultiCandidateDoctor(
                cfg, gt, rng, segmenter=seg, n_cand=mc)
        if name == "multi_doctor_ns":   # 对照: 多样候选但不打分 (拆解选择 vs 多样性)
            mc = args.multi_cand or int(cfg.raw["doctor"].get("multi_cand", 8))
            return lambda gt, rng, i=0: MultiCandidateDoctor(
                cfg, gt, rng, segmenter=None, n_cand=mc, select=False)
        raise ValueError(f"未知策略: {name}")

    out = {"config": args.config, "ckpt": args.ckpt, "n_eval": n_eval,
           "max_clicks": max_clicks, "targets": list(TARGETS),
           "area_frac": area.tolist(), "episodes": {}, "metrics": {}}
    q1, q2 = np.quantile(area, [1 / 3, 2 / 3])
    bin_ids = np.digitize(area, [q1, q2])
    bin_names = list(ev["size_bins"].keys())

    for name in policies:
        t0 = time.time()
        eps, _ = run_episodes(seg, images, masks, _factory(name), cfg)
        curves = dice_curves(eps, max_clicks)
        out["episodes"][name] = {
            "dices": [e["dices"] for e in eps],
            "clicks": [[(int(s), float(y), float(x), bool(k == BOX_KIND))
                        for (s, y, x, k) in e["clicks"]] for e in eps],
            "stop": [e["stop"] for e in eps],
        }
        m = {"dice_curve": curves.mean(axis=0).round(4).tolist(),
             "eval_seconds": round(time.time() - t0, 1)}
        for t in TARGETS:
            k_need, ok = clicks_to_target(curves, t)
            m[f"attc@{t}"] = round(float(k_need.mean()), 3)
            m[f"success@{t}"] = round(float(ok.mean()), 4)
        sb = {}
        for bi, bn in enumerate(bin_names):
            sel = bin_ids == bi
            if sel.sum():
                k_need, ok = clicks_to_target(curves[sel], 0.85)
                sb[bn] = {"n": int(sel.sum()), "dice_at_3": round(float(curves[sel, 3].mean()), 4),
                          "success@0.85": round(float(ok.mean()), 4)}
        m["size_bins"] = sb
        out["metrics"][name] = m
        print(f"[eval] {name}: Dice@1={m['dice_curve'][1]:.3f} "
              f"Dice@3={m['dice_curve'][3]:.3f} "
              f"成功@0.85={m['success@0.85']:.1%} ATTC@0.85={m['attc@0.85']:.2f} "
              f"({m['eval_seconds']}s)", flush=True)

    out_path = cfg.resolve("outputs") / Path(args.out).name if not Path(args.out).is_absolute() else Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False)
    print(f"[saved] {out_path}", flush=True)


if __name__ == "__main__":
    main()
