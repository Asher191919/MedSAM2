"""⑤ 配对统计: McNemar 精确检验 + 目标 Dice 敏感性 (ATTC/成功率 @0.80/0.85/0.90)。

读取 eval_flexible.py 保存的每回合 JSON (同一批图像、同一个初始抖动框),
不重跑回合。McNemar 用精确二项检验 (b/c 不一致对), 成功率差给 bootstrap CI。

用法:
  python scripts/eval_stats.py --episodes outputs/episodes_kvasir_rgb.json \
      --pairs bc_policy,single_click bc_policy,simulated_doctor \
      --out outputs/stats_kvasir_rgb.md
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np

TARGETS = (0.80, 0.85, 0.90)
N_BOOT = 10000


def _curves(episodes: dict, max_clicks: int) -> np.ndarray:
    """与 rollout.dice_curves 相同的停后保持对齐。"""
    out = np.zeros((len(episodes["dices"]), max_clicks + 1), dtype=np.float32)
    for i, curve in enumerate(episodes["dices"]):
        for k in range(max_clicks + 1):
            out[i, k] = curve[min(k, len(curve) - 1)]
    return out


def _success(curves: np.ndarray, target: float) -> np.ndarray:
    hit_any = (curves[:, 1:] >= target).any(axis=1)
    return hit_any


def _mcnemar_exact(b: int, c: int) -> float:
    """双侧精确 McNemar: 不一致对 b, c; p = 2*P(X <= min(b,c)), X~Bin(b+c, .5)。"""
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    tail = sum(math.comb(n, i) for i in range(0, k + 1)) / 2 ** n
    return float(min(1.0, 2.0 * tail))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", required=True)
    ap.add_argument("--pairs", nargs="+",
                    default=["bc_policy,single_click", "bc_policy,simulated_doctor"],
                    help="每对 'A,B' (A 相对 B 检验)")
    ap.add_argument("--targets", nargs="+", type=float, default=list(TARGETS))
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    data = json.load(open(args.episodes, encoding="utf-8"))
    max_clicks = int(data["max_clicks"])
    curves = {name: _curves(ep, max_clicks)
              for name, ep in data["episodes"].items()}
    n = next(iter(curves.values())).shape[0]
    print(f"图像数 n={n}, max_clicks={max_clicks}, 来源: {args.episodes}")

    lines = ["# 配对统计检验", "",
             f"> 来源: `{args.episodes}`; n={n}; McNemar 精确检验 (双侧); "
             f"成功率差的 95% CI 为 {N_BOOT} 次 bootstrap (种子 0)。", ""]

    for pair in args.pairs:
        a_name, b_name = pair.split(",")
        if a_name not in curves or b_name not in curves:
            print(f"[skip] {pair}: 缺少策略 ({list(curves)})")
            continue
        ca, cb = curves[a_name], curves[b_name]
        lines.append(f"## {a_name} vs {b_name}")
        lines.append("")
        lines.append("| 目标 Dice | A 成功率 | B 成功率 | 差值 [95% CI] | "
                     "不一致对 (A赢/B赢) | McNemar p | A ATTC | B ATTC |")
        lines.append("|---|---|---|---|---|---|---|---|")
        for t in args.targets:
            sa, sb = _success(ca, t), _success(cb, t)
            b_win = int((sa & ~sb).sum())   # A 成功 B 失败
            c_win = int((~sa & sb).sum())   # B 成功 A 失败
            p = _mcnemar_exact(b_win, c_win)
            rng = np.random.default_rng(0)
            diffs = []
            for _ in range(N_BOOT):
                idx = rng.integers(0, n, n)
                diffs.append(sa[idx].mean() - sb[idx].mean())
            lo, hi = np.quantile(diffs, [0.025, 0.975])
            ka, oka = clicks_to_target_np(ca, t)
            kb, okb = clicks_to_target_np(cb, t)
            lines.append(
                f"| {t:.2f} | {sa.mean():.1%} | {sb.mean():.1%} | "
                f"{sa.mean() - sb.mean():+.1%} [{lo:+.1%}, {hi:+.1%}] | "
                f"{b_win}/{c_win} | {p:.4f} | {ka.mean():.2f} | {kb.mean():.2f} |")
        lines.append("")

    report = "\n".join(lines)
    print(report)
    if args.out:
        Path(args.out).write_text(report, encoding="utf-8")
        print(f"[saved] {args.out}")


def clicks_to_target_np(curves: np.ndarray, target: float):
    """与 rollout.clicks_to_target 相同 (本地复制避免拖入 torch 依赖)。"""
    N, K = curves.shape
    k_needed = np.full(N, K, dtype=np.float32)
    for i in range(N):
        hit = np.nonzero(curves[i, 1:] >= target)[0]
        if len(hit):
            k_needed[i] = hit[0] + 1
    return k_needed, k_needed < K


if __name__ == "__main__":
    main()
