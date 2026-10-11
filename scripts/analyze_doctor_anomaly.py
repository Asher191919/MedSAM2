"""⑦ 随机医生反常分析: 为什么随机修正反超规则医生? (ISIC: 97.6% vs 89.2%)

读取 eval_flexible.py 保存的配对回合 JSON, 逐图对比三个假设:
H1 冗余: 规则医生相邻步点击距离近 (最大误差域质心+3px 抖动 -> 提示高度相关),
   随机医生点击散布在整个误差带 -> 每次交互提供的新信息多;
H2 时机: 不一致回合里规则医生是不是提前停手 (no_error) 或卡死 (max_clicks);
H3 数量: 赢的回合用了更多次交互 (信息量换来的)。

用法: python scripts/analyze_doctor_anomaly.py --episodes outputs/zeroshot_isic.json
"""
from __future__ import annotations

import argparse
import json

import numpy as np


def _success(data: dict, name: str, target: float) -> np.ndarray:
    curves = np.array([[c[min(k, len(c) - 1)]
                        for k in range(data["max_clicks"] + 1)]
                       for c in data["episodes"][name]["dices"]])
    return (curves[:, 1:] >= target).any(axis=1)


def _geometry(data: dict, name: str, r_eq: np.ndarray):
    """每回合: 点击数 / 相邻点击间距 / 全回合两两散布 (均以肿瘤等效半径为单位)。"""
    eps = data["episodes"][name]["clicks"]
    n = len(eps)
    nk = np.array([sum(1 for c in ep if not c[3]) for ep in eps], dtype=float)
    step_d, spread = [], []
    for i, ep in enumerate(eps):
        ck = [(y, x) for (_, y, x, box) in ep if not box]
        if len(ck) >= 2:
            step_d.append(np.mean([np.hypot(ck[j][0] - ck[j - 1][0],
                                            ck[j][1] - ck[j - 1][1])
                                   for j in range(1, len(ck))]) / r_eq[i])
            dm = [np.hypot(p[0] - q[0], p[1] - q[1])
                  for j, p in enumerate(ck) for q in ck[j + 1:]]
            spread.append(np.mean(dm) / r_eq[i] if dm else 0.0)
    return nk, np.array(step_d), np.array(spread)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", required=True)
    ap.add_argument("--target", type=float, default=0.85)
    ap.add_argument("--a", default="random_doctor", help="反常中的强者")
    ap.add_argument("--b", default="simulated_doctor", help="反常中的弱者")
    args = ap.parse_args()

    data = json.load(open(args.episodes, encoding="utf-8"))
    area = np.array(data["area_frac"])
    r_eq = np.sqrt(area * 128.0 * 128.0 / np.pi) + 1e-6
    sa, sb = (_success(data, nm, args.target) for nm in (args.a, args.b))
    n = len(sa)

    da = np.array([c[-1] for c in data["episodes"][args.a]["dices"]])
    db = np.array([c[-1] for c in data["episodes"][args.b]["dices"]])
    print(f"n={n} 目标={args.target}")
    print(f"{args.a}: 成功 {sa.mean():.1%}, 终 Dice {da.mean():.3f}")
    print(f"{args.b}: 成功 {sb.mean():.1%}, 终 Dice {db.mean():.3f}")
    a_win, b_win = int((sa & ~sb).sum()), int((~sa & sb).sum())
    print(f"不一致对: {args.a} 赢 {a_win} / {args.b} 赢 {b_win}\n")

    print("== H1 点击冗余 (肿瘤等效半径为单位) ==")
    for name in (args.a, args.b):
        nk, sd, sp = _geometry(data, name, r_eq)
        print(f"[{name}] 平均点击 {nk.mean():.2f} | 相邻点击间距 "
              f"{sd.mean():.2f}±{sd.std():.2f} r_eq | 全回合散布 "
              f"{sp.mean():.2f}±{sp.std():.2f} r_eq")

    print("\n== H2/H3 不一致回合拆解 ==")
    eps_a, eps_b = data["episodes"][args.a], data["episodes"][args.b]
    for tag, idxs in ((f"{args.a} 独赢", np.nonzero(sa & ~sb)[0]),
                      (f"{args.b} 独赢", np.nonzero(~sa & sb)[0])):
        if len(idxs) == 0:
            print(f"== {tag}: 无 ==")
            continue
        nk_a = [sum(1 for c in eps_a["clicks"][i] if not c[3]) for i in idxs]
        nk_b = [sum(1 for c in eps_b["clicks"][i] if not c[3]) for i in idxs]
        stop_b = [eps_b["stop"][i] for i in idxs]
        stop_a = [eps_a["stop"][i] for i in idxs]
        fin_a = [eps_a["dices"][i][-1] for i in idxs]
        fin_b = [eps_b["dices"][i][-1] for i in idxs]
        print(f"== {tag} ({len(idxs)} 个) ==")
        print(f"  {args.a}: 点击 {np.mean(nk_a):.1f}, 终 Dice {np.mean(fin_a):.3f}, "
              f"停手 {sorted(set(stop_a))}")
        print(f"  {args.b}: 点击 {np.mean(nk_b):.1f}, 终 Dice {np.mean(fin_b):.3f}, "
              f"停手 {sorted(set(stop_b))} (max_clicks 占 "
              f"{stop_b.count('max_clicks')}/{len(idxs)}, "
              f"no_error 占 {stop_b.count('no_error')}/{len(idxs)})")


if __name__ == "__main__":
    main()
