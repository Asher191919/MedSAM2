"""评测 + 报告: 对照方案的四条指标。

1. 达到目标 Dice 所需点击次数 (ATTC, 含截断惩罚与成功率)
2. 每次交互后的 Dice 提升 (Dice-vs-clicks 曲线 + 平均边际增益)
3. 不同肿瘤大小下的稳定性 (按肿瘤面积三分位分箱)
4. 医生与模型的交互策略差异 (点击类型比/位置分布/停手时机)
"""
from __future__ import annotations

import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

from medsam2il.eval.rollout import (BOX_KIND, clicks_to_target, dice_curves,  # noqa: E402
                                    run_episodes)
from medsam2il.policy.doctor import (DoctorPolicy, RandomDoctor,  # noqa: E402
                                     SingleClickDoctor)
from medsam2il.policy.state import build_state, cell_center  # noqa: E402

plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "PingFang SC",
                                   "Noto Sans CJK SC", "DejaVu Sans"]


class _BCAdapter:
    """把策略网络输出转成执行动作; init_mode=box 时第 0 步用自身热图推断框位置。

    第 0 步 (空状态): 正通道热图 ≈ 肿瘤掩码置信 (掩码监督), 取 >=0.5*max 区域
    的外接框作为框选提示 —— 不用 GT, 是策略自己的定位信念。
    """

    def __init__(self, net, max_clicks: int, grid: int, size: int, init_mode: str,
                 embed: np.ndarray | None = None, stop_guard: bool = False):
        self.net = net
        self.max_clicks = max_clicks
        self.grid = grid
        self.size = size
        self.init_mode = init_mode
        self.embed = embed            # 本回合图像的 SAM2 嵌入 [C,16,16] 或 None
        self.stop_guard = stop_guard  # 系统护栏: 连续 2 次自信度无提升则停
        self._conf_hist: list[float] = []
        self.stop_reason = "policy_stop"

    def _step0_box(self, image, prob, pos_map, neg_map):
        s = build_state(image, prob, pos_map, neg_map, 0, self.max_clicks)
        with torch.no_grad():
            heat = self.net.forward(torch.from_numpy(s[None]))[0]
        p = torch.sigmoid(heat[0])[0].numpy()
        cell = self.size / self.grid
        ys, xs = np.nonzero(p >= 0.5 * p.max())
        if len(ys) == 0:
            cy, cx = np.unravel_index(int(p.argmax()), p.shape)
            y, x = cell_center(int(cy), int(cx), self.grid, self.size)
            return "click", y, x, True
        y0 = int(np.clip(ys.min() - 1, 0, self.grid)) * cell
        x0 = int(np.clip(xs.min() - 1, 0, self.grid)) * cell
        y1 = min(int(np.clip(ys.max() + 2, 0, self.grid)) * cell, self.size - 1)
        x1 = min(int(np.clip(xs.max() + 2, 0, self.grid)) * cell, self.size - 1)
        return "box", int(y0), int(x0), int(y1), int(x1)

    def act(self, image, prob, pos_map, neg_map, step, conf: float = 0.0,
            embed=None, hyp_dis=None, score_std: float = 0.0, max_clicks=None):
        if step == 0 and self.init_mode == "box":
            return self._step0_box(image, prob, pos_map, neg_map)
        # 停手护栏 (不依赖 GT): 连续两次交互后自信度仍无提升 -> 停止
        self._conf_hist.append(float(conf))
        if self.stop_guard and step >= 3 and len(self._conf_hist) >= 3:
            recent = self._conf_hist[-3:]
            if recent[-1] <= recent[-2] <= recent[-3]:
                self.stop_reason = "stop_guard"
                return "stop"
        a = self.net.act(image, prob, pos_map, neg_map, step, self.max_clicks,
                         conf, embed if embed is not None else self.embed,
                         hyp_dis, score_std)
        if a == "stop":
            return a
        if a[0] == "interact":   # 双修正: 主/次网格单元 -> 像素坐标, 按类型归位
            _, pc, nc, positive = a
            py, px = cell_center(pc[0], pc[1], self.grid, self.size)
            if positive:
                neg = None if nc is None else cell_center(nc[0], nc[1],
                                                          self.grid, self.size)
                return "interact", (py, px), neg
            pos = None if nc is None else cell_center(nc[0], nc[1],
                                                      self.grid, self.size)
            return "interact", pos, (py, px)
        _, cy, cx, positive = a
        y, x = cell_center(cy, cx, self.grid, self.size)
        # 亚格子偏移 (改进3): 偏移头把"格子中心"细化到格内 4px 级
        if getattr(self.net, "fine_offset", False):
            dy, dx = getattr(self.net, "_last_offset", (0.0, 0.0))
            cell_px = self.size / self.grid
            y = float(np.clip(y + dy * cell_px, 0, self.size - 1))
            x = float(np.clip(x + dx * cell_px, 0, self.size - 1))
        return "click", y, x, positive

    def candidates(self, image, prob, pos_map, neg_map, step, conf: float = 0.0,
                   embed=None, hyp_dis=None, score_std: float = 0.0, k: int = 3,
                   sep: int = 2):
        """测试时候选搜索: 返回像素坐标的候选动作列表 (策略提议, 分割器打分)。"""
        if step == 0 and self.init_mode == "box":
            return []
        if self.net.act(image, prob, pos_map, neg_map, step, self.max_clicks,
                        conf, embed if embed is not None else self.embed,
                        hyp_dis, score_std) == "stop":
            return []
        cands = self.net.top_candidates(image, prob, pos_map, neg_map, step,
                                        self.max_clicks, conf,
                                        embed if embed is not None else self.embed,
                                        hyp_dis, score_std, k=k, sep=sep)
        return [("click", *cell_center(cy, cx, self.grid, self.size), positive)
                for (cy, cx, positive, _) in cands]


POLICY_ORDER = ["single_click", "random_doctor", "simulated_doctor", "bc_policy"]
POLICY_NAMES = {"single_click": "仅单次交互", "random_doctor": "随机修正",
                "simulated_doctor": "模拟医生", "bc_policy": "模仿学习策略 (BC)"}


def evaluate_policies(cfg, segmenter, images, masks, area_frac, bc_net,
                      embeds_eval=None) -> dict:
    ev = cfg.raw["eval"]
    target = float(ev["target_dice"])
    max_clicks = int(ev["max_clicks"])
    init_mode = cfg.raw["doctor"].get("init_mode", "click")
    stop_guard = bool(cfg.raw["eval"].get("stop_guard", False))

    def _bc_factory(gt, rng, i=0):
        emb = None if embeds_eval is None else embeds_eval[i]
        return _BCAdapter(bc_net, max_clicks, int(cfg.raw["data"]["grid"]),
                          int(cfg.raw["data"]["image_size"]), init_mode,
                          embed=emb, stop_guard=stop_guard)

    factories = {
        "single_click": lambda gt, rng, i=0: SingleClickDoctor(cfg, gt, rng),
        "random_doctor": lambda gt, rng, i=0: RandomDoctor(cfg, gt, rng),
        "simulated_doctor": lambda gt, rng, i=0: DoctorPolicy(cfg, gt, rng),
        "bc_policy": _bc_factory,
    }
    out = {"episodes": {}, "metrics": {}, "size_bins": {}, "strategy": {},
           "target_dice": target, "max_clicks": max_clicks, "init_mode": init_mode,
           "stop_guard": stop_guard, "per_episode_success": {}}

    q1, q2 = np.quantile(area_frac, [1 / 3, 2 / 3])
    bin_names = list(cfg.raw["eval"]["size_bins"].keys())
    bin_ids = np.digitize(area_frac, [q1, q2])  # 0 small, 1 medium, 2 large

    for name in POLICY_ORDER:
        eps, _ = run_episodes(segmenter, images, masks, factories[name], cfg)
        curves = dice_curves(eps, max_clicks)
        k_need, ok = clicks_to_target(curves, target)
        out["per_episode_success"][name] = ok.astype(bool).tolist()
        n_clicks = np.array([len(e["clicks"]) for e in eps], dtype=np.float32)
        all_clicks = [c for e in eps for c in e["clicks"]]
        click_only = [c for c in all_clicks if c[3] != BOX_KIND]
        pos_ratio = (float(np.mean([bool(c[3]) for c in click_only]))
                     if click_only else 0.0)
        box_ratio = (float(np.mean([c[3] == BOX_KIND for c in all_clicks]))
                     if all_clicks else 0.0)
        # 点击到肿瘤质心的归一化距离 (肿瘤等效半径为单位)
        dists = []
        for i, e in enumerate(eps):
            ys, xs = np.nonzero(masks[i])
            cy, cx = ys.mean(), xs.mean()
            r_eq = np.sqrt(masks[i].sum() / np.pi) + 1e-6
            for (_, y, x, _) in e["clicks"]:
                dists.append(np.hypot(y - cy, x - cx) / r_eq)
        stop_dice = curves[:, -1]
        m = {
            "dice_curve": curves.mean(axis=0).round(4).tolist(),
            "auc_dice_mean": float(curves[:, 1:].mean()),
            "attc_mean_censored": float(k_need.mean()),
            "attc_median_success": float(np.median(k_need[ok])) if ok.any() else None,
            "success_rate": float(ok.mean()),
            "marginal_gain_per_click": float(np.diff(curves, axis=1).mean()),
            "n_clicks_mean": float(n_clicks.mean()),
            "stop_dice_mean": float(stop_dice.mean()),
        }
        out["metrics"][name] = m
        out["episodes"][name] = eps
        # 分肿瘤大小
        sb = {}
        for bi, bn in enumerate(bin_names):
            sel = bin_ids == bi
            if sel.sum() == 0:
                continue
            d3 = curves[sel, min(3, max_clicks)]
            sb[bn] = {"n": int(sel.sum()),
                      "dice_at_3": float(d3.mean()),
                      "attc": float(k_need[sel].mean())}
        out["size_bins"][name] = sb
        # 策略差异要素
        out["strategy"][name] = {
            "pos_click_ratio": float(pos_ratio),
            "box_ratio": float(box_ratio),
            "click_norm_dist_mean": float(np.mean(dists)) if dists else 0.0,
            "stop_dice_std": float(stop_dice.std()),
        }
        print(f"[eval] {name}: Dice@1={m['dice_curve'][1]:.3f} "
              f"Dice@3={m['dice_curve'][3]:.3f} ATTC={m['attc_mean_censored']:.2f} "
              f"成功率={m['success_rate']:.1%}")
    out["size_quantiles"] = [float(q1), float(q2)]
    out["images"] = images
    out["masks"] = masks
    return out


# ------------------------------------------------------------------ 图 ----
def _plot_curves(res, out_path):
    plt.figure(figsize=(7, 4.5))
    xs = np.arange(res["max_clicks"] + 1)
    for name in POLICY_ORDER:
        plt.plot(xs, res["metrics"][name]["dice_curve"], marker="o", ms=4,
                 label=POLICY_NAMES[name])
    plt.axhline(res["target_dice"], ls="--", c="gray", lw=1,
                label=f"目标 Dice={res['target_dice']}")
    plt.xlabel("交互次数 (点击数)")
    plt.ylabel("Dice")
    plt.title("每次交互后的 Dice 提升 (test)")
    plt.ylim(0, 1.02)
    plt.grid(alpha=0.3)
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def _plot_size(res, out_path):
    bins = list(next(iter(res["size_bins"].values())).keys())
    x = np.arange(len(bins))
    w = 0.2
    plt.figure(figsize=(7, 4.5))
    for i, name in enumerate(POLICY_ORDER):
        vals = [res["size_bins"][name].get(b, {}).get("dice_at_3", np.nan)
                for b in bins]
        plt.bar(x + (i - 1.5) * w, vals, w, label=POLICY_NAMES[name])
    plt.xticks(x, [f"{b}\n(n={next(iter(res['size_bins'].values()))[b]['n']})"
                   for b in bins])
    plt.ylabel("点击 3 次后的 Dice")
    plt.title("不同肿瘤大小下的稳定性 (Dice@3)")
    plt.ylim(0, 1.02)
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def _plot_strategy(res, out_path):
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    names = ["simulated_doctor", "bc_policy"]
    # (a) 点击类型占比
    pos = [res["strategy"][n]["pos_click_ratio"] for n in names]
    axes[0].bar([0, 1], pos, 0.5, color=["#4C72B0", "#DD8452"])
    axes[0].bar([0, 1], [1 - p for p in pos], 0.5, bottom=pos,
                color=["#4C72B0", "#DD8452"], alpha=0.35)
    axes[0].set_xticks([0, 1], ["模拟医生", "BC 策略"])
    axes[0].set_ylabel("占比")
    axes[0].set_title("点击类型: 正点击(实) vs 负点击/擦除(浅)")
    for i, p in enumerate(pos):
        axes[0].text(i, p + 0.02, f"{p:.1%}", ha="center", fontsize=9)
    # (b) 停手时 Dice 分布
    for i, n in enumerate(names):
        sd = [e["dices"][-1] for e in res["episodes"][n]]
        axes[1].hist(sd, bins=20, alpha=0.55, label=POLICY_NAMES[n],
                     color=["#4C72B0", "#DD8452"][i])
    axes[1].axvline(res["target_dice"], ls="--", c="gray", lw=1)
    axes[1].set_xlabel("停手时 Dice")
    axes[1].set_ylabel("回合数")
    axes[1].set_title("停手时机: 医生 vs 模型")
    axes[1].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def _plot_trajectories(res, images, masks, out_path, k=3):
    """选 小/中/大 各 1 例, 对比 医生 vs BC 的点击轨迹。"""
    area = masks.reshape(len(masks), -1).mean(axis=1)
    order = np.argsort(area)
    picks = [order[len(order) // 6], order[len(order) // 2], order[-len(order) // 6]]
    fig, axes = plt.subplots(k, 2, figsize=(6, 3 * k))
    for r, idx in enumerate(picks):
        for c, name in enumerate(["simulated_doctor", "bc_policy"]):
            ax = axes[r, c]
            ep = res["episodes"][name][idx]
            ax.imshow(images[idx], cmap="gray")
            ax.contour(masks[idx], levels=[0.5], colors=["red"], linewidths=1.2)
            for (_, y, x, kind) in ep["clicks"]:
                if kind == BOX_KIND:      # 框选: 画方框 (记录的是框中心)
                    ax.add_patch(plt.Rectangle((x - 16, y - 16), 32, 32,
                                               fill=False, ec="gold", lw=2))
                    ax.plot(x, y, marker="s", ms=6, color="gold")
                else:
                    ax.plot(x, y, marker="+" if kind else "_", ms=12,
                            color="deepskyblue" if kind else "gold",
                            markeredgewidth=2.4)
            ax.set_title(f"{POLICY_NAMES[name]} | 点击{len(ep['clicks'])}次 "
                         f"Dice={ep['dices'][-1]:.2f}", fontsize=9)
            ax.axis("off")
    fig.suptitle("交互轨迹对比: 红=肿瘤GT边界, 金框=初始框选(人类提示), "
                 "蓝+=正点击, 黄-=擦除", fontsize=10)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


# ----------------------------------------------------------------- 报告 ----
def _boot_ci(success, n_boot: int = 2000, seed: int = 0) -> tuple[float, float]:
    """成功率的 bootstrap 95% 置信区间。"""
    rng = np.random.default_rng(seed)
    arr = np.asarray(success, dtype=float)
    boots = [rng.choice(arr, len(arr), replace=True).mean() for _ in range(n_boot)]
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return float(lo), float(hi)


def write_report(cfg, res: dict, n_eval: int, tag: str = "") -> None:
    out_dir = cfg.resolve("outputs")
    out_dir.mkdir(parents=True, exist_ok=True)
    def _pf(name):
        return out_dir / (f"plot_{tag}_{name}.png" if tag else f"plot_{name}.png")
    _plot_curves(res, _pf("dice_vs_clicks"))
    _plot_size(res, _pf("size_stability"))
    _plot_strategy(res, _pf("strategy"))
    _plot_trajectories(res, res["images"], res["masks"], _pf("trajectories"))

    L = []
    L.append("# MedSAM2 + 交互式肿瘤分割模仿学习 — 评估报告\n")
    init_note = ("初始交互 = 框选 (真实内镜标注工作流)" if res.get("init_mode") == "box"
                 else "初始交互 = 中心正点击")
    guard = "含停手护栏; " if res.get("stop_guard") else ""
    L.append(f"> 测试回合: {n_eval}; 目标 Dice ≥ {res['target_dice']}; "
             f"最多交互 {res['max_clicks']} 次; {init_note}; {guard}"
             f"ATTC 未达标按 {res['max_clicks'] + 1} 次计惩罚。\n")
    L.append("## 1. 达到目标 Dice 所需点击次数 (ATTC)\n")
    L.append("| 策略 | 平均点击(含惩罚) | 成功者中位点击 | 成功率 [95% CI] | 平均点击数 |")
    L.append("|---|---|---|---|---|")
    for n in POLICY_ORDER:
        m = res["metrics"][n]
        med = f"{m['attc_median_success']:.1f}" if m["attc_median_success"] else "-"
        lo, hi = _boot_ci(res["per_episode_success"][n])
        L.append(f"| {POLICY_NAMES[n]} | {m['attc_mean_censored']:.2f} | {med} "
                 f"| {m['success_rate']:.1%} [{lo:.1%}, {hi:.1%}] "
                 f"| {m['n_clicks_mean']:.2f} |")
    L.append("\n## 2. 每次交互后的 Dice 提升\n")
    L.append("| 策略 | Dice@1 | Dice@3 | Dice@5 | 平均边际增益/次 | 交互后平均Dice(AUC) |")
    L.append("|---|---|---|---|---|---|")
    for n in POLICY_ORDER:
        m = res["metrics"][n]
        c = m["dice_curve"]
        L.append(f"| {POLICY_NAMES[n]} | {c[1]:.3f} | {c[3]:.3f} | {c[5]:.3f} "
                 f"| {m['marginal_gain_per_click']:.3f} | {m['auc_dice_mean']:.3f} |")
    L.append(f"\n![dice曲线](plot_{tag}_dice_vs_clicks.png)\n")
    L.append("## 3. 不同肿瘤大小下的稳定性\n")
    q1, q2 = res["size_quantiles"]
    L.append(f"(按肿瘤面积三分位分箱: q33={q1:.4f}, q66={q2:.4f})\n")
    L.append("| 策略 | " + " | ".join(
        f"{b} Dice@3 / ATTC" for b in next(iter(res["size_bins"].values()))) + " |")
    L.append("|---|" + "---|" * len(next(iter(res["size_bins"].values()))))
    for n in POLICY_ORDER:
        cells = []
        for b, v in res["size_bins"][n].items():
            cells.append(f"{v['dice_at_3']:.3f} / {v['attc']:.2f}")
        L.append(f"| {POLICY_NAMES[n]} | " + " | ".join(cells) + " |")
    L.append(f"\n![大小稳定性](plot_{tag}_size_stability.png)\n")
    L.append("## 4. 医生与模型的交互策略差异\n")
    L.append("| 策略 | 正点击占比 | 框选占比 | 点击离肿瘤质心距离(等效半径) | 停手Dice标准差 | 停手Dice均值 |")
    L.append("|---|---|---|---|---|---|")
    for n in POLICY_ORDER:
        s = res["strategy"][n]
        L.append(f"| {POLICY_NAMES[n]} | {s['pos_click_ratio']:.1%} "
                 f"| {s.get('box_ratio', 0):.1%} "
                 f"| {s['click_norm_dist_mean']:.2f} | {s['stop_dice_std']:.3f} "
                 f"| {res['metrics'][n]['stop_dice_mean']:.3f} |")
    L.append(f"\n![策略差异](plot_{tag}_strategy.png)\n")
    L.append("## 5. 轨迹可视化\n")
    L.append(f"![轨迹](plot_{tag}_trajectories.png)\n")
    report_name = f"report_{tag}.md" if tag else "report_medsam2.md"
    with open(out_dir / report_name, "w", encoding="utf-8") as f:
        f.write("\n".join(L))
    print(f"[report] -> {out_dir / 'report_medsam2.md'}")


def save_results(cfg, res: dict, tag: str = "") -> None:
    out_dir = cfg.resolve("outputs")

    def _clean(o):
        if isinstance(o, dict):
            return {str(k): _clean(v) for k, v in o.items()}
        if isinstance(o, (list, tuple)):
            return [_clean(v) for v in o]
        if isinstance(o, np.ndarray):
            return _clean(o.tolist())
        if isinstance(o, (np.floating, float)):
            f = float(o)
            return f if np.isfinite(f) else None
        if isinstance(o, (np.integer, int)):
            return int(o)
        return o

    slim = {k: v for k, v in res.items() if k not in ("episodes", "images", "masks")}
    res_name = f"eval_results_{tag}.json" if tag else "eval_results.json"
    with open(out_dir / res_name, "w", encoding="utf-8") as f:
        json.dump(_clean(slim), f, ensure_ascii=False, indent=2)
