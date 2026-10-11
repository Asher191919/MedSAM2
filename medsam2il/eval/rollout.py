"""交互回合执行器: 策略 x 分割后端 的完整标注回合 (轨迹收集与评测共用)。

双后端:
  - 批处理替身 (UNetSegmenter): 整批前向, CPU 高效;
  - 交互式后端 (MedSAM2Segmenter): 逐图 reset/add_box/add_click (编码只跑一次)。

协议 (doctor.init_mode):
  - "click": 策略自己完成第 0 步中心正点击 (合成数据, 可学习);
  - "box"  : 初始框选 = 人类用户给的提示 (所有策略共享同一个抖动 GT 框,
             模拟"医生先框住目标"), 策略从第 1 步开始学习修正 ——
             对应方案原话"训练模型模仿医生下一步如何修正肿瘤边界"。

动作: ("click", y, x, positive) | ("box", y0, x0, y1, x1) | "stop"。
日志约定: clicks 元组 (step, y, x, kind), kind: 1=正 0=负 2=框选(记框中心)。
"""
from __future__ import annotations

import time

import numpy as np

from medsam2il.policy.state import (N_STATE_CH, action_to_label, build_state,
                                    cell_of, pack_state, stop_label)
from medsam2il.segmenter.base import box_map, click_map, dice_binary
from medsam2il.segmenter.train_surrogate import _predict

BOX_KIND = 2


def _gt_box(gt: np.ndarray, rng: np.random.Generator, jitter: float
            ) -> tuple[int, int, int, int]:
    ys, xs = np.nonzero(gt)
    H, W = gt.shape
    clip = lambda v, hi: int(np.clip(v + rng.normal(0, jitter), 0, hi - 1))
    return (clip(ys.min(), H), clip(xs.min(), W),
            clip(ys.max() + 1, H), clip(xs.max() + 1, W))


def _new_book(N):
    return [{"dices": [0.0], "clicks": [], "stop": ""} for _ in range(N)]


def run_episodes(segmenter, images: np.ndarray, masks: np.ndarray,
                 policy_factory, cfg, record: bool = False,
                 labeler_factory=None) -> tuple[list[dict], dict]:
    """labeler_factory: DAgger 模式下提供专家标签 (执行用 policy, 标签用 labeler)。"""
    if hasattr(segmenter, "predict_batch"):
        return _run_batched(segmenter.model, images, masks, policy_factory, cfg,
                            record, labeler_factory)
    return _run_interactive(segmenter, images, masks, policy_factory, cfg,
                            record, labeler_factory)


def _record_sample(rec, state, label, ep, step, state_dice: float = 0.0,
                   pos_cell: int = -1, neg_cell: int = -1):
    rec["states"].append(state)
    rec["labels"].append(label)
    rec["episode"].append(ep)
    rec["step"].append(step)
    rec.setdefault("state_dice", []).append(state_dice)
    rec.setdefault("pos_cell", []).append(pos_cell)
    rec.setdefault("neg_cell", []).append(neg_cell)


def _finish_record(rec, size):
    n_ch = rec["states"][0].shape[0] if rec["states"] else N_STATE_CH
    out = {
        "states": np.stack(rec["states"]) if rec["states"] else
        np.zeros((0, n_ch, size, size), dtype=np.uint8),
        "labels": np.array(rec["labels"], dtype=np.int32),
        "episode": np.array(rec["episode"], dtype=np.int32),
        "step": np.array(rec["step"], dtype=np.int32),
    }
    if rec.get("state_dice"):
        out["state_dice"] = np.array(rec["state_dice"], dtype=np.float32)
    if rec.get("pos_cell"):
        out["pos_cell"] = np.array(rec["pos_cell"], dtype=np.int32)
        out["neg_cell"] = np.array(rec["neg_cell"], dtype=np.int32)
    return out


def _box_mode(cfg) -> bool:
    return cfg.raw["doctor"].get("init_mode", "click") == "box"


# ------------------------------------------------------------- 批处理替身 ----
def _run_batched(model, images, masks, policy_factory, cfg, record,
                 labeler_factory=None):
    d, su, ev = cfg.raw["data"], cfg.raw["surrogate"], cfg.raw["eval"]
    doc = cfg.raw["doctor"]
    size, grid = int(d["image_size"]), int(d["grid"])
    sigma = float(su["sigma_click"])
    max_clicks = int(ev["max_clicks"])
    box_init = _box_mode(cfg)
    N = len(images)
    seed0 = cfg.seed * 100003 + (17 if record else 0)

    pos = np.zeros((N, size, size), dtype=np.float32)
    neg = np.zeros((N, size, size), dtype=np.float32)
    zeros = np.zeros((N, size, size), dtype=np.float32)
    prob = np.zeros((N, size, size), dtype=np.float32)
    active = np.ones(N, dtype=bool)
    policies = [policy_factory(masks[i].astype(bool),
                               np.random.default_rng(seed0 + i), i)
                for i in range(N)]
    labelers = ([labeler_factory(masks[i].astype(bool),
                                 np.random.default_rng(seed0 + 7 + i), i)
                 for i in range(N)] if labeler_factory else None)
    episodes = _new_book(N)
    rec = {"states": [], "labels": [], "episode": [], "step": []}

    if box_init:
        # 共享的人类初始提示: 抖动 GT 框 (替身不支持框 -> 质心正点击兜底)
        for i in range(N):
            rng_i = np.random.default_rng(seed0 + i)
            y0, x0, y1, x1 = _gt_box(masks[i].astype(bool), rng_i,
                                     float(doc["click_jitter"]))
            prob[i] = _predict(model, images[i:i + 1],
                               box_map(size, (y0, x0, y1, x1))[None],
                               neg[i:i + 1], zeros[i:i + 1], cfg.device)[0]
            episodes[i]["dices"].append(dice_binary(prob[i] > 0.5, masks[i]))
            episodes[i]["clicks"].append((0, (y0 + y1) // 2, (x0 + x1) // 2, BOX_KIND))
        start_step = 1
    else:
        start_step = 0

    for step in range(start_step, max_clicks + 1):
        act_idx = np.nonzero(active)[0]
        if len(act_idx) == 0:
            break
        pending = []
        for i in act_idx:
            conf_i = float(prob[i].max())
            a = policies[i].act(images[i], prob[i], pos[i], neg[i], step,
                                conf=conf_i, embed=None)
            if a == "stop":
                if record:
                    _record_sample(rec, pack_state(build_state(
                        images[i], prob[i], pos[i], neg[i], step, max_clicks,
                        conf_i)), stop_label(grid), i, step,
                        float(episodes[i]["dices"][-1]))
                episodes[i]["stop"] = getattr(policies[i], "stop_reason", "policy_stop")
                active[i] = False
            elif a[0] == "box":   # 框模式起步后策略不应再框; 兜底转中心点击
                _, y0, x0, y1, x1 = a
                pending.append((i, ("click", (y0 + y1) // 2, (x0 + x1) // 2, True)))
            elif a[0] == "interact":   # 双修正: (pos_xy|None, neg_xy|None)
                pending.append((i, a))
            else:
                pending.append((i, a))
        if not pending:
            break
        for i, act_i in pending:
            if act_i[0] == "interact":   # 双修正: 展开为一正一负两个点击
                _, pxy, nxy = act_i
                clicks_i = ([] + ([(*pxy, True)] if pxy else [])
                            + ([(*nxy, False)] if nxy else []))
            else:
                clicks_i = [(*act_i[1:3], act_i[3])]
            if record:
                conf_i = float(prob[i].max())
                lab_a = labelers[i].act(images[i], prob[i], pos[i], neg[i], step,
                                        conf=conf_i, embed=None) if labelers else None
                if lab_a == "stop":
                    lab = stop_label(grid)
                elif lab_a is not None and lab_a[0] == "interact":
                    fn_n = ((masks[i] > 0) & (prob[i] <= 0.5)).sum()
                    fp_n = ((prob[i] > 0.5) & (masks[i] == 0)).sum()
                    if lab_a[1] is not None and (lab_a[2] is None or fn_n >= fp_n):
                        lab = action_to_label(grid, *cell_of(lab_a[1][0],
                                                             lab_a[1][1],
                                                             grid, size), True)
                    else:
                        lab = action_to_label(grid, *cell_of(lab_a[2][0],
                                                             lab_a[2][1],
                                                             grid, size), False)
                elif lab_a is not None:
                    lab = action_to_label(grid, *cell_of(lab_a[1], lab_a[2], grid,
                                                         size), lab_a[3])
                else:
                    lab = action_to_label(grid, *cell_of(clicks_i[0][0],
                                                         clicks_i[0][1], grid, size),
                                          clicks_i[0][2])
                pc = next(((lambda c: c[0] * grid + c[1])(
                    cell_of(c[0], c[1], grid, size)) for c in clicks_i if c[2]), -1)
                nc = next(((lambda c: c[0] * grid + c[1])(
                    cell_of(c[0], c[1], grid, size)) for c in clicks_i
                    if not c[2]), -1)
                _record_sample(rec, pack_state(build_state(
                    images[i], prob[i], pos[i], neg[i], step, max_clicks, conf_i)),
                    lab, i, step, float(episodes[i]["dices"][-1]),
                    pos_cell=pc, neg_cell=nc)
            for (y, x, positive) in clicks_i:
                m = click_map(size, (y, x), sigma)
                if positive:
                    pos[i] = np.maximum(pos[i], m)
                else:
                    neg[i] = np.maximum(neg[i], m)
                episodes[i]["clicks"].append((int(step), int(y), int(x),
                                               int(bool(positive))))

        idxs = [i for i, _ in pending]
        p = _predict(model, images[idxs], pos[idxs], neg[idxs], zeros[idxs], cfg.device)
        for j, i in enumerate(idxs):
            prob[i] = p[j]
            episodes[i]["dices"].append(dice_binary(p[j] > 0.5, masks[i]))
            if step + 1 > max_clicks:
                active[i] = False
                episodes[i]["stop"] = "max_clicks"
    return episodes, _finish_record(rec, size)


# --------------------------------------------------- 交互式后端 (MedSAM2) ----
def _run_interactive(segmenter, images, masks, policy_factory, cfg, record,
                     labeler_factory=None):
    d, ev = cfg.raw["data"], cfg.raw["eval"]
    doc = cfg.raw["doctor"]
    size, grid = int(d["image_size"]), int(d["grid"])
    sigma = float(cfg.raw["surrogate"]["sigma_click"])
    max_clicks = int(ev["max_clicks"])
    n_cand = int(ev.get("candidates", 1))       # 测试时候选搜索: 1=关
    cand_sep = int(ev.get("candidate_sep", 2))  # 候选间最小网格距离
    box_init = _box_mode(cfg)
    N = len(images)
    seed0 = cfg.seed * 100003 + (17 if record else 0)

    episodes = _new_book(N)
    rec = {"states": [], "labels": [], "episode": [], "step": []}
    t0 = time.time()

    for i in range(N):
        rng_i = np.random.default_rng(seed0 + i)
        pol = policy_factory(masks[i].astype(bool), rng_i, i)
        labeler = (labeler_factory(masks[i].astype(bool),
                                   np.random.default_rng(seed0 + 7 + i), i)
                   if labeler_factory else None)
        segmenter.reset(images[i])
        pos = np.zeros((size, size), dtype=np.float32)
        neg = np.zeros((size, size), dtype=np.float32)
        ep = episodes[i]
        start_step = 0
        if box_init:
            # 共享的人类初始提示: 抖动 GT 框 (不进训练样本 —— 不是模型的决策)
            y0, x0, y1, x1 = _gt_box(masks[i].astype(bool), rng_i,
                                     float(doc["click_jitter"]))
            segmenter.add_box(y0, x0, y1, x1)
            ep["dices"].append(dice_binary(segmenter.mask(), masks[i]))
            ep["clicks"].append((0, (y0 + y1) // 2, (x0 + x1) // 2, BOX_KIND))
            start_step = 1
        emb_i = segmenter.embedding()
        for step in range(start_step, max_clicks + 1):
            conf_i = segmenter.confidence
            hyp = segmenter.hypotheses()
            if hyp is not None and len(hyp[0]) > 1:
                dis_i = hyp[0].std(axis=0)
                sstd_i = float(hyp[1].std())
            else:
                dis_i, sstd_i = None, 0.0
            a = pol.act(images[i], segmenter.prob, pos, neg, step,
                        conf=conf_i, embed=emb_i, hyp_dis=dis_i,
                        score_std=sstd_i)
            # 测试时候选搜索: 策略提议 top-k, 分割器 predicted-IoU 选优后执行
            if a != "stop" and a[0] == "click" and n_cand > 1 and \
                    hasattr(pol, "candidates") and \
                    getattr(segmenter, "supports_rollback", False):
                cands = pol.candidates(images[i], segmenter.prob, pos, neg, step,
                                       conf=conf_i, embed=emb_i, hyp_dis=dis_i,
                                       score_std=sstd_i, k=n_cand, sep=cand_sep)
                if len(cands) > 1:
                    best_s, best_a = -np.inf, None
                    for c in cands:
                        s = segmenter.try_click(int(c[1]), int(c[2]), c[3])
                        if s > best_s:
                            best_s, best_a = s, c
                        segmenter.pop_click()
                    if best_a is not None:
                        a = best_a
            if a == "stop":
                if record:
                    _record_sample(rec, pack_state(build_state(
                        images[i], segmenter.prob, pos, neg, step, max_clicks,
                        conf_i, dis_i, sstd_i)), stop_label(grid), i, step)
                ep["stop"] = getattr(pol, "stop_reason", "policy_stop")
                break
            if record:
                lab_a = labeler.act(images[i], segmenter.prob, pos, neg, step,
                                    conf=conf_i, embed=emb_i, hyp_dis=dis_i,
                                    score_std=sstd_i) if labeler else None
                if lab_a == "stop":
                    lab = stop_label(grid)
                elif lab_a is not None and lab_a[0] == "interact":
                    fn_n = ((masks[i] > 0) & (segmenter.prob <= 0.5)).sum()
                    fp_n = ((segmenter.prob > 0.5) & (masks[i] == 0)).sum()
                    if lab_a[1] is not None and (lab_a[2] is None or fn_n >= fp_n):
                        lab = action_to_label(grid, *cell_of(lab_a[1][0],
                                                             lab_a[1][1],
                                                             grid, size), True)
                    else:
                        lab = action_to_label(grid, *cell_of(lab_a[2][0],
                                                             lab_a[2][1],
                                                             grid, size), False)
                elif lab_a is not None:
                    lab = action_to_label(grid, *cell_of(lab_a[1], lab_a[2], grid,
                                                         size), lab_a[3])
                elif a[0] == "interact":
                    # 主标签 = 误差较大一侧的点击类型 (与医生优先级一致)
                    fn_n = ((masks[i] > 0) & (segmenter.prob <= 0.5)).sum()
                    fp_n = ((segmenter.prob > 0.5) & (masks[i] == 0)).sum()
                    if a[1] is not None and (a[2] is None or fn_n >= fp_n):
                        lab = action_to_label(grid, *cell_of(a[1][0], a[1][1],
                                                             grid, size), True)
                    else:
                        lab = action_to_label(grid, *cell_of(a[2][0], a[2][1],
                                                             grid, size), False)
                else:
                    lab = action_to_label(grid, *cell_of(a[1], a[2], grid, size),
                                          a[3])
                _record_sample(rec, pack_state(build_state(
                    images[i], segmenter.prob, pos, neg, step, max_clicks,
                    conf_i, dis_i, sstd_i)), lab, i, step,
                    float(ep["dices"][-1]),
                    pos_cell=(lambda c: c[0] * grid + c[1])(
                        cell_of(a[1][0], a[1][1], grid, size))
                    if a[0] == "interact" and a[1] else -1,
                    neg_cell=(lambda c: c[0] * grid + c[1])(
                        cell_of(a[2][0], a[2][1], grid, size))
                    if a[0] == "interact" and a[2] else -1)
            if a[0] == "interact":   # 双修正: 两个点击, 分割器重推理一次
                _, pxy, nxy = a
                for item, positive in ((pxy, True), (nxy, False)):
                    if item is None:
                        continue
                    yy, xx = item
                    m = click_map(size, (yy, xx), sigma)
                    if positive:
                        pos = np.maximum(pos, m)
                    else:
                        neg = np.maximum(neg, m)
                    segmenter.add_click(int(yy), int(xx), bool(positive))
                    ep["clicks"].append((int(step), int(yy), int(xx),
                                         int(bool(positive))))
            else:
                _, y, x, positive = a
                m = click_map(size, (y, x), sigma)
                if positive:
                    pos = np.maximum(pos, m)
                else:
                    neg = np.maximum(neg, m)
                segmenter.add_click(int(y), int(x), bool(positive))
                ep["clicks"].append((int(step), int(y), int(x),
                                     int(bool(positive))))
            ep["dices"].append(dice_binary(segmenter.mask(), masks[i]))
            if step >= max_clicks:
                ep["stop"] = "max_clicks"
                break
        if (i + 1) % 50 == 0:
            rate = (time.time() - t0) / (i + 1)
            print(f"    ... {i + 1}/{N} 图 ({rate:.1f}s/图, 预计剩余 "
                  f"{rate * (N - i - 1) / 60:.0f} 分钟)")
    return episodes, _finish_record(rec, size)


def dice_curves(episodes: list[dict], max_clicks: int) -> np.ndarray:
    """每步 Dice 矩阵 [N, max_clicks+1], 停止后保持最后值 (曲线对齐)。"""
    N = len(episodes)
    out = np.zeros((N, max_clicks + 1), dtype=np.float32)
    for i, ep in enumerate(episodes):
        curve = ep["dices"]
        for k in range(max_clicks + 1):
            out[i, k] = curve[min(k, len(curve) - 1)]
    return out


def clicks_to_target(curves: np.ndarray, target: float) -> tuple[np.ndarray, np.ndarray]:
    """达到 target 所需交互数 (含初始框); 未达到 -> max_clicks+1 (惩罚)。"""
    N, K = curves.shape
    k_needed = np.full(N, K, dtype=np.float32)
    success = np.zeros(N, dtype=bool)
    for i in range(N):
        hit = np.nonzero(curves[i, 1:] >= target)[0]
        if len(hit):
            k_needed[i] = hit[0] + 1
            success[i] = True
    return k_needed, success
