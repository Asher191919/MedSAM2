"""专家轨迹收集: 模拟医生在分割后端上标注 (expert 模式),
或 DAgger 混合执行 (医生+BC) 且标签始终来自医生 (dagger 模式)。"""
from __future__ import annotations

import numpy as np

from medsam2il.eval.report import _BCAdapter
from medsam2il.eval.rollout import run_episodes
from medsam2il.policy.doctor import (DoctorPolicy, MixedExpertPolicy,
                                     MultiCandidateDoctor)


def _make_doctor(cfg, gt, rng, segmenter=None):
    """doctor.teacher 选择模拟教师:
      rule     = 质心规则医生 (默认, 历史行为);
      multi_ns = 多候选多样采样教师, 不打分 (ISIC 实测 95.6% vs 规则 90.4%);
      multi    = 多候选 + predicted-IoU 试选 (实测选优有害, 94.4%)。
    """
    t = cfg.raw["doctor"].get("teacher", "rule")
    if t == "rule":
        return DoctorPolicy(cfg, gt, rng)
    n_cand = int(cfg.raw["doctor"].get("multi_cand", 8))
    if t == "multi_ns":
        return MultiCandidateDoctor(cfg, gt, rng, segmenter=segmenter,
                                    n_cand=n_cand, select=False)
    if t == "multi":
        return MultiCandidateDoctor(cfg, gt, rng, segmenter=segmenter,
                                    n_cand=n_cand, select=True)
    raise ValueError(f"未知 teacher: {t}")


def collect_trajectories(cfg, segmenter, images: np.ndarray, masks: np.ndarray,
                         bc_net=None, dagger_beta: float = 0.0) -> dict:
    d = cfg.raw["data"]
    dagger = bc_net is not None and dagger_beta > 0

    if dagger:
        g, mx = int(d["grid"]), int(cfg.raw["eval"]["max_clicks"])
        size = int(d["image_size"])
        init_mode = cfg.raw["doctor"].get("init_mode", "click")

        def policy_factory(gt, rng, i=0):
            doc = _make_doctor(cfg, gt, rng, segmenter)
            # BC 必须经 _BCAdapter: 网络输出网格单元, 执行需要像素坐标
            # (直接执行网络输出 = 把 0-15 的格号当成像素, 点击全挤在左上角)
            bc = _BCAdapter(bc_net, mx, g, size, init_mode)
            return MixedExpertPolicy(doc, bc, dagger_beta, rng, mx, g, size)

        def labeler_factory(gt, rng, i=0):
            return _make_doctor(cfg, gt,
                                np.random.default_rng(rng.integers(2 ** 31)),
                                segmenter)
    else:
        policy_factory = lambda gt, rng, i=0: _make_doctor(cfg, gt, rng, segmenter)  # noqa: E731
        labeler_factory = None

    episodes, recorded = run_episodes(segmenter, images, masks, policy_factory,
                                      cfg, record=True, labeler_factory=labeler_factory)
    n = len(episodes)
    clicks = [len(ep["clicks"]) for ep in episodes]
    finals = [ep["dices"][-1] for ep in episodes]
    stops: dict[str, int] = {}
    for ep in episodes:
        stops[ep["stop"]] = stops.get(ep["stop"], 0) + 1
    tag = "DAgger混合" if dagger else "专家"
    print(f"[collect] {tag}轨迹: {n} 回合, 平均交互 {np.mean(clicks):.2f}, "
          f"最终 Dice {np.mean(finals):.3f}, 结束原因 {stops}")
    print(f"[collect] 状态-动作样本: {len(recorded['labels'])}")
    recorded["final_dice"] = np.array(finals, dtype=np.float32)
    recorded["n_clicks"] = np.array(clicks, dtype=np.int32)
    return recorded


def collect_from_records(cfg, segmenter, images: np.ndarray, masks: np.ndarray,
                         names: list[str], records_by_name: dict) -> dict:
    """回放真实医生的标注轨迹 (annotate_record.py 录制), 生成训练样本。

    每个回放回合: 录制的第一次框选视为人类初始提示 (非训练样本, 与协议一致);
    每次点击 = 一个训练样本 (标签 = 真实医生的点击); 回合结束 = 确认样本。
    """
    from medsam2il.eval.rollout import (BOX_KIND, _gt_box, _new_book,
                                        _finish_record)
    from medsam2il.policy.state import (action_to_label, build_state, cell_of,
                                        pack_state, stop_label)
    from medsam2il.segmenter.base import click_map, dice_binary

    d, ev = cfg.raw["data"], cfg.raw["eval"]
    size, grid = int(d["image_size"]), int(d["grid"])
    max_clicks = int(ev["max_clicks"])
    episodes, rec = [], {"states": [], "labels": [], "episode": [], "step": []}
    covered = 0

    for i, name in enumerate(names):
        rec_i = records_by_name.get(name)
        if rec_i is None or not rec_i.get("interactions"):
            continue
        covered += 1
        rng_i = np.random.default_rng(cfg.seed * 100003 + 17 + i)
        segmenter.reset(images[i])
        pos = np.zeros((size, size), dtype=np.float32)
        neg = np.zeros((size, size), dtype=np.float32)
        ep = {"dices": [0.0], "clicks": [], "stop": "real_doctor"}
        start_step = 0
        for step, it in enumerate(rec_i["interactions"][: max_clicks + 1]):
            conf_i = segmenter.confidence
            is_first_box = (step == 0 and it["kind"] == "box"
                            and cfg.raw["doctor"].get("init_mode") == "box")
            if not is_first_box:   # 初始框 = 人类提示, 不进训练样本
                lab = (stop_label(grid) if it["kind"] == "confirm" else
                       action_to_label(grid, *cell_of(it["y"], it["x"], grid, size),
                                       it["kind"] == "click_pos"))
                _ = None
                rec["states"].append(pack_state(build_state(
                    images[i], segmenter.prob, pos, neg, step, max_clicks, conf_i)))
                rec["labels"].append(lab)
                rec["episode"].append(int(i))
                rec["step"].append(int(step))
                rec.setdefault("state_dice", []).append(float(ep["dices"][-1]))
            if it["kind"] == "box":
                segmenter.add_box(int(it["y0"]), int(it["x0"]),
                                  int(it["y1"]), int(it["x1"]))
                ep["clicks"].append((int(step), (it["y0"] + it["y1"]) // 2,
                                     (it["x0"] + it["x1"]) // 2, BOX_KIND))
            else:
                m = click_map(size, (it["y"], it["x"]), float(
                    cfg.raw["surrogate"]["sigma_click"]))
                if it["kind"] == "click_pos":
                    pos = np.maximum(pos, m)
                else:
                    neg = np.maximum(neg, m)
                segmenter.add_click(int(it["y"]), int(it["x"]), it["kind"] == "click_pos")
                ep["clicks"].append((int(step), int(it["y"]), int(it["x"]),
                                     int(it["kind"] == "click_pos")))
            ep["dices"].append(dice_binary(segmenter.mask(), masks[i]))
            start_step = step + 1
        # 回合末尾: 医生确认 -> stop 样本
        rec["states"].append(pack_state(build_state(
            images[i], segmenter.prob, pos, neg,
            min(start_step, max_clicks), max_clicks, segmenter.confidence)))
        rec["labels"].append(stop_label(grid))
        rec["episode"].append(int(i))
        rec["step"].append(int(min(start_step, max_clicks)))
        rec.setdefault("state_dice", []).append(float(ep["dices"][-1]))
        episodes.append(ep)

    recorded = _finish_record(rec, size)
    finals = [ep["dices"][-1] for ep in episodes]
    clicks = [len(ep["clicks"]) for ep in episodes]
    print(f"[collect] 真实医生轨迹: 覆盖 {covered}/{len(names)} 图, "
          f"平均交互 {np.mean(clicks) if clicks else 0:.2f}, "
          f"最终 Dice {np.mean(finals) if finals else 0:.3f}")
    print(f"[collect] 状态-动作样本: {len(recorded['labels'])}")
    recorded["final_dice"] = np.array(finals, dtype=np.float32)
    recorded["n_clicks"] = np.array(clicks, dtype=np.int32)
    return recorded
