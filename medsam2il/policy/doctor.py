"""模拟医生专家策略 —— 交互式标注轨迹的来源。

行为规则 (带噪声的真实感):
  第 0 步: 在肿瘤最大连通域质心处放一个正点击 (抖动);
  之后每步: FN/FP 谁大修谁 —— 在对应最大误差连通域里点一下
            (默认质心, 以 suboptimal_prob 概率点在域内随机位置, 模拟手抖);
  Dice >= target_dice 时 -> 确认 (stop); 达到 max_clicks 强制停。
"""
from __future__ import annotations

import numpy as np

from medsam2il.segmenter.base import dice_binary, largest_component_centroid


class DoctorPolicy:
    """需要 GT (医生当然知道答案); 只用于生成专家轨迹与"医生"参照基线。

    init_mode (config doctor.init_mode):
      - "click": 第 0 步在肿瘤质心放正点击 (合成数据/替身后端)
      - "box"  : 第 0 步给抖动的 GT 框 (真实数据/MedSAM2, 内镜标注工作流)
    """

    def __init__(self, cfg, gt_mask: np.ndarray, rng: np.random.Generator):
        self.gt = gt_mask.astype(bool)
        self.doc = cfg.raw["doctor"]
        self.init_mode = self.doc.get("init_mode", "click")
        self.dual = cfg.raw.get("policy", {}).get("dual_mode", False)
        self.rng = rng
        self.finished = False
        self.stop_reason = ""

    def _gt_box(self) -> tuple[int, int, int, int]:
        ys, xs = np.nonzero(self.gt)
        j = float(self.doc["click_jitter"])
        H, W = self.gt.shape
        clip = lambda v, hi: int(np.clip(v + self.rng.normal(0, j), 0, hi - 1))
        return (clip(ys.min(), H), clip(xs.min(), W),
                clip(ys.max() + 1, H), clip(xs.max() + 1, W))

    def act(self, image: np.ndarray, prob: np.ndarray,
            pos_map: np.ndarray, neg_map: np.ndarray, step: int,
            conf: float = 0.0, embed=None, hyp_dis=None,
            score_std: float = 0.0) -> tuple | str:
        doc = self.doc
        if step >= int(doc["max_clicks"]):
            self.finished, self.stop_reason = True, "max_clicks"
            return "stop"
        pred = prob > 0.5
        if step > 0 and dice_binary(pred, self.gt) >= float(doc["target_dice"]):
            self.finished, self.stop_reason = True, "satisfied"
            return "stop"
        if step == 0 and self.init_mode == "box":
            y0, x0, y1, x1 = self._gt_box()
            return "box", y0, x0, y1, x1
        fn = self.gt & ~pred
        fp = pred & ~self.gt
        if self.dual:
            # 双修正: 一次交互同时处理漏 (正点击) 与多 (负点击)
            pc = largest_component_centroid(fn, self.rng,
                                            jitter=float(doc["click_jitter"])) \
                if fn.sum() >= 1 else None
            nc = largest_component_centroid(fp, self.rng,
                                            jitter=float(doc["click_jitter"])) \
                if fp.sum() >= 1 else None
            if pc is None and nc is None:
                self.finished, self.stop_reason = True, "no_error"
                return "stop"
            return "interact", pc, nc
        if step == 0:
            region = self.gt & ~pred
            sub = False
        else:
            region = fn if fn.sum() >= fp.sum() else fp
            sub = self.rng.random() < float(doc["suboptimal_prob"])
        c = largest_component_centroid(region, self.rng,
                                       jitter=float(doc["click_jitter"]),
                                       suboptimal=sub)
        if c is None:  # 没有误差可修
            self.finished, self.stop_reason = True, "no_error"
            return "stop"
        return "click", c[0], c[1], step == 0 or (self.gt & ~pred).sum() >= (pred & ~self.gt).sum()


class SingleClickDoctor(DoctorPolicy):
    """"只交互一次"基线: 医生的初始动作(点击或框选)之后就确认。"""

    def act(self, image, prob, pos_map, neg_map, step, conf: float = 0.0,
            embed=None, hyp_dis=None, score_std: float = 0.0):
        if step >= 1:
            self.finished, self.stop_reason = True, "single_shot"
            return "stop"
        return super().act(image, prob, pos_map, neg_map, step, conf)


class RandomDoctor(DoctorPolicy):
    """弱基线: 初始动作后, 在最大误差连通域内随机取点 (不挑质心)。"""

    def act(self, image, prob, pos_map, neg_map, step, conf: float = 0.0,
            embed=None, hyp_dis=None, score_std: float = 0.0):
        if step >= int(self.doc["max_clicks"]):
            self.finished, self.stop_reason = True, "max_clicks"
            return "stop"
        pred = prob > 0.5
        if step > 0 and dice_binary(pred, self.gt) >= float(self.doc["target_dice"]):
            self.finished, self.stop_reason = True, "satisfied"
            return "stop"
        if step == 0:
            if self.init_mode == "box":
                y0, x0, y1, x1 = self._gt_box()
                return "box", y0, x0, y1, x1
            if self.dual:
                ys, xs = np.nonzero(self.gt)
                i = self.rng.integers(0, len(ys))
                return "click", int(ys[i]), int(xs[i]), True
            region = self.gt
        else:
            fn = self.gt & ~pred
            fp = pred & ~self.gt
            if self.dual:
                acts = []
                for m_, pos_ in ((fn, True), (fp, False)):
                    ys, xs = np.nonzero(m_)
                    if len(ys):
                        i = self.rng.integers(0, len(ys))
                        acts.append((int(ys[i]), int(xs[i])))
                if not acts:
                    self.finished, self.stop_reason = True, "no_error"
                    return "stop"
                return "interact", acts[0], acts[1] if len(acts) > 1 else None
            region = fn if fn.sum() >= fp.sum() else fp
        ys, xs = np.nonzero(region)
        if len(ys) == 0:
            self.finished, self.stop_reason = True, "no_error"
            return "stop"
        i = self.rng.integers(0, len(ys))
        return "click", int(ys[i]), int(xs[i]), \
            step == 0 or (self.gt & ~pred).sum() >= (pred & ~self.gt).sum()


class MixedExpertPolicy:
    """DAgger 执行策略: 以 beta 概率执行医生动作, 否则执行 BC 动作。

    医生说停就停 (保证回合有专家深度); BC 提前停或输出框时也听医生的。
    标签始终来自 rollout 的 labeler (另一位医生), 与执行动作解耦。
    """

    def __init__(self, doctor, bc_net, beta: float, rng: np.random.Generator,
                 max_clicks: int, grid: int, size: int):
        self.doctor = doctor
        self.bc = bc_net
        self.beta = beta
        self.rng = rng
        self.max_clicks = max_clicks
        self.grid = grid
        self.size = size
        self.stop_reason = "satisfied"

    def act(self, image, prob, pos_map, neg_map, step, conf: float = 0.0,
            embed=None, hyp_dis=None, score_std: float = 0.0):
        doc_a = self.doctor.act(image, prob, pos_map, neg_map, step, conf,
                                embed)
        if doc_a == "stop":
            self.stop_reason = getattr(self.doctor, "stop_reason", "satisfied")
            return "stop"
        if self.rng.random() < self.beta:
            return doc_a
        bc_a = self.bc.act(image, prob, pos_map, neg_map, step,
                           max_clicks=self.max_clicks, conf=conf, embed=embed,
                           hyp_dis=hyp_dis, score_std=score_std)
        if bc_a == "stop" or (isinstance(bc_a, tuple) and bc_a[0] == "box"):
            return doc_a          # BC 提前停/不会框 -> 听医生的
        return bc_a


class MultiCandidateDoctor(DoctorPolicy):
    """升级版模拟教师: 每步在 GT 误差区域 (FN 与 FP 都取) 均匀多样采样 M 个
    候选点击, 逐个喂给分割器试点, 用 predicted-IoU 选最好的执行 (try/pop 回滚,
    选择阶段不用 GT —— 与人类"试一笔看看"的标注行为同构)。

    动机: ISIC 上随机修正 (97.6%) 反超规则医生 (89.2%) —— 规则医生每步只点
    最大误差连通域的质心 (+3px 抖动), 瘦长条形的边界误差带质心落在边界线上,
    相邻步点击高度相关, 每次交互提供的新信息少。多候选 + 分割器打分兼得
    多样性与选择, 作为"更强教师"抬高蒸馏上限。
    """

    def __init__(self, cfg, gt_mask: np.ndarray, rng: np.random.Generator,
                 segmenter=None, n_cand: int = 8, select: bool = True):
        super().__init__(cfg, gt_mask, rng)
        self.seg = segmenter
        self.n_cand = max(1, int(n_cand))
        self.select = bool(select)   # False: 只多样采样不打分 (对照, 拆解反常机制)

    def act(self, image, prob, pos_map, neg_map, step, conf: float = 0.0,
            embed=None, hyp_dis=None, score_std: float = 0.0):
        if self.select and (self.seg is None or
                            not getattr(self.seg, "supports_rollback", False)):
            return super().act(image, prob, pos_map, neg_map, step, conf)
        doc = self.doc
        if step >= int(doc["max_clicks"]):
            self.finished, self.stop_reason = True, "max_clicks"
            return "stop"
        pred = prob > 0.5
        if step > 0 and dice_binary(pred, self.gt) >= float(doc["target_dice"]):
            self.finished, self.stop_reason = True, "satisfied"
            return "stop"
        if step == 0 and self.init_mode == "box":
            y0, x0, y1, x1 = self._gt_box()
            return "box", y0, x0, y1, x1
        fn = self.gt & ~pred
        fp = pred & ~self.gt
        if fn.sum() == 0 and fp.sum() == 0:
            self.finished, self.stop_reason = True, "no_error"
            return "stop"
        # 修漏 (FN -> 正点击) 与修多 (FP -> 负点击) 的候选名额均分,
        # 某一区域为空时名额全部让给另一区域; 区域内无放回采样保证候选多样
        regions = ([(fn, True)] if fn.sum() > 0 else []) + \
                  ([(fp, False)] if fp.sum() > 0 else [])
        base, extra = divmod(self.n_cand, len(regions))
        cands: list[tuple[int, int, bool]] = []
        for j, (region, pos) in enumerate(regions):
            ys, xs = np.nonzero(region)
            n_take = min(base + (1 if j < extra else 0), len(ys))
            idx = (self.rng.permutation(len(ys))[:n_take] if n_take >= len(ys)
                   else self.rng.choice(len(ys), size=n_take, replace=False))
            cands.extend((int(ys[i]), int(xs[i]), pos) for i in idx)
        if not cands:
            return super().act(image, prob, pos_map, neg_map, step, conf)
        if not self.select:
            j = int(self.rng.integers(0, len(cands)))
            y, x, pos = cands[j]
            return "click", y, x, pos
        best, best_s = None, -np.inf
        for (y, x, pos) in cands:
            s = float(self.seg.try_click(y, x, pos))
            self.seg.pop_click()          # 试点即回滚, 选中者由执行器正式应用
            if s > best_s:
                best, best_s = (y, x, pos), s
        return "click", best[0], best[1], best[2]
