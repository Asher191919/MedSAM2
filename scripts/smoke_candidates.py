"""改进1冒烟测试: top_candidates 格式 + rollout 候选搜索回路 (mock 分割器)。"""
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from medsam2il.config import Config
from medsam2il.eval.report import _BCAdapter
from medsam2il.eval.rollout import run_episodes
from medsam2il.policy.net import InteractionPolicyNet

rng = np.random.default_rng(0)
H = W = 128
grid = 16

# ---- 1. top_candidates 输出格式 ----
net = InteractionPolicyNet(grid=grid, embed_ch=0)
net.eval()
img = rng.random((H, W)).astype(np.float32)
prob = np.clip(rng.random((H, W)).astype(np.float32), 0, 1)
cands = net.top_candidates(img, prob, np.zeros((H, W), np.float32),
                           np.zeros((H, W), np.float32), 1, 10, conf=0.7,
                           k=3, sep=2)
assert len(cands) == 3, cands
for cy, cx, positive, score in cands:
    assert 0 <= cy < grid and 0 <= cx < grid
    assert isinstance(positive, bool) and 0.0 <= score <= 1.0
# 空间去重: 两两切比雪夫距离 > sep
for a in range(len(cands)):
    for b in range(a + 1, len(cands)):
        d = max(abs(cands[a][0] - cands[b][0]), abs(cands[a][1] - cands[b][1]))
        assert d > 2, (cands[a], cands[b])
print("[ok] top_candidates:", [(c[0], c[1], c[2], round(c[3], 3)) for c in cands])


# ---- 2. rollout 候选搜索回路 (mock 分割器) ----
class MockSeg:
    """支持回滚的 mock: mask = 概率图阈值化, 点击把局部概率推向 1。"""

    supports_rollback = True
    size = H

    def __init__(self):
        self.pts = []
        self.base = None
        self.n_try = 0

    def reset(self, image):
        self.pts = []
        self.base = image

    def _prob(self):
        p = 0.2 + 0.6 * self.base
        yy, xx = np.mgrid[0:H, 0:W]
        for (y, x, pos) in self.pts:
            g = np.exp(-((yy - y) ** 2 + (xx - x) ** 2) / (2 * 8.0 ** 2))
            p = p + (0.5 if pos else -0.5) * g
        return np.clip(p, 0, 1)

    @property
    def prob(self):
        return self._prob()

    @property
    def confidence(self):
        return float(self._prob().max())

    def mask(self):
        return (self._prob() > 0.5).astype(np.uint8)

    def embedding(self):
        return None

    def hypotheses(self):
        return None

    def add_click(self, y, x, positive):
        self.pts.append((int(y), int(x), bool(positive)))
        return self.mask()

    def pop_click(self):
        if self.pts:
            self.pts.pop()

    def try_click(self, y, x, positive):
        self.n_try += 1
        self.add_click(y, x, positive)
        return self.confidence


cfg = Config(raw={
    "seed": 0,
    "data": {"image_size": H, "grid": grid},
    "surrogate": {"sigma_click": 3.0},
    "doctor": {"init_mode": "click", "target_dice": 0.88, "max_clicks": 10,
               "click_jitter": 3.0, "suboptimal_prob": 0.15, "init_box_prob": 0.0},
    "eval": {"target_dice": 0.85, "max_clicks": 10, "candidates": 3,
             "candidate_sep": 2, "size_bins": {}},
})
net.tau = 1.01          # 永不停手, 强制走满交互以验证候选循环
N = 4
images = rng.random((N, H, W)).astype(np.float32)
masks = (rng.random((N, H, W)) > 0.6).astype(np.uint8)
seg = MockSeg()


def factory(gt, r, i=0):
    return _BCAdapter(net, 10, grid, H, "click")


eps, rec = run_episodes(seg, images, masks, factory, cfg)
assert len(eps) == N
for ep in eps:
    assert len(ep["clicks"]) >= 1
    assert len(ep["dices"]) == len(ep["clicks"]) + 1
assert seg.n_try == 3 * sum(len(e["clicks"]) for e in eps), \
    (seg.n_try, sum(len(e["clicks"]) for e in eps))
print(f"[ok] rollout candidates=3: 每回合点击数 "
      f"{[len(e['clicks']) for e in eps]} Dice "
      f"{[round(e['dices'][-1], 3) for e in eps]}; 试错次数 "
      f"{seg.n_try} = 3 x 点击数 (候选循环确实执行)")

# candidates=1 应完全等价于旧行为 (无 try_click 调用)
cfg.raw["eval"]["candidates"] = 1
eps2, _ = run_episodes(seg, images, masks, factory, cfg)
assert [len(e["clicks"]) for e in eps2] == [len(e["clicks"]) for e in eps]
print("[ok] candidates=1 与 candidates=3 同分布 (mock 下一致), 链路无异常")

# pop 后状态自洽: 试错不污染最终提示
seg.reset(images[0])
n0 = len(seg.pts)
seg.try_click(30, 40, True); seg.pop_click()
assert len(seg.pts) == n0
print("[ok] try/pop 自洽")
print("\n全部通过")
