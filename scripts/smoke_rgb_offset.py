"""改进2/3冒烟测试: RGB 状态通道、新网络结构 (5元组/高分辨率误差头/偏移头)、
train_bc 全循环 (含预训练偏移监督) 在小型合成数据上跑通。"""
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from medsam2il.config import Config
from medsam2il.policy.net import InteractionPolicyNet
from medsam2il.policy.state import build_state, n_state_ch
from medsam2il.train.train_bc import train_bc

rng = np.random.default_rng(0)
H = W = 128
grid = 16

# ---- 1. 状态通道数: 灰度 11 / 彩色 13 ----
gray = rng.random((H, W)).astype(np.float32)
rgb = rng.random((H, W, 3)).astype(np.float32)
prob = np.clip(rng.random((H, W)).astype(np.float32), 0, 1)
z = np.zeros((H, W), np.float32)
assert n_state_ch(gray) == 11 and n_state_ch(rgb) == 13
s_gray = build_state(gray, prob, z, z, 1, 10, 0.7)
s_rgb = build_state(rgb, prob, z, z, 1, 10, 0.7)
assert s_gray.shape == (11, H, W) and s_rgb.shape == (13, H, W)
assert np.allclose(s_rgb[:3].transpose(1, 2, 0), rgb, atol=1 / 255)
print("[ok] build_state: 灰度 11ch / 彩色 13ch, RGB 通道原样透传")

# ---- 2. 新网络结构: forward 5 元组, 分辨率正确 ----
net = InteractionPolicyNet(grid=grid, embed_ch=0, in_ch=13)
net.fine_offset = True
net.eval()
x = torch.from_numpy(s_rgb[None])
heat, stop, err, dice_logit, off = net(x)
assert heat.shape == (1, 2, grid, grid)
assert err.shape == (1, 2, 32, 32), err.shape     # 高分辨率误差头
assert off.shape == (1, 2, 32, 32), off.shape     # 亚格子偏移头
assert stop.shape == (1,) and dice_logit.shape == (1,)
print("[ok] forward 5 元组: heat[2,16,16] err/off[2,32,32] stop/dice[1]")

# ---- 3. act: 偏移头确实改变点击像素 ----
from medsam2il.eval.report import _BCAdapter
net.tau = 1.01          # 随机权重下永不停手, 强制走点击路径
ad = _BCAdapter(net, 10, grid, H, "click")
pix = []
for t in range(6):
    prob_t = np.clip(prob + 0.05 * t * rng.standard_normal(prob.shape), 0, 1)
    a = ad.act(rgb, prob_t.astype(np.float32), z, z, 3, conf=0.6)
    assert a[0] == "click"
    pix.append((round(a[1], 2), round(a[2], 2)))
uniq = len(set(pix))
assert uniq >= 2, f"偏移头未生效 (6 次点击全部相同: {pix})"
cell_px = H / grid
for (y, x_) in pix:
    assert 0 <= y < H and 0 <= x_ < W
    # 偏移后不应再是格子中心 (格子中心 = (cell+0.5)*cell_px, 即 cell_px 的整数倍+半格)
    assert (y / cell_px) % 1.0 != 0.5 or (x_ / cell_px) % 1.0 != 0.5
print(f"[ok] act+偏移: 6 个不同状态得到 {uniq} 个不同像素位置 (亚格子偏移生效)")
# 关闭偏移后应与格子中心一致
net.fine_offset = False
a = ad.act(rgb, prob, z, z, 3, conf=0.6)
net.fine_offset = True
a0 = ad.act(rgb, prob, z, z, 3, conf=0.6)
print(f"[info] fine_offset 开/关: ({a0[1]:.1f},{a0[2]:.1f}) vs 关时 ({a[1]},{a[2]})")

# ---- 4. train_bc 全循环 (小数据, 快) ----
N = 12
images = rng.random((N, H, W, 3)).astype(np.float32)
masks = (rng.random((N, H, W)) > 0.55).astype(np.uint8)
states, labels, episode, step = [], [], [], []
for i in range(N):
    prob_i = np.clip(images[i].mean(axis=2) * 0.8, 0, 1).astype(np.float32)
    for st in range(3):
        s = build_state(images[i], prob_i, z, z, st, 10, 0.7)
        states.append(np.clip(s * 255, 0, 255).astype(np.uint8))
        labels.append(int(rng.integers(0, grid * grid * 2)))
        episode.append(i)
        step.append(st)
recorded = {"states": np.stack(states), "labels": np.array(labels),
            "episode": np.array(episode), "step": np.array(step)}
cfg = Config(raw={
    "seed": 0, "device": "cpu",
    "data": {"image_size": H, "grid": grid, "color": True},
    "policy": {"batch": 8, "lr": 0.001, "epochs": 2, "patience": 2,
               "heatmap_sigma": 0.8, "stop_weight": 1.0, "trunk_lr_scale": 0.2,
               "flip_augment": True, "pretrain_epochs": 1,
               "click_pretrain_epochs": 1, "click_pretrain_samples": 24,
               "fine_offset": True, "offset_weight": 0.5, "err_weight": 0.5,
               "dice_loss_weight": 0.0, "dual_mode": False},
    "eval": {"max_clicks": 10},
})
net2 = train_bc(cfg, recorded, masks, None, images_traj=images)
assert net2.in_ch == 13 and net2.fine_offset
print("[ok] train_bc 全循环通过 (13ch + 偏移监督 + 高分辨率误差头)")

# ---- 5. checkpoint 往返 ----
payload = torch.load(cfg.resolve("outputs") / "bc_policy.pt",
                     map_location="cpu", weights_only=True)
assert payload["in_ch"] == 13 and payload["fine_offset"] is True
net3 = InteractionPolicyNet(grid=payload["grid"], embed_ch=0, in_ch=payload["in_ch"])
net3.load_state_dict(payload["state_dict"])
net3.fine_offset = payload["fine_offset"]
net3.eval()
heat, stop, err, dl, off = net3(x)
assert err.shape == (1, 2, 32, 32)
print("[ok] checkpoint 往返 (in_ch/fine_offset 持久化)")
print("\n全部通过")
