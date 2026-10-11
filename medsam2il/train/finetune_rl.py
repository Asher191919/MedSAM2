"""BC 之后的 RL 微调 (REINFORCE with baseline)。

奖励设计 (训练时用 GT 计算, 推理不用):
  - 点击交互 t: r_t = Dice_t - Dice_{t-1} (本次交互的实际 Dice 增益)
  - 确认 (stop): r = +0.5 若停手时 Dice >= target, 否则 -0.5
回报 G_t = 折扣累积; 优势 A = G - batch 均值; 损失 = -E[log pi(a) A] + 熵正则。

策略分布: 把 (正/负热图, stop 分数) 视为 2*g*g+1 联合分类, 采样执行。
每轮 RL 后在验证集上测任务成功率, 只保留最优 (不碰测试集)。
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F

from medsam2il.eval.rollout import dice_curves, run_episodes
from medsam2il.policy.state import label_to_action, n_actions


class RLSamplingPolicy:
    """以联合分类分布采样动作 (探索), 供 RL 回合执行。"""

    def __init__(self, net, max_clicks: int, grid: int, size: int,
                 embed: np.ndarray | None, temperature: float = 1.0,
                 rng: np.random.Generator | None = None):
        self.net = net
        self.max_clicks = max_clicks
        self.grid = grid
        self.size = size
        self.embed = embed
        self.temperature = temperature
        self.rng = rng or np.random.default_rng()
        self.stop_reason = "rl_sampled"

    def act(self, image, prob, pos_map, neg_map, step, conf: float = 0.0,
            embed=None, hyp_dis=None, score_std: float = 0.0):
        from medsam2il.policy.state import build_state, cell_center
        e = embed if embed is not None else self.embed
        s = build_state(image, prob, pos_map, neg_map, step, self.max_clicks,
                        conf, hyp_dis, score_std)
        et = None if e is None else torch.from_numpy(
            np.ascontiguousarray(e.astype(np.float32))[None])
        with torch.no_grad():
            heat, stop, _, _, _ = self.net.forward(torch.from_numpy(s[None]), et)
        logits = torch.cat([heat[0].reshape(-1), stop[0].reshape(-1)])
        probs = F.softmax(logits / self.temperature, dim=-1).cpu().numpy()
        a = int(self.rng.choice(len(probs), p=probs / probs.sum()))
        if a >= self.grid * self.grid * 2:
            return "stop"
        positive = a < self.grid * self.grid
        v = a - (self.grid * self.grid if not positive else 0)
        cy, cx = int(v // self.grid), int(v % self.grid)
        y, x = cell_center(cy, cx, self.grid, self.size)
        return "click", y, x, positive


def _joint_logprob(net, states, labels, device, embeds):
    """联合分类分布下执行动作的 log pi(a|s)。"""
    x = torch.from_numpy(states.astype(np.float32)).to(device)
    e = None
    if embeds is not None:
        e = torch.from_numpy(embeds.astype(np.float32)).to(device)
    heat, stop, _, _, _ = net.forward(x, e)
    B = heat.shape[0]
    logits = torch.cat([heat.reshape(B, -1), stop.reshape(B, -1)], dim=1)
    logp = F.log_softmax(logits, dim=-1)
    lab = torch.as_tensor(labels, dtype=torch.int64, device=device)
    return logp[torch.arange(B, device=device), lab]


def finetune_rl(cfg, net, segmenter, images, masks, embeds=None,
                log_every: int = 200) -> tuple:
    """RL 微调主循环; 返回 (最优 net, 最优 val 成功率)。"""
    d, pl, ev = cfg.raw["data"], cfg.raw["policy"], cfg.raw["eval"]
    device = cfg.device
    gamma = float(pl.get("rl_gamma", 0.9))
    lr = float(pl.get("rl_lr", 2e-5))
    n_eps = int(pl.get("rl_images", 500))
    epochs = int(pl.get("rl_epochs", 2))
    target = float(ev["target_dice"])
    max_clicks = int(ev["max_clicks"])
    torch.manual_seed(cfg.seed)

    # 验证集划分 (与 train_bc 同一确定性划分)
    from medsam2il.train.train_bc import val_image_ids
    fake_rec = {"episode": np.arange(len(images))}   # 图片 id 即 episode id
    val_ids = val_image_ids(fake_rec, cfg.seed)
    tr_idx = np.array([i for i in range(len(images)) if i not in val_ids])

    from medsam2il.eval.report import _BCAdapter

    def _val_success(model):
        idx = sorted(val_ids)
        factory = lambda gt, rng, i=0: _BCAdapter(   # noqa: E731
            model, max_clicks, int(d["grid"]), int(d["image_size"]),
            cfg.raw["doctor"].get("init_mode", "click"),
            embed=None if embeds is None else embeds[idx[i]])
        eps, _ = run_episodes(segmenter, images[idx], masks[idx], factory, cfg)
        curves = dice_curves(eps, max_clicks)
        _, ok = __import__("medsam2il.eval.rollout", fromlist=["clicks_to_target"]
                           ).clicks_to_target(curves, target)
        return float(ok.mean())

    best_success = _val_success(net)
    best_state = {k: v.clone() for k, v in net.state_dict().items()}
    print(f"[rl] 起始 val 成功率={best_success:.1%}")

    opt = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=1e-5)
    rng = np.random.default_rng(cfg.seed)

    for epoch in range(1, epochs + 1):
        idx = rng.choice(tr_idx, size=min(n_eps, len(tr_idx)), replace=False)
        # 1) 采样回合 (探索执行)
        def factory(gt, rng_e, i=0):
            e = None if embeds is None else embeds[idx[i]]
            return RLSamplingPolicy(net, max_clicks, int(d["grid"]),
                                    int(d["image_size"]), e,
                                    float(pl.get("rl_temperature", 1.0)),
                                    np.random.default_rng(cfg.seed + epoch * 1000 + i))
        episodes, recorded = run_episodes(segmenter, images[idx], masks[idx],
                                          factory, cfg, record=True)
        # 2) 奖励: 点击 = ΔDice; 确认 = ±0.5 (达标与否)
        labels = recorded["labels"].astype(np.int64)
        base_ep = np.mod(recorded["episode"], 10_000_000)
        rewards = np.zeros(len(labels), dtype=np.float32)
        for ei, ep in enumerate(episodes):
            dcs = ep["dices"]
            row_sel = np.nonzero(recorded["episode"] == ei)[0]
            for row in row_sel:
                s = int(recorded["step"][row])
                a = int(labels[row])
                stop_id = n_actions(int(d["grid"])) - 1
                if a >= stop_id:
                    rewards[row] = 0.5 if dcs[min(s, len(dcs) - 1)] >= target else -0.5
                else:
                    if s + 1 < len(dcs):
                        rewards[row] = dcs[s + 1] - dcs[s]
        # 3) 折扣回报 (按回合内 step 顺序)
        returns = np.zeros(len(labels), dtype=np.float32)
        for ei, ep in enumerate(episodes):
            row_sel = np.nonzero(recorded["episode"] == ei)[0]
            order = sorted(row_sel, key=lambda r: int(recorded["step"][r]))
            G = 0.0
            for row in reversed(order):
                G = rewards[row] + gamma * G
                returns[row] = G
        # 4) REINFORCE 更新
        net.train()
        order = rng.permutation(len(labels))
        bs = int(pl.get("batch", 32))
        ep_losses = []
        for b0 in range(0, len(labels), bs):
            sel = order[b0:b0 + bs]
            A = returns[sel] - returns[sel].mean()
            lp = _joint_logprob(net, recorded["states"][sel], labels[sel],
                                device, None if embeds is None
                                else embeds[base_ep[sel]])
            entropy = -(lp.exp() * lp).sum(dim=-1).mean()
            loss = -(lp * torch.as_tensor(A, device=device)).mean() \
                - 0.01 * entropy
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
            opt.step()
            ep_losses.append(float(loss.detach()))
        vs = _val_success(net)
        print(f"[rl] epoch {epoch}: loss={np.mean(ep_losses):.4f} "
              f"回合数={len(episodes)} val 成功率={vs:.1%}")
        if vs > best_success:
            best_success = vs
            best_state = {k: v.clone() for k, v in net.state_dict().items()}
            print("[rl] 新最优, 保留")
    net.load_state_dict(best_state)
    net.val_success = best_success
    return net, best_success
