"""训练模仿学习策略 (Behavior Cloning on 专家交互轨迹)。

标签约定 (来自 rollout):
  y >= 0        点击: 热图目标 = 中心高斯 (正/负通道)
  y >= stop_id  确认
损失 = 点击热图 BCE + stop BCE; 模型选择只看热图损失 (stop 收敛快会带偏)。
embeds_traj: 每张轨迹图的 SAM2 图像嵌入 [N,256,16,16] fp16 (可选, 按图像 id 索引)。
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from medsam2il.policy.net import InteractionPolicyNet
from medsam2il.segmenter.base import dice_binary
from medsam2il.policy.state import (N_STATE_CH, build_state, n_state_ch,
                                    unpack_state)


def _flip_label(label: int, grid: int, horizontal: bool, vertical: bool) -> int:
    stop = grid * grid * 2
    if label < 0 or label >= stop or not (horizontal or vertical):
        return label
    neg = label >= grid * grid
    v = label - (grid * grid if neg else 0)
    cy, cx = v // grid, v % grid
    if horizontal:
        cx = grid - 1 - cx
    if vertical:
        cy = grid - 1 - cy
    out = cy * grid + cx
    return out + grid * grid if neg else out


class StateDataset(Dataset):
    """返回 (x, embed, y, ep, h, v); ep 为图像 base id (DAgger 多轮共用)。"""

    def __init__(self, states: np.ndarray, labels: np.ndarray, episode: np.ndarray,
                 grid: int, flip: bool = False, embeds: np.ndarray | None = None,
                 state_dice: np.ndarray | None = None,
                 pos_cell: np.ndarray | None = None,
                 neg_cell: np.ndarray | None = None):
        self.states = states          # [M,11,H,W] uint8
        self.labels = labels.astype(np.int64)
        self.base_ep = np.mod(episode.astype(np.int64), 10_000_000)
        self.grid = grid
        self.flip = flip
        self.embeds = embeds
        self.state_dice = (np.zeros(len(states), dtype=np.float32)
                           if state_dice is None
                           else state_dice.astype(np.float32))
        self.pos_cell = (np.full(len(states), -1, dtype=np.int32)
                         if pos_cell is None else pos_cell.astype(np.int32))
        self.neg_cell = (np.full(len(states), -1, dtype=np.int32)
                         if neg_cell is None else neg_cell.astype(np.int32))

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, i):
        x = torch.from_numpy(unpack_state(self.states[i]))
        y = int(self.labels[i])
        h = v = False
        if self.flip:
            h = bool(torch.randint(0, 2, (1,)).item())
            v = bool(torch.randint(0, 2, (1,)).item())
            if h:
                x = torch.flip(x, dims=[2])
            if v:
                x = torch.flip(x, dims=[1])
            if y >= 0:
                y = _flip_label(y, self.grid, h, v)
        e = torch.zeros(0)      # 无嵌入分支: 占位张量 (None 无法被 default_collate 拼接)
        if self.embeds is not None:
            e = torch.from_numpy(self.embeds[int(self.base_ep[i])].astype(np.float32))
            if h:
                e = torch.flip(e, dims=[2])
            if v:
                e = torch.flip(e, dims=[1])
        pc = int(self.pos_cell[i]); nc = int(self.neg_cell[i])
        if self.flip and (h or v):
            for name in ():
                pass
            for attr in ('pc', 'nc'):
                cell = locals()[attr]
                if cell < 0:
                    continue
                cy, cx = cell // self.grid, cell % self.grid
                if h:
                    cx = self.grid - 1 - cx
                if v:
                    cy = self.grid - 1 - cy
                if attr == 'pc':
                    pc = cy * self.grid + cx
                else:
                    nc = cy * self.grid + cx
        return x, e, y, int(self.base_ep[i]), int(h), int(v), \
            float(self.state_dice[i]), pc, nc


def dual_heatmap_targets(pos_cells: torch.Tensor, neg_cells: torch.Tensor,
                         grid: int, sigma: float,
                         device: str) -> tuple[torch.Tensor, torch.Tensor]:
    """双修正目标: 正通道 = FN 质心高斯, 负通道 = FP 质心高斯; 双缺 -> stop。"""
    B = len(pos_cells)
    t = torch.zeros(B, 2, grid, grid, device=device)
    yy, xx = torch.meshgrid(torch.arange(grid, device=device, dtype=torch.float32),
                            torch.arange(grid, device=device, dtype=torch.float32),
                            indexing="ij")
    for b in range(B):
        for ch, cell in ((0, int(pos_cells[b])), (1, int(neg_cells[b]))):
            if cell < 0:
                continue
            cy, cx = cell // grid, cell % grid
            t[b, ch] = torch.exp(-((yy - cy) ** 2 + (xx - cx) ** 2)
                                 / (2 * sigma ** 2))
    is_stop = (pos_cells < 0) & (neg_cells < 0)
    return t, is_stop.float()


def heatmap_click_targets(labels: torch.Tensor, grid: int, sigma: float,
                          device: str) -> tuple[torch.Tensor, torch.Tensor]:
    """点击步的高斯目标 + is_stop。"""
    B = len(labels)
    stop_id = grid * grid * 2
    t = torch.zeros(B, 2, grid, grid, device=device)
    is_stop = (labels >= stop_id).float()
    idx = torch.nonzero((labels >= 0) & (labels < stop_id)).squeeze(-1)
    if len(idx):
        lab = labels[idx]
        neg = lab >= grid * grid
        v = torch.where(neg, lab - grid * grid, lab)
        cy, cx = (v // grid).float(), (v % grid).float()
        ch = neg.long()
        yy, xx = torch.meshgrid(torch.arange(grid, device=device, dtype=torch.float32),
                                torch.arange(grid, device=device, dtype=torch.float32),
                                indexing="ij")
        g = torch.exp(-((yy[None] - cy[:, None, None]) ** 2 +
                        (xx[None] - cx[:, None, None]) ** 2) / (2 * sigma ** 2))
        t[idx, ch] = g
    return t, is_stop


def _pooled_mask(mask: np.ndarray, grid: int) -> torch.Tensor:
    m = torch.from_numpy(mask.astype(np.float32))[None, None]
    return F.adaptive_avg_pool2d(m, grid)[0, 0]


def _pred_agree(heat: torch.Tensor, stop: torch.Tensor, labels: torch.Tensor,
                grid: int, tol: int = 1) -> float:
    """宽容准确率 (只统计点击/确认样本): stop 对 stop, 或同类型且距离 <= tol。"""
    stop_id = grid * grid * 2
    p = torch.sigmoid(heat)
    B = p.shape[0]
    flat = p.reshape(B, -1)
    idx = flat.argmax(dim=1).cpu()
    ch = (idx // (grid * grid)).numpy()
    rem = (idx % (grid * grid)).numpy()
    cy, cx = rem // grid, rem % grid
    pred_stop = (torch.sigmoid(stop).cpu() >= flat.max(dim=1).values.cpu()).numpy()
    labels = labels.cpu().numpy()
    ok, n = 0, 0
    for i in range(len(labels)):
        lab = int(labels[i])
        if lab < 0:
            continue
        n += 1
        if lab >= stop_id:
            ok += int(bool(pred_stop[i]))
        else:
            if pred_stop[i]:
                continue
            neg = lab >= grid * grid
            v = lab - (grid * grid if neg else 0)
            lcy, lcx = int(v // grid), int(v % grid)
            same_type = int(ch[i]) == (1 if neg else 0)
            ok += int(same_type and max(abs(int(cy[i]) - lcy), abs(int(cx[i]) - lcx)) <= tol)
    return ok / max(n, 1)


def _pretrain_trunk(cfg, model: InteractionPolicyNet, states: np.ndarray,
                    base_ep: np.ndarray, masks_traj: np.ndarray) -> None:
    """定位预训练: 从状态图像通道预测肿瘤 mask, 教 trunk '肿瘤长什么样'。"""
    pl = cfg.raw["policy"]
    device = cfg.device
    grid = int(cfg.raw["data"]["grid"])
    targets = torch.stack([_pooled_mask(masks_traj[base_ep[i]], grid)
                           for i in range(len(states))]).to(device)
    x = torch.from_numpy(unpack_state(states)).to(device)
    head = nn.Conv2d(64, 1, 1).to(device)
    opt = torch.optim.AdamW(list(model.stem.parameters())
                            + list(model.trunk.parameters())
                            + list(head.parameters()),
                            lr=float(pl["lr"]), weight_decay=1e-4)
    bs = int(pl["batch"])
    model.train()
    for epoch in range(1, int(pl["pretrain_epochs"]) + 1):
        perm = torch.randperm(len(x))
        losses = []
        for i in range(0, len(x), bs):
            idx = perm[i:i + bs]
            opt.zero_grad()
            h = model.trunk(model.stem(x[idx]))   # stem(高分辨率) -> trunk(网格)
            logits = head(h)
            if logits.shape[-1] != grid:
                logits = F.interpolate(logits, size=(grid, grid), mode="bilinear",
                                       align_corners=False)
            loss = F.binary_cross_entropy_with_logits(logits[:, 0], targets[idx])
            loss.backward()
            opt.step()
            losses.append(float(loss.detach()))
        print(f"[bc-pretrain] epoch {epoch}: mask BCE={np.mean(losses):.4f}")
    del head


def _box_blur_np(x: np.ndarray, k: int) -> np.ndarray:
    pad = k // 2
    xp = np.pad(x, pad, mode="edge")
    cs = np.cumsum(np.cumsum(xp, axis=0), axis=1)
    cs = np.pad(cs, ((1, 0), (1, 0)))
    return ((cs[k:, k:] - cs[:-k, k:] - cs[k:, :-k] + cs[:-k, :-k]) / (k * k)).astype(np.float32)


def _perturb_mask(gt: np.ndarray, kind: str, rng: np.random.Generator) -> np.ndarray:
    """模拟"分割器当前输出"的退化形态, 用于合成点击定位监督。"""
    H, W = gt.shape
    gt = gt.astype(np.float32)
    if kind == "box":                       # 框提示后的粗 mask (最贴近真实协议)
        ys, xs = np.nonzero(gt)
        jy, jx = rng.normal(0, 3, size=(2, 2))
        y0 = int(np.clip(ys.min() + jy[0], 0, H - 2)); y1 = int(np.clip(ys.max() + 1 + jy[1], y0 + 2, H))
        x0 = int(np.clip(xs.min() + jx[0], 0, W - 2)); x1 = int(np.clip(xs.max() + 1 + jx[1], x0 + 2, W))
        m = np.zeros_like(gt); m[y0:y1, x0:x1] = 1
    elif kind == "erode":                   # 欠分割 (丢边界)
        r = int(rng.integers(2, 5))
        m = (_box_blur_np(gt, 2 * r + 1) >= 0.999).astype(np.float32)
        if m.sum() < 10:
            m = gt.copy()
    elif kind == "dilate":                  # 过分割
        r = int(rng.integers(2, 5))
        m = (_box_blur_np(gt, 2 * r + 1) > 0).astype(np.float32)
    elif kind == "shift":                   # 错位
        dx, dy = int(rng.integers(-9, 10)), int(rng.integers(-9, 10))
        m = np.zeros_like(gt)
        m[max(0, dy):H + min(0, dy), max(0, dx):W + min(0, dx)] = \
            gt[max(0, -dy):H + min(0, -dy), max(0, -dx):W + min(0, -dx)]
    else:                                   # none: 完美 mask (教 stop)
        m = gt.copy()
    return m


def _pretrain_click(cfg, model: InteractionPolicyNet, images: np.ndarray,
                    masks: np.ndarray) -> None:
    """点击定位专项预训练 (不需要轨迹, 从 GT 直接合成监督):

    对每张图模拟退化的当前 mask (框/腐蚀/膨胀/错位/完美), 按医生规则
    (修最大误差区域) 合成"应点位置"的高斯热图目标, 直接监督
    trunk + click_head; 完美 mask -> 教 stop 头"该停"。
    """
    pl = cfg.raw["policy"]
    device = cfg.device
    grid = int(cfg.raw["data"]["grid"])
    size = int(cfg.raw["data"]["image_size"])
    sigma = float(pl.get("heatmap_sigma", 0.8))
    n_samples = int(pl.get("click_pretrain_samples", 3000))
    kinds = ["box", "erode", "dilate", "shift", "none"]
    kind_p = [0.35, 0.2, 0.2, 0.15, 0.1]
    rng = np.random.default_rng(cfg.seed + 1)
    from medsam2il.segmenter.base import largest_component_centroid, dice_binary

    yy, xx = np.mgrid[0:grid, 0:grid].astype(np.float32)

    model.train()
    params = (list(model.stem.parameters()) + list(model.trunk.parameters())
              + list(model.click_head.parameters())
              + list(model.err_head.parameters())
              + list(model.offset_head.parameters())
              + list(model.dice_head.parameters()))
    opt = torch.optim.AdamW(params, lr=float(pl["lr"]), weight_decay=1e-4)
    bs = int(pl["batch"])
    stop_w = float(pl.get("stop_weight", 1.0))

    for epoch in range(1, int(pl.get("click_pretrain_epochs", 0)) + 1):
        perm = rng.permutation(len(images))
        losses, nb = [], 0
        for b0 in range(0, n_samples, bs):
            idxs = perm[np.arange(b0, min(b0 + bs, n_samples)) % len(images)]
            xb, tb, sb, eb, db, ob, cb = [], [], [], [], [], [], []
            for i in idxs:
                kind = str(rng.choice(kinds, p=kind_p))
                pm = _perturb_mask(masks[i].astype(bool), kind, rng)
                conf = 0.92 if kind == "none" else float(rng.uniform(0.4, 0.85))
                s = build_state(images[i], pm, np.zeros((size, size), np.float32),
                                np.zeros((size, size), np.float32), 1,
                                int(cfg.raw["eval"]["max_clicks"]), conf)
                xb.append(s)
                gt_b = masks[i] > 0
                pr_b = pm > 0.5
                eb.append(np.stack([(gt_b & ~pr_b).astype(np.float32),
                                    (pr_b & ~gt_b).astype(np.float32)]))
                th = np.zeros((2, grid, grid), dtype=np.float32)
                is_stop = 0.0
                d_now = dice_binary(pm > 0.5, masks[i])   # 当前 mask 的真实 Dice
                db.append(d_now)
                off_t = np.zeros(2, dtype=np.float32)      # 亚格子偏移目标
                cell_t = (0, 0)                            # 目标点击所在网格单元
                if kind == "none" or d_now >= 0.98:
                    is_stop = 1.0            # 完美 -> 医生会停
                else:
                    fn = (masks[i] > 0) & (pm < 0.5)
                    fp = (pm > 0.5) & (masks[i] == 0)
                    ch = 0 if fn.sum() >= fp.sum() else 1
                    c = largest_component_centroid(fn if ch == 0 else fp)
                    if c is not None:
                        gcy, gcx = c[0] * grid / size, c[1] * grid / size
                        th[ch] = np.exp(-((yy - gcy) ** 2 + (xx - gcx) ** 2)
                                        / (2 * sigma ** 2))
                        # 偏移目标: 质心像素在其所属网格单元内的归一化偏移
                        cell_px = size / grid
                        ccy, ccx = int(c[0]) * grid // size, int(c[1]) * grid // size
                        off_t[0] = np.clip((c[0] - (ccy + 0.5) * cell_px) / cell_px,
                                           -0.5, 0.5)
                        off_t[1] = np.clip((c[1] - (ccx + 0.5) * cell_px) / cell_px,
                                           -0.5, 0.5)
                        cell_t = (ccy, ccx)
                    else:
                        is_stop = 1.0
                tb.append(th)
                sb.append(is_stop)
                ob.append(off_t)
                cb.append(cell_t)
            x = torch.from_numpy(np.stack(xb)).to(device)
            t = torch.from_numpy(np.stack(tb)).to(device)
            is_stop_t = torch.tensor(sb, dtype=torch.float32, device=device)
            err_np = torch.from_numpy(np.stack(eb)).to(device)   # [B,2,H,W] 原分辨率
            db_t = torch.tensor(db, dtype=torch.float32, device=device)
            off_t = torch.from_numpy(np.stack(ob)).to(device)
            cell_t = torch.tensor(cb, dtype=torch.long, device=device)
            opt.zero_grad()
            heat, stop, err, dice_pred, off = model(x)
            loss = F.binary_cross_entropy_with_logits(
                heat, t, pos_weight=torch.tensor(6.0, device=device)) \
                + stop_w * F.binary_cross_entropy_with_logits(stop, is_stop_t) \
                + float(pl.get("dice_loss_weight", 1.0)) * F.l1_loss(dice_pred, db_t)
            # 误差头目标: 池化到误差头特征分辨率 (image_size/4)
            if float(pl.get("err_weight", 0.5)) > 0:
                err_t = F.adaptive_avg_pool2d(err_np, err.shape[-1])
                loss = loss + float(pl.get("err_weight", 0.5)) * \
                    F.binary_cross_entropy_with_logits(
                        err, err_t, pos_weight=torch.tensor(4.0, device=device))
            # 亚格子偏移监督: 与推理一致 —— 先按 r=32/grid 池化到网格, 再取目标单元,
            # tanh*0.5 限幅; 只在点击步 (非 stop 步) 上回归
            if float(pl.get("offset_weight", 0.5)) > 0:
                rows = (is_stop_t < 0.5).nonzero(as_tuple=True)[0]
                if len(rows):
                    r = max(off.shape[-1] // grid, 1)
                    off_pool = F.avg_pool2d(off, r)          # [B,2,grid,grid]
                    pred = torch.tanh(off_pool[rows, :, cell_t[rows, 0],
                                               cell_t[rows, 1]])
                    loss = loss + float(pl["offset_weight"]) * \
                        F.smooth_l1_loss(pred, off_t[rows])
            loss.backward()
            opt.step()
            losses.append(float(loss.detach()))
            nb += 1
        print(f"[bc-click-pretrain] epoch {epoch}: click BCE={np.mean(losses):.4f}")


def val_image_ids(recorded: dict, seed: int) -> set[int]:
    """确定性验证集图像划分 (train_bc 与 run_all 选轮共用, 保证一致)。"""
    base_ep = np.mod(recorded["episode"].astype(np.int64), 10_000_000)
    uniq = np.unique(base_ep)
    rng = np.random.default_rng(seed)
    rng.shuffle(uniq)
    n_val = max(1, int(len(uniq) * 0.15))
    return set(int(v) for v in uniq[:n_val])


def train_bc(cfg, recorded: dict, masks_traj: np.ndarray,
             embeds_traj: np.ndarray | None = None,
             images_traj: np.ndarray | None = None) -> InteractionPolicyNet:
    d, pl = cfg.raw["data"], cfg.raw["policy"]
    grid = int(d["grid"])
    torch.manual_seed(cfg.seed)
    device = cfg.device

    states, labels, episode = (recorded["states"], recorded["labels"],
                               recorded["episode"])
    base_ep = np.mod(episode, 10_000_000)
    val_eps = val_image_ids(recorded, cfg.seed)
    va = np.isin(base_ep, list(val_eps))
    tr = ~va
    n_click = int(((labels >= 0) & (labels < grid * grid * 2)).sum())
    n_stop = int((labels >= grid * grid * 2).sum())
    print(f"[bc] 样本 train={tr.sum()} val={va.sum()} "
          f"(点击 {n_click}, 确认 {n_stop}; 嵌入分支 "
          f"{'开' if embeds_traj is not None else '关'})")

    val_images = sorted(set(base_ep[va].tolist()))   # 验证集图像 id (选轮用)
    model = InteractionPolicyNet(
        grid=grid,
        embed_ch=(int(embeds_traj.shape[1]) if embeds_traj is not None else 0),
        in_ch=(n_state_ch(images_traj[0]) if images_traj is not None
               else N_STATE_CH)).to(device)
    if int(pl.get("pretrain_epochs", 0)) > 0 and masks_traj is not None:
        _pretrain_trunk(cfg, model, states[tr], base_ep[tr], masks_traj)
    if int(pl.get("click_pretrain_epochs", 0)) > 0 and images_traj is not None:
        _pretrain_click(cfg, model, images_traj, masks_traj)
    model.fine_offset = bool(pl.get("fine_offset", False))
    print(f"[bc] 输入通道={model.in_ch} (彩色) "
          f"亚格子偏移头={'开' if model.fine_offset else '关'}")

        # Dice 回归目标: 优先用 rollout 记录; 旧缓存从状态概率通道重建 (等价)
    if "state_dice" in recorded and len(recorded["state_dice"]) == len(states):
        state_dice_all = np.asarray(recorded["state_dice"], dtype=np.float32)
    else:
        state_dice_all = np.zeros(len(states), dtype=np.float32)
        for i in range(len(states)):
            prob = unpack_state(states[i])[1]
            gt = masks_traj[int(base_ep[i])] > 0
            state_dice_all[i] = dice_binary(prob > 0.5, gt)

    sigma = float(pl.get("heatmap_sigma", 0.8))
    stop_w = float(pl.get("stop_weight", 1.0))
    dual_mode = bool(pl.get("dual_mode", False)) and \
        "pos_cell" in recorded and len(recorded["pos_cell"]) == len(states)
    if pl.get("dual_mode", False) and not dual_mode:
        print("[bc] 警告: dual_mode 开但轨迹缓存无双通道标签, 回退单动作模式")
    lr = float(pl["lr"])
    head_params = list(model.click_head.parameters()) + \
        list(model.stop_head.parameters()) + \
        list(model.err_head.parameters())     # 高分辨率误差头: 主损失继续辅助训练
    if model.embed_branch is not None:
        head_params += list(model.embed_branch.parameters()) + \
            list(model.fuse.parameters())
    opt = torch.optim.AdamW([
        {"params": list(model.stem.parameters())
         + list(model.trunk.parameters()),
         "lr": lr * float(pl.get("trunk_lr_scale", 0.2))},
        {"params": head_params, "lr": lr},
    ], weight_decay=1e-4)

    dl_tr = DataLoader(StateDataset(states[tr], labels[tr], episode[tr], grid,
                                    flip=bool(pl.get("flip_augment", True)),
                                    embeds=embeds_traj,
                                    state_dice=state_dice_all,
                                    pos_cell=recorded.get("pos_cell"),
                                    neg_cell=recorded.get("neg_cell")),
                       batch_size=int(pl["batch"]), shuffle=True)
    dl_va = DataLoader(StateDataset(states[va], labels[va], episode[va], grid,
                                    embeds=embeds_traj,
                                    state_dice=state_dice_all,
                                    pos_cell=recorded.get("pos_cell"),
                                    neg_cell=recorded.get("neg_cell")),
                       batch_size=int(pl["batch"]) * 2)
    uniq_ep = np.unique(base_ep)
    pooled = {int(e): _pooled_mask(masks_traj[e], grid) for e in uniq_ep}


    best_val, best_state, patience = np.inf, None, 0
    for epoch in range(1, int(pl["epochs"]) + 1):
        model.train()
        tr_loss, nb = 0.0, 0
        for xb, eb, yb, epb, hb, vb, sdb, pcb, ncb in dl_tr:
            xb, yb = xb.to(device), yb.to(device)
            eb = eb.to(device) if eb is not None else None
            sdb = sdb.to(device)
            if dual_mode:
                t_heat, t_stop = dual_heatmap_targets(pcb.to(device),
                                                      ncb.to(device), grid,
                                                      sigma, device)
            else:
                t_heat, t_stop = heatmap_click_targets(yb, grid, sigma, device)
            opt.zero_grad()
            heat, stop, err, dice_pred, _ = model(xb, eb)
            loss_h = F.binary_cross_entropy_with_logits(
                heat, t_heat, pos_weight=torch.tensor(6.0, device=device))
            loss_s = F.binary_cross_entropy_with_logits(stop, t_stop)
            loss = loss_h + stop_w * loss_s
            # 辅助误差头 (高分辨率): 用 GT vs 当前分割(概率>0.5) 合成 FN/FP 目标,
            # 目标分辨率跟随误差头特征 (image_size/4), 不写死
            if float(pl.get('err_weight', 0.5)) > 0:
                res = err.shape[-1]
                with torch.no_grad():
                    gt32 = torch.stack([
                        _pooled_mask(masks_traj[int(e)], res) for e in epb
                    ]).to(device)
                pred32 = F.adaptive_avg_pool2d((xb[:, 1] > 0.5).float()[:, None],
                                               res)[:, 0]
                gt_b, pred_b = gt32 > 0.5, pred32 > 0.5
                fn = (gt_b & ~pred_b).float()
                fp = (pred_b & ~gt_b).float()
                loss_e = F.binary_cross_entropy_with_logits(
                    err, torch.stack([fn, fp], dim=1),
                    pos_weight=torch.tensor(4.0, device=device))
                loss = loss + float(pl.get('err_weight', 0.5)) * loss_e
            loss = loss + float(pl.get('dice_loss_weight', 1.0)) * \
                F.l1_loss(dice_pred, sdb)
            loss.backward()
            opt.step()
            tr_loss += float(loss.detach())
            nb += 1
        model.eval()
        vloss, vloss_h, vac, vdice, vn = 0.0, 0.0, 0.0, 0.0, 0
        with torch.no_grad():
            for xb, eb, yb, epb, hb, vb, sdb, pcb, ncb in dl_va:
                xb, yb = xb.to(device), yb.to(device)
                eb = eb.to(device) if eb is not None else None
                if dual_mode:
                    t_heat, t_stop = dual_heatmap_targets(pcb.to(device),
                                                          ncb.to(device), grid,
                                                          sigma, device)
                else:
                    t_heat, t_stop = heatmap_click_targets(yb, grid, sigma, device)
                heat, stop, _, dice_logit, _ = model(xb, eb)
                lh = F.binary_cross_entropy_with_logits(
                    heat, t_heat, pos_weight=torch.tensor(6.0, device=device))
                ls = F.binary_cross_entropy_with_logits(stop, t_stop)
                vloss += float(lh + stop_w * ls)
                vloss_h += float(lh)
                vac += _pred_agree(heat, stop, yb, grid)
                vdice += float(F.l1_loss(torch.sigmoid(dice_logit), sdb))
                vn += 1
        vloss, vloss_h, vac = vloss / vn, vloss_h / vn, vac / vn
        vdice = vdice / vn
        print(f"[bc] epoch {epoch:02d} loss={tr_loss / nb:.4f} "
              f"val_loss={vloss:.4f} val_heat={vloss_h:.4f} "
              f"val_diceMAE={vdice:.3f} val_acc(±1格)={vac:.3f}")
        if vloss_h < best_val - 1e-4:
            best_val, patience = vloss_h, 0
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            patience += 1
            if patience >= int(pl["patience"]):
                print("[bc] early stop")
                break

    model.load_state_dict(best_state)

    # F0.5 校准停手阈值 (停手精度优先: 误停的代价 > 多点几下)
    model.eval()
    with torch.no_grad():
        p_stops, is_stops, d_preds = [], [], []
        for xb, eb, yb, epb, hb, vb, sdb, pcb, ncb in dl_va:
            _, stop, _, dlogit, _ = model(xb.to(device),
                                          eb.to(device) if eb is not None else None)
            p_stops.append(torch.sigmoid(stop).cpu())
            d_preds.append(torch.sigmoid(dlogit).cpu())
            is_stops.append((yb >= grid * grid * 2).float())
    p_stop = torch.cat(p_stops).numpy()
    is_stop = torch.cat(is_stops).numpy().astype(bool)
    best_tau, best_f = 0.5, -1.0
    for tau in np.quantile(p_stop, np.linspace(0.02, 0.98, 97)):
        pred = p_stop >= tau
        tp = float((pred & is_stop).sum())
        fp = float((pred & ~is_stop).sum())
        fn = float((~pred & is_stop).sum())
        prec = tp / max(tp + fp, 1)
        rec = tp / max(tp + fn, 1)
        f = 1.25 * prec * rec / max(0.25 * prec + rec, 1e-9)
        if f > best_f:
            best_f, best_tau = f, float(tau)
    model.tau = best_tau
    print(f"[bc] 停止阈值校准: tau={best_tau:.3f} (val F0.5={best_f:.3f}, "
          f"stop 占比 {is_stop.mean():.1%})")
    # dice_theta: 预测 Dice 的停手阈值 (F0.5 同法)
    d_pred = torch.cat(d_preds).numpy()
    best_dtheta, best_df = 0.85, -1.0
    for th in np.quantile(d_pred, np.linspace(0.02, 0.98, 97)):
        pred = d_pred >= th
        tp = float((pred & is_stop).sum())
        fp = float((pred & ~is_stop).sum())
        fn = float((~pred & is_stop).sum())
        prec = tp / max(tp + fp, 1)
        rec = tp / max(tp + fn, 1)
        f = 1.25 * prec * rec / max(0.25 * prec + rec, 1e-9)
        if f > best_df:
            best_df, best_dtheta = f, float(th)
    model.stop_mode = str(pl.get("stop_mode", "classifier"))
    model.dice_theta = best_dtheta
    print(f"[bc] dice_theta 校准: {best_dtheta:.3f} (val F0.5={best_df:.3f}) -> "
          f"stop_mode={model.stop_mode}")
    # apply_theta: 双模式下"次点击是否存在"的峰值阈值 (F1 校准)
    model.dual_mode = dual_mode
    model.apply_theta = 0.5
    if dual_mode and "pos_cell" in recorded and len(recorded["pos_cell"]) == len(states):
        with torch.no_grad():
            pp, npg, pres = [], [], []
            for xb, eb, yb, epb, hb, vb, sdb, pcb, ncb in dl_va:
                heat, _, _, _, _ = model(xb.to(device),
                                         eb.to(device) if eb is not None else None)
                p2 = torch.sigmoid(heat)
                pp.append(p2[:, 0].amax(dim=(1, 2)).cpu())
                npg.append(p2[:, 1].amax(dim=(1, 2)).cpu())
                pres.append(torch.stack([(pcb >= 0).float(),
                                         (ncb >= 0).float()], dim=1))
            peaks = torch.cat(pp).numpy()      # [M] 正通道峰值
            peaks_n = torch.cat(npg).numpy()   # [M] 负通道峰值
            pr2 = torch.cat(pres).numpy()      # [M,2] 医生是否点了该类型
            best_at, best_f1 = 0.5, -1.0
            for th in np.quantile(np.concatenate([peaks, peaks_n]),
                                  np.linspace(0.05, 0.95, 37)):
                tp = fp = fn = 0.0
                for ch, peaks_ch in ((0, peaks), (1, peaks_n)):
                    pred_p = peaks_ch >= th
                    true_p = pr2[:, ch] > 0
                    tp += float((pred_p & true_p).sum())
                    fp += float((pred_p & ~true_p).sum())
                    fn += float((~pred_p & true_p).sum())
                f1 = 2 * tp / max(2 * tp + fp + fn, 1)
                if f1 > best_f1:
                    best_f1, best_at = f1, float(th)
        model.apply_theta = best_at
        print(f"[bc] apply_theta 校准: {best_at:.3f} (val F1={best_f1:.3f})")

    out = cfg.resolve("outputs") / "bc_policy.pt"
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": model.state_dict(), "grid": grid,
                "tau": best_tau, "embed_ch": model.embed_ch,
                "in_ch": model.in_ch,
                "val_heat": float(best_val), "val_images": val_images,
                "stop_mode": model.stop_mode,
                "dice_theta": best_dtheta,
                "apply_theta": float(model.apply_theta),
                "fine_offset": bool(model.fine_offset),
                "dual_mode": bool(model.dual_mode)}, out)
    model.val_heat = float(best_val)   # 供多轮 DAgger 按验证集选轮
    model.val_images = val_images
    print(f"[bc] 保存 -> {out}")
    return model
