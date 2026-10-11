"""训练点击条件 U-Net 替身。

关键: 每个 epoch 用上一轮模型模拟"误差驱动"的交互序列 (和医生行为一致),
让替身学会"点击后修正"的行为 —— 而不只是单点分割。
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F

from medsam2il.data.dataset import get_dataset
from medsam2il.segmenter.base import box_map, click_map, dice_binary, largest_component_centroid
from medsam2il.segmenter.unet_surrogate import ClickUNet


def _gt_box(mask: np.ndarray, rng: np.random.Generator, jitter: int = 3
            ) -> tuple[int, int, int, int]:
    ys, xs = np.nonzero(mask)
    y0, y1, x0, x1 = ys.min(), ys.max(), xs.min(), xs.max()
    j = lambda v, lo, hi: int(np.clip(v + rng.normal(0, jitter), lo, hi))
    return j(y0, 0, mask.shape[0] - 1), j(x0, 0, mask.shape[1] - 1), \
        j(y1, 0, mask.shape[0] - 1), j(x1, 0, mask.shape[1] - 1)


def _predict(model: ClickUNet, images, pos, neg, box, device, bs=64) -> np.ndarray:
    out = np.empty((len(images), images.shape[-1], images.shape[-1]), dtype=np.float32)
    with torch.no_grad():
        for i in range(0, len(images), bs):
            x = np.stack([images[i:i + bs], pos[i:i + bs], neg[i:i + bs],
                          box[i:i + bs]], axis=1)
            out[i:i + bs] = torch.sigmoid(
                model(torch.from_numpy(x).to(device)))[:, 0].cpu().numpy()
    return out


def simulate_clicks(model, images: np.ndarray, masks: np.ndarray, cfg, rng
                    ) -> tuple[np.ndarray, np.ndarray]:
    """误差驱动的交互模拟: 返回 (X [M,4,H,W], y [M,H,W]) —— 每步一个样本。"""
    d = cfg.raw["data"]
    su = cfg.raw["surrogate"]
    doc = cfg.raw["doctor"]
    size, sigma = int(d["image_size"]), float(su["sigma_click"])
    steps = int(su["sim_steps"])

    N = len(images)
    pos = np.zeros((N, size, size), dtype=np.float32)
    neg = np.zeros((N, size, size), dtype=np.float32)
    box = np.zeros((N, size, size), dtype=np.float32)

    # 初始交互: 框选 (抖动 GT 框) 或 中心正点击
    use_box = rng.random(N) < float(su.get("init_box_prob", 0.3))
    for i in range(N):
        if use_box[i]:
            box[i] = box_map(size, _gt_box(masks[i], rng))
        else:
            c = largest_component_centroid(masks[i], rng, jitter=1.0)
            pos[i] = click_map(size, c, sigma)

    Xs, ys = [], []
    prob = _predict(model, images, pos, neg, box, cfg.device)
    for _ in range(steps):
        pred = prob > 0.5
        for i in range(N):
            fn = (masks[i] > 0) & (pred[i] == 0)
            fp = (pred[i] > 0) & (masks[i] == 0)
            if fn.sum() >= fp.sum():
                c = largest_component_centroid(fn)
                if c is not None:
                    pos[i] = np.maximum(pos[i], click_map(size, c, sigma))
            else:
                c = largest_component_centroid(fp)
                if c is not None:
                    neg[i] = np.maximum(neg[i], click_map(size, c, sigma))
        prob = _predict(model, images, pos, neg, box, cfg.device)
        Xs.append(np.stack([images, pos, neg, box], axis=1))
        ys.append(masks.astype(np.float32))
    return np.concatenate(Xs), np.concatenate(ys)


def dice_loss(logits: torch.Tensor, target: torch.Tensor, eps: float = 1.0) -> torch.Tensor:
    p = torch.sigmoid(logits)
    inter = (p * target).sum(dim=(1, 2))
    card = p.sum(dim=(1, 2)) + target.sum(dim=(1, 2))
    return 1.0 - ((2 * inter + eps) / (card + eps)).mean()


def train_surrogate(cfg) -> "UNetSegmenterLike":
    from medsam2il.segmenter.unet_surrogate import UNetSegmenter
    d, su = cfg.raw["data"], cfg.raw["surrogate"]
    device = cfg.device
    torch.manual_seed(cfg.seed)
    rng = np.random.default_rng(cfg.seed)

    images, masks, meta = cfg.dataset
    n_sur = int(d["n_surrogate_train"])
    imgs_tr, msk_tr = images[:n_sur], masks[:n_sur]
    n_val = max(32, int(n_sur * su["val_frac"]))
    imgs_va, msk_va = imgs_tr[-n_val:], msk_tr[-n_val:]
    imgs_tr, msk_tr = imgs_tr[:-n_val], msk_tr[:-n_val]

    model = ClickUNet(base=int(su["base"])).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=float(su["lr"]))
    size = int(d["image_size"])

    best_val, best_state = -1.0, None
    for epoch in range(1, int(su["epochs"]) + 1):
        # 1) 用上一轮模型模拟误差驱动交互
        sub = rng.choice(len(imgs_tr), size=min(int(su["sim_subset"]), len(imgs_tr)),
                         replace=False)
        X, y = simulate_clicks(model, imgs_tr[sub], msk_tr[sub], cfg, rng)
        # 2) 训练一个 epoch
        model.train()
        order = rng.permutation(len(X))
        losses = []
        for i in range(0, len(X), int(su["batch"])):
            idx = order[i:i + int(su["batch"])]
            xb = torch.from_numpy(X[idx]).to(device)
            yb = torch.from_numpy(y[idx]).to(device)
            opt.zero_grad()
            logits = model(xb)[:, 0]
            loss = F.binary_cross_entropy_with_logits(logits, yb) + dice_loss(logits, yb)
            loss.backward()
            opt.step()
            losses.append(float(loss.detach()))
        # 3) 验证: 模拟交互看 Dice 随点击提升
        model.eval()
        vd = _val_dice(model, imgs_va, msk_va, cfg)
        print(f"[surrogate] epoch {epoch}: loss={np.mean(losses):.4f}  "
              f"val Dice@click1..{len(vd)} = " +
              " ".join(f"{v:.3f}" for v in vd))
        if vd[-1] > best_val:
            best_val = vd[-1]
            best_state = {k: v.clone() for k, v in model.state_dict().items()}

    model.load_state_dict(best_state)
    out = cfg.resolve("outputs") / "surrogate_unet.pt"
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": model.state_dict(), "base": int(su["base"])}, out)
    print(f"[surrogate] 保存 -> {out}")
    return UNetSegmenter(model, size, float(su["sigma_click"]), device)


def _val_dice(model, imgs, msk, cfg, n_steps: int = 3) -> list[float]:
    """确定性初始中心点击 + 误差驱动点击, 返回每步 Dice。"""
    d, su = cfg.raw["data"], cfg.raw["surrogate"]
    size, sigma = int(d["image_size"]), float(su["sigma_click"])
    rng = np.random.default_rng(0)
    N = len(imgs)
    pos = np.zeros((N, size, size), dtype=np.float32)
    neg = np.zeros((N, size, size), dtype=np.float32)
    box = np.zeros((N, size, size), dtype=np.float32)
    for i in range(N):
        c = largest_component_centroid(msk[i])
        pos[i] = click_map(size, c, sigma)
    prob = _predict(model, imgs, pos, neg, box, cfg.device)
    dices = [np.mean([dice_binary(prob[i] > 0.5, msk[i]) for i in range(N)])]
    for _ in range(n_steps - 1):
        pred = prob > 0.5
        for i in range(N):
            fn = (msk[i] > 0) & (pred[i] == 0)
            fp = (pred[i] > 0) & (msk[i] == 0)
            if fn.sum() >= fp.sum():
                c = largest_component_centroid(fn)
                if c is not None:
                    pos[i] = np.maximum(pos[i], click_map(size, c, sigma))
            else:
                c = largest_component_centroid(fp)
                if c is not None:
                    neg[i] = np.maximum(neg[i], click_map(size, c, sigma))
        prob = _predict(model, imgs, pos, neg, box, cfg.device)
        dices.append(np.mean([dice_binary(prob[i] > 0.5, msk[i]) for i in range(N)]))
    return dices
