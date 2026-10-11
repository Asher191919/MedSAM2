"""一键运行: 数据 -> 分割后端 -> (嵌入缓存) -> 多轮 DAgger 轨迹 -> BC -> 评测报告。

用法:
  python scripts/run_all.py --config configs/config_real.yaml   # 真数据 + 真 MedSAM2
  python scripts/run_all.py                                     # 合成数据端到端验证
每轮轨迹/策略均按文件缓存, 重跑只增量执行缺失阶段。
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from medsam2il.config import Config  # noqa: E402
from medsam2il.data.dataset import get_dataset  # noqa: E402
from medsam2il.eval.report import evaluate_policies, save_results, write_report  # noqa: E402
from medsam2il.policy.net import InteractionPolicyNet  # noqa: E402
from medsam2il.segmenter.medsam2_adapter import make_segmenter  # noqa: E402
from medsam2il.segmenter.train_surrogate import train_surrogate  # noqa: E402
from medsam2il.train.collect import collect_trajectories  # noqa: E402
from medsam2il.train.train_bc import train_bc, val_image_ids  # noqa: E402


def _load_net(cfg, path: Path) -> InteractionPolicyNet:
    import torch
    payload = torch.load(path, map_location=cfg.device, weights_only=True)
    net = InteractionPolicyNet(grid=payload["grid"],
                               embed_ch=int(payload.get("embed_ch", 0)),
                               in_ch=int(payload.get("in_ch", 11))
                               ).to(cfg.device)
    net.load_state_dict(payload["state_dict"])
    net.tau = float(payload.get("tau", 0.5))
    net.stop_mode = payload.get("stop_mode", "classifier")
    net.dice_theta = float(payload.get("dice_theta", 0.85))
    net.dual_mode = bool(payload.get("dual_mode", False))
    net.apply_theta = float(payload.get("apply_theta", 0.5))
    net.fine_offset = bool(payload.get("fine_offset", False))
    net.val_heat = float(payload.get("val_heat", np.inf))
    net.val_images = payload.get("val_images")
    net.eval()   # BN 必须用 running stats (batch=1 推理)
    return net


def _merge_recs(recs: list[dict]) -> dict:
    keys = set(recs[0].keys())
    for r in recs[1:]:
        keys &= set(r.keys())          # 只保留各轮共有的键 (防旧缓存缺列)
    return {k: np.concatenate([r[k] for r in recs])
            for k in sorted(keys)}


def _embedding_cache(cfg, seg, images: np.ndarray, tag: str,
                     cache_dir: Path) -> np.ndarray | None:
    """每张图提一次 SAM2 图像嵌入 (编码器昂贵, 全程只算一次并缓存)。"""
    path = cache_dir / f"embeds_{tag}.npz"
    if path.exists():
        return np.load(path)["embeds"]
    seg.reset(images[0])
    if seg.embedding() is None:
        return None
    embs = []
    t0 = time.time()
    for k, img in enumerate(images):
        seg.reset(img)
        embs.append(seg.embedding())
        if (k + 1) % 100 == 0:
            rate = (time.time() - t0) / (k + 1)
            print(f"    ... {k + 1}/{len(images)} 嵌入 ({rate:.1f}s/图, "
                  f"剩余 {rate * (len(images) - k - 1) / 60:.0f} 分钟)")
    arr = np.stack(embs)
    np.savez_compressed(path, embeds=arr)
    print(f"[embeds] {tag}: {arr.shape} -> {path.name}")
    return arr


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    args = ap.parse_args()

    cfg = Config.load(args.config)
    d = cfg.raw["data"]
    pl_cfg = cfg.raw.get("policy", {})
    t0 = time.time()
    np.random.seed(cfg.seed)
    out_dir = cfg.resolve("outputs")

    print("\n========== [1/6] 数据 ==========")
    ext = d.get("external_dir")
    ds_tag = Path(ext).name if ext else "synthetic"
    if d.get("run_name"):       # 实验命名空间: 缓存/报告/图全部隔离 (消融可复跑)
        ds_tag = f"{ds_tag}_{d['run_name']}"
    cache_dir = out_dir / f"cache_{ds_tag}"   # 缓存按数据集+实验隔离
    cache_dir.mkdir(parents=True, exist_ok=True)
    images, masks, meta = get_dataset(cfg, split_seed=0)
    cfg.dataset = (images, masks, meta)
    n_sur, n_traj, n_eval = int(d["n_surrogate_train"]), int(d["n_traj"]), int(d["n_eval"])
    assert len(images) >= n_sur + n_traj + n_eval, \
        f"数据不足: {len(images)} < {n_sur}+{n_traj}+{n_eval}"
    print(f"[data/{ds_tag}] 共 {len(images)} 张: 替身 {n_sur} | 轨迹 {n_traj} | "
          f"评测 {n_eval}, 肿瘤面积占比中位数 "
          f"{np.median(meta['area_frac']):.3f}")

    print("\n========== [2/6] 分割后端 ==========")
    m = cfg.raw.get("medsam2", {})
    use_medsam2 = bool(m.get("checkpoint")) and cfg.resolve(m["checkpoint"]).exists()
    if use_medsam2:
        seg = make_segmenter(cfg)          # 真后端: 无需替身
    elif (out_dir / "surrogate_unet.pt").exists():
        seg = make_segmenter(cfg)
        print("[skip] 复用已有替身")
    else:
        seg = train_surrogate(cfg)

    print("\n========== [3/6] 图像嵌入缓存 (策略网络视觉特征) ==========")
    masks_traj = masks[n_sur:n_sur + n_traj]
    embeds_traj = embeds_eval = None
    if use_medsam2 and cfg.raw.get("policy", {}).get("use_sam_embed", True):
        embeds_traj = _embedding_cache(cfg, seg, images[n_sur:n_sur + n_traj],
                                       "traj", cache_dir)
        embeds_eval = _embedding_cache(cfg, seg, images[-n_eval:], "eval", cache_dir)
        if embeds_traj is not None:
            print(f"[embeds] 嵌入分支启用: {embeds_traj.shape[1]} 通道 @ 16x16")

    print("\n========== [4/6] 多轮 DAgger 轨迹收集 ==========")
    coll = cfg.raw.get("collect", {"mode": "expert"})
    if coll.get("mode") == "dagger":
        betas = list(coll.get("dagger_betas", [coll.get("dagger_beta", 0.5)]))
    else:
        betas = []

    def _load_or_collect(path, bc_net=None, beta=0.0, tag=""):
        if path.exists():
            z = np.load(path)
            print(f"[skip] 复用已有{tag}轨迹 ({len(z['labels'])} 样本)")
            return dict(z)
        rec = collect_trajectories(cfg, seg, images[n_sur:n_sur + n_traj],
                                   masks_traj, bc_net=bc_net, dagger_beta=beta)
        np.savez_compressed(path, **{k: v for k, v in rec.items()})
        return rec

    # r0: 真实医生轨迹 (若提供录制文件) 否则规则模拟医生
    real_key = d.get("real_trajectories")
    real_file = Path(cfg.resolve(real_key)) if real_key else None
    if real_file and real_file.exists():
        records_by_name = {}
        with open(real_file, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    rec = json.loads(line)
                    records_by_name[rec["image"]] = rec
        names = meta.get("names", [f"img_{i}" for i in range(n_traj)])
        tr_path = cache_dir / "trajectories_real.npz"
        if tr_path.exists():
            recs = [dict(np.load(tr_path))]
            print(f"[skip] 复用真实医生轨迹 ({len(recs[0]['labels'])} 样本)")
        else:
            from medsam2il.train.collect import collect_from_records
            rec = collect_from_records(cfg, seg, images[n_sur:n_sur + n_traj],
                                       masks_traj, names[:n_traj],
                                       records_by_name)
            np.savez_compressed(tr_path, **{k: v for k, v in rec.items()})
            recs = [rec]
        r0_cache = cache_dir / "bc_round0.pt"
        if not r0_cache.exists():
            net = train_bc(cfg, _merge_recs(recs), masks_traj, embeds_traj,
                           images_traj=images[n_sur:n_sur + n_traj])
            shutil.copy(out_dir / "bc_policy.pt", r0_cache)
    else:
        recs = [_load_or_collect(cache_dir / "trajectories.npz", tag="专家")]

    net = None
    best = (-np.inf, None)   # (验证集任务成功率, cache path) —— 不用测试集选轮

    def _val_task_success(cfg, seg, net, embeds, images_t, masks_t, val_eps):
        """在验证集图像上跑 BC 回合, 返回任务成功率 (选轮用, 不碰测试集)。"""
        from medsam2il.eval.report import _BCAdapter
        from medsam2il.eval.rollout import (clicks_to_target, dice_curves,
                                            run_episodes)
        idx = [i for i in range(len(images_t)) if i in val_eps]
        if not idx:
            return None
        ev = cfg.raw["eval"]

        def factory(gt, rng, i=0):
            return _BCAdapter(net, int(ev["max_clicks"]), int(d["grid"]),
                              int(d["image_size"]),
                              cfg.raw["doctor"].get("init_mode", "click"),
                              embed=None if embeds is None else embeds[idx[i]])

        eps, _ = run_episodes(seg, images_t[idx], masks_t[idx], factory, cfg)
        curves = dice_curves(eps, int(ev["max_clicks"]))
        _, ok = clicks_to_target(curves, float(ev["target_dice"]))
        return float(ok.mean())

    for r in range(len(betas) + 1):
        print(f"\n---- 训练 round{r} (样本轮次: {r + 1}) ----")
        cache = cache_dir / f"bc_round{r}.pt"
        if cache.exists():
            net = _load_net(cfg, cache)
            print(f"[skip] 复用 round{r} 策略")
        else:
            net = train_bc(cfg, _merge_recs(recs), masks_traj, embeds_traj,
                           images_traj=images[n_sur:n_sur + n_traj])
            shutil.copy(out_dir / "bc_policy.pt", cache)
        val_eps = getattr(net, "val_images", None)
        if val_eps is None:   # 旧缓存: 按同一确定性划分重算
            val_eps = val_image_ids(_merge_recs(recs), cfg.seed)
        vs = _val_task_success(cfg, seg, net, embeds_traj,
                               images[n_sur:n_sur + n_traj], masks_traj,
                               set(val_eps))
        net.val_success = vs
        # 回写缓存, 供下次跳过重算
        if vs is not None:
            import torch
            payload = torch.load(cache, map_location=cfg.device, weights_only=True)
            if "val_success" not in payload:
                payload["val_success"] = vs
                torch.save(payload, cache)
        print(f"[select] round{r}: val 任务成功率="
              f"{f'{vs:.1%}' if vs is not None else '不可用(回退 val_heat)'}")
        score = vs if vs is not None else -net.val_heat
        if score > best[0]:
            best = (score, cache)
        if r < len(betas):
            path = cache_dir / ("trajectories_dagger.npz" if r == 0
                                else f"trajectories_dagger{r + 1}.npz")
            recs.append(_load_or_collect(path, bc_net=net, beta=betas[r],
                                         tag=f"DAgger r{r + 1}"))
    # 按验证集任务成功率选最优轮 (DAgger 后期轮可能无增益甚至退化)
    if best[1] is not None and Path(best[1]).resolve() != \
            Path(out_dir / "bc_policy.pt").resolve():
        shutil.copy(best[1], out_dir / "bc_policy.pt")
        net = _load_net(cfg, out_dir / "bc_policy.pt")
        print(f"[select] 按验证集任务成功率选轮: {best[1].name} "
              f"(val 成功率={best[0]:.1%})")
    print(f"\n[collect] 最终训练样本: {len(_merge_recs(recs)['labels'])}")

    print("\n========== [5/6] 评测 (四条指标) ==========")
    res = evaluate_policies(cfg, seg, images[-n_eval:], masks[-n_eval:],
                            np.array(meta["area_frac"][-n_eval:]), net,
                            embeds_eval=embeds_eval)
    write_report(cfg, res, n_eval, tag=ds_tag)
    save_results(cfg, res, tag=ds_tag)
    print(f"\n========== [6/6] 完成, 用时 {time.time() - t0:.0f}s ==========\n"
          f"报告: {out_dir / f'report_{ds_tag}.md'}")


if __name__ == "__main__":
    main()
