"""下载/转换公开真实数据集 -> images/ + masks/ 文件夹 (管线通用格式)。

支持:
  - kvasir   : Kvasir-SEG 胃肠息肉 (1000 张, CC BY 4.0) —— 可直连下载
  - isic2018 : ISIC 2018 Task1 皮肤病灶边界 (2594 train + 100 val, 经 hf-mirror)
  - local    : 用户自备 zip/文件夹 (LiTS 切片等), 本脚本只做布局规整

用法:
  python scripts/prepare_real_data.py --dataset kvasir
  python scripts/prepare_real_data.py --dataset isic2018 --shards 25
  python scripts/prepare_real_data.py --dataset local --zip path/to/x.zip
产物: data/kvasir_seg/ 或 data/isic2018/
"""
from __future__ import annotations

import argparse
import io
import json
import shutil
import sys
import zipfile
from pathlib import Path
from urllib.request import Request, urlopen

import numpy as np
from PIL import Image

REPO = Path(__file__).resolve().parents[1]
KVASIR_URL = "https://datasets.simula.no/downloads/kvasir-seg.zip"
ISIC_REPO = "surajbijjahalli/ISIC2018"      # HF: ISIC2018 Task1, image+mask
ISIC_BASE = "https://hf-mirror.com"         # API: {BASE}/api/datasets/{REPO}
ISIC_FILE = f"{ISIC_BASE}/datasets/{ISIC_REPO}/resolve/main"
IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}


def _download(url: str, dest: Path) -> None:
    print(f"[data] 下载 {url}")
    req = Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urlopen(req, timeout=180) as r, open(dest, "wb") as f:
        total = int(r.headers.get("Content-Length", 0))
        done = 0
        while True:
            chunk = r.read(1 << 20)
            if not chunk:
                break
            f.write(chunk)
            done += len(chunk)
            if total:
                print(f"\r  {done / 1e6:.0f}/{total / 1e6:.0f} MB", end="", flush=True)
    print(f"\n[data] -> {dest} ({dest.stat().st_size / 1e6:.0f} MB)")


def _unzip(zp: Path, dest: Path) -> Path:
    dest.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zp) as z:
        z.extractall(dest)
    print(f"[data] 解压 -> {dest}")
    return dest


def _find_dir(root: Path, name: str) -> Path | None:
    """在解压目录里找名为 name 的文件夹 (任意深度)。"""
    hits = [p for p in root.rglob(name) if p.is_dir()]
    return hits[0] if hits else None


def prepare_kvasir(out: Path, work: Path) -> Path:
    zp = work / "kvasir-seg.zip"
    if not zp.exists():
        _download(KVASIR_URL, zp)
    extracted = _unzip(zp, work / "kvasir_extract")
    img_dir = _find_dir(extracted, "images")
    msk_dir = _find_dir(extracted, "masks")
    if img_dir is None or msk_dir is None:
        raise FileNotFoundError(f"解压内容异常: {list(extracted.rglob('*'))[:20]}")
    return _copy_pairs(img_dir, msk_dir, out)


def _copy_pairs(img_dir: Path, msk_dir: Path, out: Path) -> Path:
    (out / "images").mkdir(parents=True, exist_ok=True)
    (out / "masks").mkdir(parents=True, exist_ok=True)
    n = 0
    for f in sorted(img_dir.iterdir()):
        if f.suffix.lower() not in IMG_EXTS:
            continue
        m = None
        for cand in (msk_dir / f.name, msk_dir / f.with_suffix(".png").name):
            if cand.exists():
                m = cand
                break
        if m is None:
            hits = list(msk_dir.glob(f.stem + ".*"))
            m = hits[0] if hits else None
        if m is None:
            continue
        shutil.copy(f, out / "images" / f"{f.stem}{f.suffix.lower()}")
        shutil.copy(m, out / "masks" / f"{f.stem}.png")
        n += 1
    print(f"[data] 规整 {n} 对 图像/掩码 -> {out}")
    return out


def prepare_isic2018(out: Path, work: Path, shards: int, split: str = "train") -> Path:
    """ISIC 2018 Task1 (HF parquet 分片, 经 hf-mirror): 逐分片下载并转换。"""
    import pyarrow.parquet as pq

    api_url = f"{ISIC_BASE}/api/datasets/{ISIC_REPO}"
    listing = json.loads(urlopen(Request(api_url,
                                         headers={"User-Agent": "Mozilla/5.0"}),
                                 timeout=60).read())
    shard_files = sorted(f["rfilename"] for f in listing.get("siblings", [])
                         if f["rfilename"].startswith(f"data/{split}-")
                         and f["rfilename"].endswith(".parquet"))
    if not shard_files:
        raise FileNotFoundError(f"{ISIC_REPO} 里没有 {split} 分片")
    shard_files = shard_files[:shards]
    print(f"[data] ISIC2018 {split}: 使用 {len(shard_files)}/{shards} 个分片")

    (out / "images").mkdir(parents=True, exist_ok=True)
    (out / "masks").mkdir(parents=True, exist_ok=True)
    n = 0
    for si, rf in enumerate(shard_files):
        zp = work / f"isic_{split}_{si:02d}.parquet"
        if not zp.exists():
            _download(f"{ISIC_FILE}/{rf}", zp)
        tbl = pq.read_table(zp)
        for row in tbl.to_pylist():
            img_d, msk_d = row.get("image"), row.get("label")
            if not img_d or not msk_d:
                continue
            stem = Path(img_d.get("path") or f"isic_{si:02d}_{n}").stem
            ip = out / "images" / f"{stem}.png"
            mp = out / "masks" / f"{stem}.png"
            if ip.exists():
                n += 1
                continue
            Image.open(io.BytesIO(img_d["bytes"])).convert("L").save(ip)
            m = np.asarray(Image.open(io.BytesIO(msk_d["bytes"])).convert("L"))
            Image.fromarray(((m > 127) * 255).astype(np.uint8)).save(mp)
            n += 1
        zp.unlink()                       # 转完即删, 省 2GB 级磁盘
        print(f"[data] 分片 {si + 1}/{len(shard_files)} 完成, 累计 {n} 对")
    print(f"[data] ISIC2018 规整 {n} 对 -> {out}")
    return out


def prepare_local(zip_path: Path | None, folder: Path | None, out: Path,
                  work: Path) -> Path:
    if zip_path:
        extracted = _unzip(zip_path, work / "local_extract")
        img_dir = _find_dir(extracted, "images") or _find_dir(extracted, "image")
        msk_dir = _find_dir(extracted, "masks") or _find_dir(extracted, "ground_truth") \
            or _find_dir(extracted, "mask")
        if img_dir is None or msk_dir is None:
            raise FileNotFoundError("zip 里找不到 images/ 与 masks/ 文件夹, 请手动规整")
        return _copy_pairs(img_dir, msk_dir, out)
    if folder:
        img_dir = folder / "images"
        msk_dir = folder / "masks"
        if not img_dir.exists():
            img_dir, msk_dir = folder, folder  # 已是规整布局
        return _copy_pairs(img_dir, msk_dir, out)
    raise ValueError("需要 --zip 或 --folder")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=["kvasir", "isic2018", "local"],
                    default="kvasir")
    ap.add_argument("--shards", type=int, default=25,
                    help="isic2018: 下载的 train 分片数 (每片约 52 图)")
    ap.add_argument("--zip", default=None, help="local 模式: 数据集 zip 路径")
    ap.add_argument("--folder", default=None, help="local 模式: 已解压文件夹")
    args = ap.parse_args()

    work = REPO / "data_raw"
    work.mkdir(parents=True, exist_ok=True)
    out = REPO / "data" / {"kvasir": "kvasir_seg", "isic2018": "isic2018",
                           "local": "real_dataset"}[args.dataset]
    if args.dataset == "kvasir":
        prepare_kvasir(out, work)
    elif args.dataset == "isic2018":
        prepare_isic2018(out, work, args.shards)
    else:
        prepare_local(Path(args.zip) if args.zip else None,
                      Path(args.folder) if args.folder else None, out, work)
    print(f"[data] 完成。对应 config 的 external_dir 指向: {out}")


if __name__ == "__main__":
    sys.exit(main())
