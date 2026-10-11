"""真实医生标注轨迹录制工具。

把医生的交互式标注过程录制成 JSONL (每行一个回合), 供管线作为
"真实专家轨迹"替代规则模拟医生 (run_all 检测到该文件即自动接入)。

用法:
  python scripts/annotate_record.py --data_dir data/kvasir_seg \
      --out data/real_trajectories.jsonl --limit 50

操作 (窗口内):
  左键        = 正点击 (补充肿瘤区域)
  右键        = 负点击 / 擦除 (标记多分的区域)
  b           = 框选模式: 接下来两次左键定义框的两个角
  u           = 撤销上一次交互
  Enter/Space = 确认完成本图, 保存并下一张
  s           = 跳过本图;  q = 退出

JSONL 行格式:
  {"image": "<文件名>", "interactions": [
      {"kind": "box", "y0":.., "x0":.., "y1":.., "x1":..} |
      {"kind": "click_pos"|"click_neg", "y":.., "x":..}], "seconds": ..}
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from PIL import Image  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", required=True, help="images/ 所在目录")
    ap.add_argument("--out", default="data/real_trajectories.jsonl")
    ap.add_argument("--limit", type=int, default=0, help="最多标注张数 (0=不限)")
    args = ap.parse_args()

    img_dir = Path(args.data_dir) / "images"
    if not img_dir.exists():
        img_dir = Path(args.data_dir)
    files = sorted(p for p in img_dir.iterdir()
                   if p.suffix.lower() in {".png", ".jpg", ".jpeg"})
    if args.limit:
        files = files[: args.limit]
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    print(__doc__)
    state = {"armed_box": False, "box_first": None,
             "interactions": [], "t0": time.time(), "done": False}

    def on_click(event):
        if event.inaxes is not axes or state["done"]:
            return
        y, x = int(event.ydata), int(event.xdata)
        if event.button == 1:  # 左键
            if state["armed_box"]:
                if state["box_first"] is None:
                    state["box_first"] = (y, x)
                    axes.set_title("框选: 已定第一角, 请点对角", fontsize=9)
                    fig.canvas.draw_idle()
                    return
                y0, x0 = state["box_first"]
                state["interactions"].append(
                    {"kind": "box", "y0": min(y0, y), "x0": min(x0, x),
                     "y1": max(y0, y), "x1": max(x0, x)})
                state["armed_box"], state["box_first"] = False, None
            else:
                state["interactions"].append({"kind": "click_pos", "y": y, "x": x})
        elif event.button == 3:  # 右键 = 擦除
            state["interactions"].append({"kind": "click_neg", "y": y, "x": x})
        redraw()

    def on_key(event):
        key = event.key
        if key == "u" and state["interactions"]:
            state["interactions"].pop()
        elif key == "b":
            state["armed_box"] = True
            state["box_first"] = None
            print("框选模式: 点击两个对角")
        elif key in ("enter", " "):
            state["done"] = True
        elif key == "s":
            state["interactions"] = []          # 跳过: 不落盘
            state["done"] = True
        elif key == "q":
            print("退出。")
            sys.exit(0)
        redraw()

    def redraw():
        axes.clear()
        axes.imshow(cur_img, cmap="gray")
        for it in state["interactions"]:
            if it["kind"] == "box":
                axes.add_patch(plt.Rectangle(
                    (it["x0"], it["y0"]), it["x1"] - it["x0"], it["y1"] - it["y0"],
                    fill=False, ec="gold", lw=2))
            elif it["kind"] == "click_pos":
                axes.plot(it["x"], it["y"], "+", ms=12, color="deepskyblue",
                          markeredgewidth=2.2)
            else:
                axes.plot(it["x"], it["y"], "_", ms=12, color="gold",
                          markeredgewidth=2.2)
        axes.set_title(f"{cur_name}  交互 {len(state['interactions'])} 次 "
                       f"(Enter=确认, u=撤销, b=框选, s=跳过, q=退出)", fontsize=9)
        axes.axis("off")
        fig.canvas.draw_idle()

    n_saved = 0
    for fi, f in enumerate(files):
        cur_img = np.asarray(Image.open(f).convert("L").resize((128, 128)))
        cur_name = f.name
        state.update({"armed_box": False, "box_first": None,
                      "interactions": [], "t0": time.time(), "done": False})
        fig, axes = plt.subplots(figsize=(4.5, 4.5))
        mng = plt.get_current_fig_manager()
        try:
            mng.window.wm_geometry("+80+80")
        except Exception:
            pass
        fig.canvas.mpl_connect("button_press_event", on_click)
        fig.canvas.mpl_connect("key_press_event", on_key)
        redraw()
        plt.show()
        plt.close(fig)
        if state["interactions"]:
            with open(out_path, "a", encoding="utf-8") as fout:
                fout.write(json.dumps({
                    "image": cur_name,
                    "interactions": state["interactions"],
                    "seconds": round(time.time() - state["t0"], 1),
                }, ensure_ascii=False) + "\n")
            n_saved += 1
        if args.limit and n_saved >= args.limit:
            break
    print(f"[annotate] 共保存 {n_saved} 回合 -> {out_path}")
    print(f"[annotate] 之后把 config 的 data.real_trajectories 指向该文件, "
          f"run_all 会用真实轨迹替代规则模拟医生。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
