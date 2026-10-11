"""改进1消融 (新 RGB 策略): candidates = 1/2/3 (测试时候选搜索)。

只评测 bc_policy (rng 按回合独立播种, 与完整评测可比);
k=1 用于核对与管线评测 87.0%/2.34 一致 (顺带验证 RGB 产物完整性)。
"""
import runpy
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from medsam2il.eval import report as R  # noqa: E402

R.POLICY_ORDER = ["bc_policy"]          # 只跑 BC, 指标计算路径不变

CKPT = ROOT / "outputs" / "bc_policy.pt"
CFG = "configs/config_real.yaml"

for k in (1, 2, 3):
    print(f"\n================ candidates={k} ================", flush=True)
    sys.argv = ["eval_checkpoint.py", "--ckpt", str(CKPT),
                "--candidates", str(k), "--config", CFG]
    runpy.run_path(str(ROOT / "scripts" / "eval_checkpoint.py"),
                   run_name="__main__")
print("\n[sweep] 全部完成", flush=True)
