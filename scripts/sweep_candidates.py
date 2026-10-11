"""改进1消融: 在现有 Kvasir 策略上扫 candidates = 1/2/3 (测试时候选搜索)。"""
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CKPT = ROOT / "outputs" / "bc_policy.pt"

for k in (1, 2, 3):
    print(f"\n================ candidates={k} ================", flush=True)
    r = subprocess.run([sys.executable, "scripts/eval_checkpoint.py",
                        "--ckpt", str(CKPT), "--candidates", str(k)],
                       cwd=ROOT)
    if r.returncode != 0:
        print(f"[fail] candidates={k} 退出码 {r.returncode}", flush=True)
        sys.exit(r.returncode)
print("\n[sweep] 全部完成", flush=True)
