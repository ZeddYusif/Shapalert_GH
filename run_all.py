"""Run the full pipeline in the required order. Phase 1 is frozen (step 3) before any Ghana modelling.

    python run_all.py            # everything
    python run_all.py --phase1   # OULAD only, stops after freeze + fairness
    python run_all.py --phase23  # Ghana + explanation (requires a frozen Phase 1)
"""
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).parent
PHASE1 = ["01_build_oulad.py", "02_oulad_models.py", "03_freeze.py", "04_oulad_fairness.py"]
PHASE23 = ["05_build_ghana.py", "06_ghana_experiments.py --small", "07_explain.py", "08_make_tables.py"]
steps = PHASE1 if "--phase1" in sys.argv else PHASE23 if "--phase23" in sys.argv else PHASE1 + PHASE23
if steps is PHASE23 and not (ROOT / "outputs/frozen/frozen_pipeline.json").exists():
    sys.exit("Phase 1 is not frozen yet: run `python run_all.py --phase1` first.")
for s in steps:
    print(f"\n=== {s} ===", flush=True)
    name, *args = s.split()
    subprocess.run([sys.executable, "-W", "ignore", str(ROOT / "scripts" / name), *args], check=True, cwd=ROOT)
