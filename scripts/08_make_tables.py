"""Assemble Table 1 and a single outputs/RESULTS.md that lists every table/figure produced."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pandas as pd

from shapalert import features as F
from shapalert.common import CFG, ROOT, out_dir, processed_dir, write_table

rows = []
for T in CFG["windows"]:
    o = pd.read_csv(processed_dir() / f"oulad_w{T}.csv")
    rows.append({"dataset": "OULAD", "window_days": T, "rows": len(o), "dropout_rate": round(o.y.mean(), 4),
                 "dropouts": int(o.y.sum()), "features_full": len(F.FULL_FEATURES), "features_shared": len(F.SHARED_FEATURES)})
for T in CFG["windows"]:
    g = pd.read_csv(processed_dir() / f"ghana_w{T}.csv")
    rows.append({"dataset": "Ghana (KNUST Moodle)", "window_days": T, "rows": len(g), "dropout_rate": round(g.y.mean(), 4),
                 "dropouts": int(g.y.sum()), "features_full": "n/a", "features_shared": len(F.SHARED_FEATURES)})
write_table(pd.DataFrame(rows), "table1_dataset_description")

T = out_dir("tables")
lines = ["# Results index (auto-generated)\n"]
for f in sorted(T.glob("*.md")):
    lines.append(f"## {f.stem}\n\n{f.read_text(encoding='utf-8')}\n")
lines.append("## Figures\n")
lines += [f"- figures/{p.name}" for p in sorted(out_dir('figures').glob('*.png'))]
(ROOT / CFG["paths"]["outputs"] / "RESULTS.md").write_text("\n".join(lines), encoding="utf-8")
print("wrote outputs/RESULTS.md")
