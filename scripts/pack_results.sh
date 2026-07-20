#!/usr/bin/env bash
# Pack exactly the artifacts a later integration session needs from the GPU run,
# for manual upload when no GITHUB_TOKEN is available. Excludes the bulky
# intermediate checkpoints (only final checkpoints are kept; trajectories for
# the designated subset are already small).
set -euo pipefail
cd "$(dirname "$0")/.."

OUT="gpu_week_results_$(date -u +%Y%m%d_%H%M).zip"

# CSV logs + configs + final checkpoints from every GPU run; trajectory ckpts
# for the subset that has them; the computed results dir; the parity/calibration
# reports. NOT: resume.pt, intermediate ckpts of non-trajectory runs.
python - "$OUT" <<'PY'
import sys, zipfile
from pathlib import Path

out = sys.argv[1]
zf = zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED)
def add(p):
    if p.exists(): zf.write(p)

for run in sorted(Path("experiments_gpu").glob("*")) if Path("experiments_gpu").exists() else []:
    add(run / "log.csv")
    add(run / "config.json")
    ck = run / "checkpoints"
    if ck.exists():
        add(ck / "ckpt_final.pt")
        for c in ck.glob("ckpt_0*.pt"):   # trajectory subset only (non-traj runs have just final)
            zf.write(c)
for extra in ["research/week/results", "logs/parity_report.json", "logs/calibration.json",
              "research/week/strengthening_plan.md", "research/week/grid_manifest.json"]:
    p = Path(extra)
    if p.is_dir():
        for f in p.rglob("*"):
            if f.is_file(): zf.write(f)
    else:
        add(p)
zf.close()
print("packed", out, "-", round(Path(out).stat().st_size/1e6, 1), "MB")
PY
echo "pack_results: upload $OUT to the integration session (or attach to the repo issue)."
