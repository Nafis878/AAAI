"""Grid progress monitor: one command, run every 20-30 min on Day 2.

    python -m src.progress [--launch-ts "2026-07-18 23:00:00"]

Prints per-run status (epoch, val acc, grokked/censored/running/done),
aggregate throughput vs the 11 ep/s planning basis, ETA, and free disk.
"""

import argparse
import csv
import json
import shutil
import time
from datetime import datetime
from pathlib import Path

PLANNING_EPS = 11.0
FRESH_SECONDS = 120  # CSV mtime younger than this => run considered live


def last_row(csv_path: Path):
    try:
        with open(csv_path, newline="") as f:
            rows = list(csv.DictReader(f))
        return rows[-1] if rows else None
    except OSError:
        return None


def scan(exp_dir: Path = Path("experiments")):
    out = []
    for run_dir in sorted(p for p in exp_dir.iterdir() if p.is_dir()):
        row = last_row(run_dir / "log.csv")
        if row is None:
            continue
        cfg = {}
        cfg_path = run_dir / "config.json"
        if cfg_path.exists():
            cfg = json.loads(cfg_path.read_text())
        done = (run_dir / "checkpoints" / "ckpt_final.pt").exists() or (run_dir / "ckpt_final.pt").exists()
        fresh = (time.time() - (run_dir / "log.csv").stat().st_mtime) < FRESH_SECONDS
        epoch = int(row["epoch"])
        val_acc = float(row["val_acc"])
        cap = int(cfg.get("epochs", 0))
        if done:
            status = "GROKKED" if val_acc >= 0.99 else ("CENSORED" if cap and epoch >= cap - 1 else "DONE")
        else:
            status = "running" if fresh else "STALLED?"
        out.append(dict(name=run_dir.name, epoch=epoch, val_acc=val_acc, status=status, cap=cap))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--launch-ts", default=None, help='pass "YYYY-mm-dd HH:MM:SS" to compute aggregate ep/s')
    args = ap.parse_args()

    runs = scan()
    n_done = sum(r["status"] in ("GROKKED", "CENSORED", "DONE") for r in runs)
    n_live = sum(r["status"] == "running" for r in runs)
    for r in runs:
        print(f"{r['name']:44s} ep {r['epoch']:>6d}/{r['cap']:<6d} val {r['val_acc']:.4f}  {r['status']}")
    print(f"\n{len(runs)} runs: {n_done} done, {n_live} running, "
          f"{sum(r['status'] == 'STALLED?' for r in runs)} stalled?")

    if args.launch_ts:
        t0 = datetime.strptime(args.launch_ts, "%Y-%m-%d %H:%M:%S").timestamp()
        elapsed = time.time() - t0
        # epochs done since launch: sum of current epochs of runs started after launch
        total_ep = sum(r["epoch"] for r in runs)
        eps = total_ep / elapsed if elapsed > 0 else 0.0
        print(f"elapsed {elapsed / 3600:.2f} h | ~{total_ep} epochs logged | {eps:.1f} ep/s "
              f"(planning basis {PLANNING_EPS}; pace gate trips below {PLANNING_EPS * 0.8:.1f})")

    free_gb = shutil.disk_usage(".").free / 1e9
    print(f"disk free: {free_gb:.1f} GB")


if __name__ == "__main__":
    main()
