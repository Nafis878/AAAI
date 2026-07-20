"""GPU sanity + parity + calibration (T4_RUNBOOK.md phase 2; mandatory
before any grid launch).

    python -m src.run_week --sanity                # full sequence on cuda
    python -m src.run_week --sanity --device cpu   # CPU self-test of the code path

(a) SMOKE   — 300-epoch learned-APE run on the target device; loss must fall.
(b) PARITY  — replay the first 2,000 epochs of two known CPU runs (nope_s0,
              learned_s3; identical seeds/splits) on the target device and
              compare val-accuracy curves to the committed CPU logs.
              Tolerance: mean |delta val_acc| <= 0.02 and max <= 0.05 over the
              shared eval grid. CPU<->GPU bitwise equality is NOT expected
              (reduction order); curve-level agreement is the criterion.
              FAIL => do not run the grid; report logs/parity_report.json.
(c) CALIB   — measure aggregate epochs/s at workers in {4, 6, 8} (600-epoch
              probes) and A/B --compile at the best worker count; print the
              best config and projected P0-P3 ETA from the manifest.

Writes logs/parity_report.json + logs/calibration.json. Exit 0 = grid may
launch; exit 1 = abort criteria hit.
"""

import itertools
import json
import subprocess
import sys
import time
from pathlib import Path

import pandas as pd

PARITY_RUNS = [
    ("p113_add_f0.3_nope_L1_wd1.0_s0", ["--pe", "nope", "--seed", "0"]),
    ("p113_add_f0.3_learned_L1_wd1.0_s3", ["--pe", "learned", "--seed", "3"]),
]
PARITY_EPOCHS = 2000
MEAN_TOL, MAX_TOL = 0.02, 0.05
SANITY_ROOT = "experiments_sanity"


def _train(extra, device, epochs, run_name, compile_flag=False):
    cmd = [sys.executable, "-m", "src.train", *extra,
           "--epochs", str(epochs), "--eval-every", "5", "--threads", "1",
           "--device", device, "--exp-root", SANITY_ROOT, "--run-name", run_name,
           "--ckpt-every", "-1", "--resume-every", "0"]
    if compile_flag:
        cmd.append("--compile")
    t0 = time.perf_counter()
    r = subprocess.run(cmd, capture_output=True, text=True)
    return r.returncode, time.perf_counter() - t0


def smoke(device: str) -> bool:
    rc, dt = _train(["--pe", "learned", "--seed", "0"], device, 300, "smoke_dev")
    if rc != 0:
        print(f"[sanity] SMOKE FAIL rc={rc}")
        return False
    df = pd.read_csv(Path(SANITY_ROOT) / "smoke_dev" / "log.csv")
    ok = df.train_loss.iloc[-1] < df.train_loss.iloc[0] * 0.2
    print(f"[sanity] smoke: loss {df.train_loss.iloc[0]:.3f} -> {df.train_loss.iloc[-1]:.4f} "
          f"in {dt:.0f}s: {'OK' if ok else 'FAIL (loss not falling)'}")
    return ok


def parity(device: str) -> bool:
    report = {"device": device, "tolerance": {"mean": MEAN_TOL, "max": MAX_TOL}, "runs": {}}
    ok_all = True
    for cpu_name, extra in PARITY_RUNS:
        cpu_log = Path("experiments") / cpu_name / "log.csv"
        if not cpu_log.exists():
            print(f"[sanity] parity SKIP: no CPU reference {cpu_name}")
            continue
        rc, dt = _train(extra, device, PARITY_EPOCHS, f"parity_{cpu_name}")
        if rc != 0:
            print(f"[sanity] parity FAIL rc={rc} on {cpu_name}")
            return False
        ref = pd.read_csv(cpu_log).set_index("epoch")["val_acc"]
        new = pd.read_csv(Path(SANITY_ROOT) / f"parity_{cpu_name}" / "log.csv").set_index("epoch")["val_acc"]
        shared = ref.index.intersection(new.index)
        shared = shared[shared < PARITY_EPOCHS]
        d = (ref.loc[shared] - new.loc[shared]).abs()
        run_ok = bool(d.mean() <= MEAN_TOL and d.max() <= MAX_TOL)
        report["runs"][cpu_name] = {
            "mean_abs_delta": float(d.mean()), "max_abs_delta": float(d.max()),
            "n_points": int(len(shared)), "seconds": dt, "pass": run_ok,
            "curves": {"epoch": shared.tolist(), "cpu": ref.loc[shared].tolist(),
                       "dev": new.loc[shared].tolist()},
        }
        print(f"[sanity] parity {cpu_name}: mean|d|={d.mean():.4f} max|d|={d.max():.4f} "
              f"({'PASS' if run_ok else 'FAIL'})")
        ok_all &= run_ok
    Path("logs").mkdir(exist_ok=True)
    Path("logs/parity_report.json").write_text(json.dumps(report, indent=2))
    return ok_all


def calibrate(device: str) -> dict:
    results = {}
    for workers, comp in itertools.product((4, 6, 8), (False, True)):
        if comp and workers != 6:
            continue  # A/B compile only at the middle worker count first
        procs = []
        t0 = time.perf_counter()
        for w in range(workers):
            cmd = [sys.executable, "-m", "src.train", "--pe", "nope", "--seed", str(50 + w),
                   "--epochs", "600", "--eval-every", "5", "--threads", "1",
                   "--device", device, "--exp-root", SANITY_ROOT,
                   "--run-name", f"calib_w{workers}c{int(comp)}_{w}",
                   "--ckpt-every", "-1", "--resume-every", "0"]
            if comp:
                cmd.append("--compile")
            procs.append(subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL))
        for p in procs:
            p.wait()
        dt = time.perf_counter() - t0
        eps = workers * 600 / dt
        results[f"workers={workers},compile={comp}"] = round(eps, 1)
        print(f"[sanity] calib workers={workers} compile={comp}: {eps:.1f} ep/s aggregate")
    best = max(results, key=results.get)
    out = {"device": device, "aggregate_eps": results, "best": best,
           "cpu_baseline_eps": 11.5,
           "note": "projected ETAs = manifest epochs / best eps; see run_week --dry-run for totals"}
    Path("logs/calibration.json").write_text(json.dumps(out, indent=2))
    print(f"[sanity] best config: {best} ({results[best]} ep/s)")
    return out


def run_sanity(device: str = "auto") -> int:
    from src.train import resolve_device

    dev = str(resolve_device(device))
    print(f"[sanity] target device: {dev}")
    if not smoke(dev):
        return 1
    if not parity(dev):
        print("[sanity] PARITY FAILED — do NOT run the grid; inspect logs/parity_report.json")
        return 1
    calibrate(dev)
    print("[sanity] ALL CHECKS PASSED — grid may launch")
    return 0
