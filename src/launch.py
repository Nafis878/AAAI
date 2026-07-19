"""Day-2 grid launcher: bounded worker pool over src.train jobs.

Presets map to the proposal's passes (research_proposal.md §4):
    core      — Pass A: 5 PE x seeds 0-3, cap 15k          (H1, H2)
    ood       — Pass B: 5 PE x seeds 0-1, row-band holdout (H3)
    core-ext  — seed 4 for the core grid
    ood-ext   — seed 2 for the OOD set
    depth2    — 2-layer variant, 5 PE x seeds 0-1
    p97       — p=97, 5 PE x seeds 0-1
    ops       — sub & mul, {nope, learned, rope} x seed 0
    extend40  — Pass C: 40k-epoch reruns of the 9 not-yet-grokked censored
                core runs (D16). Deterministic seeds reproduce the 15k
                trajectory exactly, then continue; saved as <name>_x40 so
                the original runs stay untouched (append-only raw data).

Usage:
    python -m src.launch --preset core
    python -m src.launch --preset ood --workers 8
"""

import argparse
import itertools
import subprocess
import sys
import time
from pathlib import Path

PES = ("nope", "learned", "sinusoidal", "rope", "alibi")
CAP = "15000"
COMMON = ["--threads", "1", "--eval-every", "5", "--epochs", CAP]


def jobs_for(preset: str):
    if preset == "core":
        for pe, s in itertools.product(PES, range(4)):
            yield ["--pe", pe, "--seed", str(s)]
    elif preset == "core-ext":
        for pe in PES:
            yield ["--pe", pe, "--seed", "4"]
    elif preset == "ood":
        for pe, s in itertools.product(PES, range(2)):
            yield ["--pe", pe, "--seed", str(s), "--ood-a-range", "100", "112"]
    elif preset == "ood-ext":
        for pe in PES:
            yield ["--pe", pe, "--seed", "2", "--ood-a-range", "100", "112"]
    elif preset == "depth2":
        for pe, s in itertools.product(PES, range(2)):
            yield ["--pe", pe, "--seed", str(s), "--layers", "2"]
    elif preset == "p97":
        for pe, s in itertools.product(PES, range(2)):
            yield ["--pe", pe, "--seed", str(s), "--p", "97"]
    elif preset == "ops":
        for op, pe in itertools.product(("sub", "mul"), ("nope", "learned", "rope")):
            yield ["--pe", pe, "--seed", "0", "--op", op]
    elif preset == "extend40b":
        # The 6 near-cap core runs (val>=0.99 at 14999, sustained window
        # incomplete): extension resolves their onsets ~15k (D18).
        targets = [("nope", 2), ("learned", 2), ("rope", 0), ("rope", 2), ("alibi", 0), ("alibi", 2)]
        for pe, s in targets:
            name = f"p113_add_f0.3_{pe}_L1_wd1.0_s{s}_x40"
            yield ["--pe", pe, "--seed", str(s), "--epochs", "40000", "--run-name", name]
    elif preset == "extend40":
        # The 9 core runs still below 99% val acc at the 15k cap (worklog 10:53):
        # all 4 sinusoidal seeds + the mid/low-transition stragglers.
        targets = [("sinusoidal", 0), ("sinusoidal", 1), ("sinusoidal", 2), ("sinusoidal", 3),
                   ("learned", 0), ("learned", 1), ("nope", 3), ("rope", 3), ("alibi", 3)]
        for pe, s in targets:
            name = f"p113_add_f0.3_{pe}_L1_wd1.0_s{s}_x40"
            yield ["--pe", pe, "--seed", str(s), "--epochs", "40000", "--run-name", name]
    else:
        raise SystemExit(f"unknown preset {preset!r}")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--preset", required=True)
    ap.add_argument("--workers", type=int, default=8, help="max concurrent runs (8 = RAM ceiling, decisions.md D9)")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--skip-existing", action="store_true",
                    help="skip jobs whose run dir already has checkpoints/ckpt_final.pt (idempotent relaunch)")
    args = ap.parse_args()

    queue = [COMMON + extra for extra in jobs_for(args.preset)]
    if args.skip_existing:
        from src.train import parse_args as train_parse, run_name_for

        kept = []
        for q in queue:
            parsed = train_parse(q)
            name = parsed.run_name or run_name_for(parsed)
            if (Path("experiments") / name / "checkpoints" / "ckpt_final.pt").exists():
                print(f"[launch] skip (done): {name}")
            else:
                kept.append(q)
        queue = kept
    print(f"[launch] preset={args.preset}: {len(queue)} jobs, {args.workers} workers")
    if args.dry_run:
        for q in queue:
            print("  python -m src.train", " ".join(q))
        return

    running: list[subprocess.Popen] = []
    t0 = time.time()
    failures = 0
    while queue or running:
        while queue and len(running) < args.workers:
            cmd = [sys.executable, "-m", "src.train"] + queue.pop(0)
            print(f"[launch] +{time.time() - t0:7.0f}s start: {' '.join(cmd[3:])}")
            running.append(subprocess.Popen(cmd))
        time.sleep(5)
        still = []
        for p in running:
            if p.poll() is None:
                still.append(p)
            elif p.returncode != 0:
                failures += 1
                print(f"[launch] FAILED (rc={p.returncode}): {' '.join(p.args[3:])}")
        running = still
    print(f"[launch] done in {(time.time() - t0) / 3600:.2f} h, {failures} failures")
    sys.exit(1 if failures else 0)


if __name__ == "__main__":
    main()
