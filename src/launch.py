"""Day-2 grid launcher: bounded worker pool over src.train jobs.

Presets map to the proposal's passes (research_proposal.md §4):
    core      — Pass A: 5 PE x seeds 0-3, cap 15k          (H1, H2)
    ood       — Pass B: 5 PE x seeds 0-1, row-band holdout (H3)
    core-ext  — seed 4 for the core grid
    ood-ext   — seed 2 for the OOD set
    depth2    — 2-layer variant, 5 PE x seeds 0-1
    p97       — p=97, 5 PE x seeds 0-1
    ops       — sub & mul, {nope, learned, rope} x seed 0

Usage:
    python -m src.launch --preset core
    python -m src.launch --preset ood --workers 8
"""

import argparse
import itertools
import subprocess
import sys
import time

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
    else:
        raise SystemExit(f"unknown preset {preset!r}")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--preset", required=True)
    ap.add_argument("--workers", type=int, default=8, help="max concurrent runs (8 = RAM ceiling, decisions.md D9)")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    queue = [COMMON + extra for extra in jobs_for(args.preset)]
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
