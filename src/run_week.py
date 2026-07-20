"""GPU-week orchestrator: executes the priority-ordered grid manifest.

    python -m src.run_week --manifest research/week/grid_manifest.json \
        --device cuda --workers 6
    python -m src.run_week --manifest ... --dry-run     # print resolved runs
    python -m src.run_week --sanity                     # runbook phase 2

Design (see T4_RUNBOOK.md):
- priority-ordered queue (P0 -> P3), N parallel single-run worker processes;
- resumable: finished runs (checkpoints/ckpt_final.pt) are skipped; partial
  runs continue bit-exactly from their resume.pt (train.py --resume-every);
  re-running the same command after a session death continues where it left;
- graceful Ctrl-C: terminates children (they lose <=1000 epochs) and prints
  the resume command;
- optional periodic git sync every 30 min when GITHUB_TOKEN is set;
- on completion: runs analysis + figures over the GPU data and packs results.
"""

import argparse
import itertools
import json
import os
import subprocess
import sys
import time
from pathlib import Path

DEFAULT_MANIFEST = Path("research/week/grid_manifest.json")
SYNC_INTERVAL_S = 1800


def expand_block(block: dict):
    """One manifest block -> list of train.py arg-lists."""
    pes = block.get("pes", ["nope", "learned", "sinusoidal", "rope", "alibi"])
    seeds = block["seeds"]
    traj_seeds = set(block.get("traj_seeds", []))
    for pe, seed in itertools.product(pes, seeds):
        args = ["--pe", pe, "--seed", str(seed),
                "--p", str(block.get("p", 113)), "--op", block.get("op", "add"),
                "--layers", str(block.get("layers", 1)),
                "--frac", str(block.get("frac", 0.3)),
                "--epochs", str(block.get("epochs", 40000)),
                "--eval-every", "5", "--threads", "1"]
        if block.get("ood_a_range"):
            args += ["--ood-a-range"] + [str(v) for v in block["ood_a_range"]]
        if block.get("ood_b_range"):
            args += ["--ood-b-range"] + [str(v) for v in block["ood_b_range"]]
        args += ["--ckpt-every", "0" if seed in traj_seeds else "-1"]
        yield args


def resolve_queue(manifest: dict):
    queue = []
    for prio in manifest["priorities"]:
        for block in prio["blocks"]:
            for args in expand_block(block):
                queue.append((prio["name"], args))
    return queue


def run_name_of(train_args, exp_root):
    from src.train import parse_args as tp, run_name_for

    parsed = tp(train_args)
    return parsed.run_name or run_name_for(parsed)


def is_done(exp_root: Path, name: str) -> bool:
    return (exp_root / name / "checkpoints" / "ckpt_final.pt").exists()


def git_sync(exp_root: Path):
    if not os.environ.get("GITHUB_TOKEN"):
        return
    try:
        subprocess.run(["git", "add", str(exp_root), "research/week", "logs"],
                       check=False, capture_output=True)
        subprocess.run(["git", "commit", "-q", "-m", "gpu-week: periodic results sync"],
                       check=False, capture_output=True)
        subprocess.run(["git", "push", "-q", "origin", "HEAD"], check=False,
                       capture_output=True, timeout=120)
        print(f"[week] synced @ {time.strftime('%H:%M:%S')}")
    except Exception as e:  # sync must never kill the grid
        print(f"[week] sync failed (continuing): {e}")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    ap.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--exp-root", default="experiments_gpu")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--compile", action="store_true")
    ap.add_argument("--sanity", action="store_true", help="see src/sanity.py (runbook phase 2)")
    ap.add_argument("--no-analysis", action="store_true", help="skip end-of-grid analysis handoff")
    args = ap.parse_args()

    if args.sanity:
        from src.sanity import run_sanity

        sys.exit(run_sanity(device=args.device))

    manifest = json.loads(args.manifest.read_text())
    exp_root = Path(args.exp_root)
    queue = resolve_queue(manifest)
    common = ["--device", args.device, "--exp-root", str(exp_root)]
    if args.compile:
        common += ["--compile"]

    todo, done = [], 0
    for prio, targs in queue:
        if is_done(exp_root, run_name_of(targs, exp_root)):
            done += 1
        else:
            todo.append((prio, targs))
    print(f"[week] manifest: {len(queue)} runs | already done: {done} | to run: {len(todo)} "
          f"| workers: {args.workers} | device: {args.device} | root: {exp_root}")
    if args.dry_run:
        for prio, targs in todo:
            print(f"  [{prio}] python -m src.train {' '.join(targs + common)}")
        return

    exp_root.mkdir(exist_ok=True)
    running: list[tuple[subprocess.Popen, str]] = []
    t0 = time.time()
    last_sync = t0
    failures = []
    try:
        while todo or running:
            while todo and len(running) < args.workers:
                prio, targs = todo.pop(0)
                name = run_name_of(targs, exp_root)
                cmd = [sys.executable, "-m", "src.train"] + targs + common
                print(f"[week] +{time.time() - t0:7.0f}s [{prio}] start {name}")
                running.append((subprocess.Popen(cmd), name))
            time.sleep(10)
            still = []
            for p, name in running:
                if p.poll() is None:
                    still.append((p, name))
                elif p.returncode != 0:
                    failures.append(name)
                    print(f"[week] FAILED rc={p.returncode}: {name}")
                else:
                    print(f"[week] +{time.time() - t0:7.0f}s done  {name}")
            running = still
            if time.time() - last_sync > SYNC_INTERVAL_S:
                git_sync(exp_root)
                last_sync = time.time()
    except KeyboardInterrupt:
        print("\n[week] SIGINT: terminating workers (each loses <=1000 epochs; "
              "re-run the same command to resume)")
        for p, _ in running:
            p.terminate()
        sys.exit(130)

    print(f"[week] grid finished in {(time.time() - t0) / 3600:.2f} h; failures: {failures or 'none'}")
    if not args.no_analysis:
        out = Path("research/week/results")
        out.mkdir(parents=True, exist_ok=True)
        for cmd in (["-m", "src.analysis", "--all", "--exp-root", str(exp_root), "--out", str(out)],
                    ["-m", "src.figures", "--all", "--exp-root", str(exp_root), "--out", str(out)]):
            subprocess.run([sys.executable] + cmd, check=False)
        git_sync(exp_root)
    sys.exit(1 if failures else 0)


if __name__ == "__main__":
    main()
