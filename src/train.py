"""Minimal full-batch training loop for the PE x grokking experiments.

One epoch = one full-batch AdamW step (proposal §2). Logs per-epoch metrics
to experiments/<run_name>/log.csv and saves periodic checkpoints.

Usage (from repo root, venv python):
    python -m src.train --pe learned --seed 0 --epochs 500
    python -m src.train --pe rope --seed 3 --p 97 --op sub --epochs 40000
"""

import argparse
import csv
import json
import time
from pathlib import Path

import torch
import torch.nn.functional as F

from src.data import make_splits
from src.model import PE_TYPES, build_model

EARLY_STOP_WINDOW = 1000  # sustained >=99% val-acc epochs before stopping
VAL_ACC_TARGET = 0.99


def ckpt_schedule(max_epochs: int, n_points: int = 35) -> set[int]:
    """~35 log-spaced checkpoint epochs (dense early, sparse late), incl. 0."""
    import numpy as np

    pts = np.unique(np.round(np.logspace(0, np.log10(max(max_epochs - 1, 1)), n_points)).astype(int))
    return {0, *pts.tolist()}


def resolve_device(spec: str) -> torch.device:
    if spec == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if spec == "cuda" and not torch.cuda.is_available():
        raise SystemExit("--device cuda requested but CUDA is not available")
    return torch.device(spec)


def save_ckpt(model, path: Path) -> None:
    """fp16 CPU state_dict snapshot — small and device-portable."""
    half = {k: v.detach().half().cpu() for k, v in model.state_dict().items()}
    torch.save(half, path)


def load_ckpt(model, path: Path) -> None:
    half = torch.load(path, map_location="cpu")
    model.load_state_dict({k: v.float() for k, v in half.items()})


def save_resume(run_dir: Path, model, opt, epoch: int, sustained_evals: int, elapsed: float) -> None:
    """Atomic full-precision training-state snapshot for exact reboot recovery.
    Full-batch training uses no RNG after init, so (model, opt, counters) is
    the complete state: resumed runs are bit-identical to uninterrupted ones
    (on the same hardware; CPU<->GPU trajectories diverge by reduction order).
    Saved on CPU so a run can resume on either device."""
    tmp = run_dir / "resume.tmp"
    cpu_model = {k: v.detach().cpu() for k, v in model.state_dict().items()}
    cpu_opt = opt.state_dict()
    cpu_opt = {"state": {k: {kk: (vv.cpu() if torch.is_tensor(vv) else vv) for kk, vv in v.items()}
                         for k, v in cpu_opt["state"].items()},
               "param_groups": cpu_opt["param_groups"]}
    torch.save({"model": cpu_model, "opt": cpu_opt,
                "epoch": epoch, "sustained": sustained_evals, "elapsed": elapsed}, tmp)
    tmp.replace(run_dir / "resume.pt")


def _trim_csv(csv_path: Path, max_epoch: int) -> None:
    """Drop rows past the resume point (they will be re-logged identically);
    prevents duplicate epochs after a mid-run crash."""
    if not csv_path.exists():
        return
    lines = csv_path.read_text().splitlines()
    if not lines:
        return
    kept = [lines[0]] + [
        ln for ln in lines[1:]
        if ln.split(",", 1)[0].isdigit() and int(ln.split(",", 1)[0]) <= max_epoch
    ]
    csv_path.write_text("\n".join(kept) + "\n")


def run_name_for(args) -> str:
    return (
        f"p{args.p}_{args.op}_f{args.frac}_{args.pe}_L{args.layers}_wd{args.wd}_s{args.seed}"
        + ("_ood" if args.ood_a_range else "")
        + ("_oodb" if getattr(args, "ood_b_range", None) else "")
    )


def evaluate(model, x, y) -> tuple[float, float]:
    with torch.no_grad():
        logits = model(x)[:, -1, :]
        loss = F.cross_entropy(logits, y).item()
        acc = (logits.argmax(-1) == y).float().mean().item()
    return loss, acc


def train(args) -> Path:
    device = resolve_device(args.device)
    torch.set_num_threads(args.threads)
    if args.deterministic:
        torch.use_deterministic_algorithms(True, warn_only=True)
    torch.manual_seed(args.seed)

    splits = make_splits(
        p=args.p, op=args.op, frac=args.frac, seed=args.seed,
        ood_a_range=tuple(args.ood_a_range) if args.ood_a_range else None,
        ood_b_range=tuple(args.ood_b_range) if args.ood_b_range else None,
    )
    model = build_model(args.p, args.pe, n_layers=args.layers).to(device)
    if args.compile:
        model = torch.compile(model)
    opt = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.wd, betas=(0.9, 0.98), eps=1e-8,
    )

    run_name = args.run_name or run_name_for(args)
    run_dir = Path(args.exp_root) / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "config.json").write_text(json.dumps(vars(args), indent=2))
    ckpt_dir = run_dir / "checkpoints"
    ckpt_dir.mkdir(exist_ok=True)
    schedule = ckpt_schedule(args.epochs) if args.ckpt_every == 0 else None

    x_tr, y_tr = splits.x_train.to(device), splits.y_train.to(device)
    x_val, y_val = splits.x_val.to(device), splits.y_val.to(device)
    x_ood = splits.x_ood.to(device)
    y_ood = splits.y_ood.to(device)
    csv_path = run_dir / "log.csv"
    sustained_evals = 0
    start_epoch = 0
    t_prev = 0.0
    resume_path = run_dir / "resume.pt"
    if resume_path.exists() and not (ckpt_dir / "ckpt_final.pt").exists():
        st = torch.load(resume_path, map_location="cpu")
        model.load_state_dict(st["model"])
        opt.load_state_dict(st["opt"])
        start_epoch = st["epoch"] + 1
        sustained_evals = st["sustained"]
        t_prev = st["elapsed"]
        _trim_csv(csv_path, st["epoch"])
        print(f"[train] resumed {run_name} at epoch {start_epoch}")
    t0 = time.perf_counter()

    new_log = start_epoch == 0 or not csv_path.exists()
    with open(csv_path, "w" if new_log else "a", newline="") as f:
        writer = csv.writer(f)
        if new_log:
            writer.writerow(["epoch", "train_loss", "train_acc", "val_loss", "val_acc", "ood_acc", "seconds"])
        for epoch in range(start_epoch, args.epochs):
            model.train()
            opt.zero_grad(set_to_none=True)
            logits = model(x_tr)[:, -1, :]
            loss = F.cross_entropy(logits, y_tr)
            loss.backward()
            opt.step()

            # Val eval every `eval_every` epochs (compute-cut D10: val forward
            # is ~40% of epoch cost; onset resolution stays within eval_every).
            if epoch % args.eval_every == 0 or epoch == args.epochs - 1:
                model.eval()
                train_acc = (logits.argmax(-1) == y_tr).float().mean().item()
                val_loss, val_acc = evaluate(model, x_val, y_val)
                ood_acc = ""
                if x_ood.shape[0]:
                    _, ood_acc = evaluate(model, x_ood, y_ood)
                writer.writerow(
                    [epoch, f"{loss.item():.6f}", f"{train_acc:.4f}", f"{val_loss:.6f}",
                     f"{val_acc:.4f}", ood_acc if ood_acc == "" else f"{ood_acc:.4f}",
                     f"{t_prev + time.perf_counter() - t0:.2f}"]
                )
                sustained_evals = sustained_evals + 1 if val_acc >= VAL_ACC_TARGET else 0

            if args.resume_every and epoch % args.resume_every == 0 and epoch > start_epoch:
                f.flush()
                save_resume(run_dir, model, opt, epoch, sustained_evals,
                            t_prev + time.perf_counter() - t0)

            if args.ckpt_every < 0:
                due = False  # final-only mode (GPU grid non-trajectory runs)
            else:
                due = (epoch in schedule) if schedule is not None else (epoch % args.ckpt_every == 0)
            if due or epoch == args.epochs - 1:
                save_ckpt(model, ckpt_dir / f"ckpt_{epoch:06d}.pt")

            if sustained_evals * args.eval_every >= EARLY_STOP_WINDOW:
                save_ckpt(model, ckpt_dir / f"ckpt_{epoch:06d}.pt")
                break

    save_ckpt(model, ckpt_dir / "ckpt_final.pt")
    return csv_path


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--pe", choices=PE_TYPES, required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--p", type=int, default=113)
    ap.add_argument("--op", choices=("add", "sub", "mul"), default="add")
    ap.add_argument("--frac", type=float, default=0.3)
    ap.add_argument("--layers", type=int, default=1, choices=(1, 2))
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--wd", type=float, default=1.0)
    ap.add_argument("--epochs", type=int, default=40000)
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--ckpt-every", type=int, default=0,
                    help="0 (default) = ~35 log-spaced checkpoints; N>0 = every N epochs; -1 = final-only")
    ap.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    ap.add_argument("--exp-root", default="experiments",
                    help="run-output root (GPU grid uses experiments_gpu — never mix hardware)")
    ap.add_argument("--compile", action="store_true", help="torch.compile the model (A/B in runbook)")
    ap.add_argument("--deterministic", action="store_true", default=True)
    ap.add_argument("--no-deterministic", dest="deterministic", action="store_false")
    ap.add_argument("--eval-every", type=int, default=1,
                    help="run val/OOD eval every N epochs (Day-2 grid uses 5; see decisions.md D10)")
    ap.add_argument("--resume-every", type=int, default=1000,
                    help="save exact-resume state every N epochs (0 disables); caps reboot loss at N epochs")
    ap.add_argument("--ood-a-range", type=int, nargs=2, default=None,
                    help="hold out all pairs with a in [LO, HI] as OOD eval (proposal: 100 112)")
    ap.add_argument("--ood-b-range", type=int, nargs=2, default=None,
                    help="alternative OOD: hold out pairs with b in [LO, HI] (S-H3)")
    ap.add_argument("--run-name", default=None)
    return ap.parse_args(argv)


if __name__ == "__main__":
    train(parse_args())
