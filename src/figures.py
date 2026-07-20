"""Figure/table pipeline: regenerates every artifact from raw data.

    python -m src.figures --all      # everything that has data; skips the rest

Outputs: paper/figures/*.pdf|png, research/day2/results/table*.{csv,md}
Style: AAAI single-column (3.3 in), vector PDF + PNG preview, CVD-validated
palette (Okabe-Ito subset; nope=#0072B2 learned=#E69F00 sinusoidal=#56B4E9
rope=#009E73 alibi=#D55E00), fixed hue per PE (never cycled), distinct
linestyles as secondary encoding, direct labels where space allows.
"""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import src.analysis as analysis
from src.analysis import RESULTS, RUN_RE, collect_h1, read_log

FIGDIR = Path("paper/figures")
EXP = Path("experiments")
PE_ORDER = ["nope", "learned", "sinusoidal", "rope", "alibi"]
PE_COLOR = {"nope": "#0072B2", "learned": "#E69F00", "sinusoidal": "#56B4E9",
            "rope": "#009E73", "alibi": "#D55E00"}
PE_LS = {"nope": "-", "learned": "--", "sinusoidal": "-.", "rope": ":", "alibi": (0, (3, 1, 1, 1))}
PE_LABEL = {"nope": "NoPE", "learned": "Learned", "sinusoidal": "Sinusoidal",
            "rope": "RoPE", "alibi": "ALiBi"}

plt.rcParams.update({
    "font.size": 8, "axes.titlesize": 8, "axes.labelsize": 8,
    "legend.fontsize": 7, "xtick.labelsize": 7, "ytick.labelsize": 7,
    "pdf.fonttype": 42, "ps.fonttype": 42,  # embed TrueType
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.alpha": 0.25, "grid.linewidth": 0.4,
    "lines.linewidth": 1.4,
})
COL_W = 3.3  # inches, AAAI column


def save(fig, name: str):
    FIGDIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGDIR / f"{name}.pdf", bbox_inches="tight")
    fig.savefig(FIGDIR / f"{name}.png", dpi=220, bbox_inches="tight")
    plt.close(fig)
    print(f"[figures] wrote {name}.pdf/.png")


def core_runs():
    """Core-condition run dirs; a finished _x40 extension replaces its original."""
    chosen = {}
    for run_dir in sorted(EXP.iterdir()):
        m = RUN_RE.match(run_dir.name)
        if not (m and not m["ood"] and m["layers"] == "1" and m["op"] == "add"
                and m["p"] == "113" and m["frac"] == "0.3" and (run_dir / "log.csv").exists()):
            continue
        key = (m["pe"], int(m["seed"]))
        is_ext = bool(m["ext"])
        ext_done = is_ext and (run_dir / "checkpoints" / "ckpt_final.pt").exists()
        if key not in chosen:
            if not is_ext or ext_done:
                chosen[key] = run_dir
        elif ext_done:
            chosen[key] = run_dir
    return [(d, pe, seed) for (pe, seed), d in chosen.items()]


def fig1_dynamics():
    """Val-accuracy curves per PE: median with IQR band across seeds."""
    runs = core_runs()
    if not runs:
        return print("[figures] SKIP fig1 (no core runs)")
    fig, ax = plt.subplots(figsize=(COL_W, 2.3))
    grid = np.unique(np.round(np.logspace(0, np.log10(15000), 240)).astype(int))
    for pe in PE_ORDER:
        curves = []
        for run_dir, rpe, _ in runs:
            if rpe != pe:
                continue
            df = read_log(run_dir)
            if len(df) < 2:
                continue
            acc = np.interp(grid, df["epoch"], df["val_acc"],
                            right=df["val_acc"].iloc[-1])
            curves.append(acc)
        if not curves:
            continue
        C = np.vstack(curves)
        med = np.median(C, axis=0)
        ax.fill_between(grid, np.percentile(C, 25, axis=0), np.percentile(C, 75, axis=0),
                        color=PE_COLOR[pe], alpha=0.18, linewidth=0)
        ax.plot(grid, med, color=PE_COLOR[pe], linestyle=PE_LS[pe] if isinstance(PE_LS[pe], str) else "-",
                label=f"{PE_LABEL[pe]} (n={len(curves)})")
    ax.set_xscale("log")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Validation accuracy")
    ax.set_ylim(0, 1.02)
    ax.axhline(0.99, color="0.4", lw=0.6, ls=":")
    if ax.get_legend_handles_labels()[0]:
        ax.legend(loc="upper left", frameon=False)
    save(fig, "fig1_dynamics")


def fig2_onset():
    """Survival curves: fraction not-yet-grokked vs epoch; censored runs end high."""
    h1 = collect_h1()
    core = h1[(~h1.ood) & (h1.layers == "1") & (h1.op == "add") & (h1.p == "113")]
    if core.empty:
        return print("[figures] SKIP fig2 (no H1 rows)")
    fig, ax = plt.subplots(figsize=(COL_W, 2.3))
    for pe in PE_ORDER:
        sub = core[core.pe == pe]
        if sub.empty:
            continue
        times = np.where(sub.censored, sub.last_epoch, sub.onset.fillna(sub.last_epoch)).astype(float)
        events = (~sub.censored).to_numpy()
        order = np.argsort(times)
        t_sorted, e_sorted = times[order], events[order]
        frac = 1.0
        xs, ys = [0], [1.0]
        for t, e in zip(t_sorted, e_sorted):
            xs.append(t)
            ys.append(frac)
            if e:
                frac -= 1 / len(sub)
            xs.append(t)
            ys.append(frac)
        cens_n = int(sub.censored.sum())
        ax.plot(xs, ys, color=PE_COLOR[pe], drawstyle="default",
                label=f"{PE_LABEL[pe]}" + (f" ({cens_n} cens.)" if cens_n else ""))
        for t in times[~events]:
            ax.plot(t, ys[-1], marker="x", color=PE_COLOR[pe], ms=5)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Fraction not yet grokked")
    ax.set_ylim(-0.02, 1.05)
    ax.legend(frameon=False, loc="lower left")
    save(fig, "fig2_onset_survival")


def fig3_trajectories():
    path = RESULTS / "h2_trajectories.csv"
    if not path.exists():
        return print("[figures] SKIP fig3 (no trajectory CSV)")
    df = pd.read_csv(path)
    df = df[(~df.ood) & (df.epoch >= 0)]
    if df.empty:
        return print("[figures] SKIP fig3 (empty)")
    fig, ax = plt.subplots(figsize=(COL_W, 2.3))
    for pe in PE_ORDER:
        sub = df[df.pe == pe]
        if sub.empty:
            continue
        med = sub.groupby("epoch")["embed_fourier_conc"].median()
        ax.plot(med.index.to_numpy() + 1, med.to_numpy(), color=PE_COLOR[pe], label=PE_LABEL[pe])
    ax.set_xscale("log")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Embedding Fourier\nconcentration (top-5)")
    ax.set_ylim(0, 1.02)
    ax.legend(frameon=False, loc="upper left")
    save(fig, "fig3_signature_trajectories")


def fig4_signatures():
    path = RESULTS / "h2_final_signatures.csv"
    if not path.exists():
        return print("[figures] SKIP fig4 (no signatures CSV)")
    df = pd.read_csv(path)
    df = df[(df.epoch == -1) & (~df.ood)]
    if df.empty:
        return print("[figures] SKIP fig4 (empty)")
    fig, ax = plt.subplots(figsize=(COL_W, 2.5))
    for pe in PE_ORDER:
        sub = df[df.pe == pe]
        if sub.empty:
            continue
        ax.scatter(sub["embed_fourier_conc"], sub["uniform_ablation_delta"],
                   color=PE_COLOR[pe], label=PE_LABEL[pe], s=22, edgecolors="white", linewidths=0.6)
    ax.set_xlabel("Embedding Fourier concentration (top-5)")
    ax.set_ylabel("Uniform-attention\nablation $\\Delta$loss")
    ax.legend(frameon=False)
    save(fig, "fig4_final_signatures")


def fig5_ood():
    sig_path = RESULTS / "h2_final_signatures.csv"
    h3_path = RESULTS / "h3_ood.json"
    if not (sig_path.exists() and h3_path.exists()):
        return print("[figures] SKIP fig5 (missing H3 inputs)")
    h3 = json.loads(h3_path.read_text())
    accs = h3.get("ood_acc_by_run", {})
    if not accs:
        return print("[figures] SKIP fig5 (no OOD accuracies)")
    df = pd.read_csv(sig_path)
    df = df[(df.epoch == -1) & (df.ood)].set_index("run")
    common = [r for r in accs if r in df.index]
    if not common:
        return print("[figures] SKIP fig5 (no overlap)")
    fig, ax = plt.subplots(figsize=(COL_W, 2.5))
    for r in common:
        pe = df.loc[r, "pe"]
        ax.scatter(df.loc[r, "embed_fourier_conc"], accs[r], color=PE_COLOR[pe],
                   s=26, edgecolors="white", linewidths=0.6)
    for pe in sorted({df.loc[r, "pe"] for r in common}):
        ax.scatter([], [], color=PE_COLOR[pe], label=PE_LABEL[pe], s=26)
    ax.set_xlabel("Embedding Fourier concentration (final)")
    ax.set_ylabel("OOD accuracy (held-out rows)")
    ax.legend(frameon=False)
    save(fig, "fig5_ood_vs_signature")


def table1():
    h1 = collect_h1()
    core = h1[(~h1.ood) & (h1.layers == "1") & (h1.op == "add") & (h1.p == "113")]
    if core.empty:
        return print("[figures] SKIP table1")
    sig_path = RESULTS / "h2_final_signatures.csv"
    sig = pd.read_csv(sig_path) if sig_path.exists() else pd.DataFrame()
    rows = []
    for pe in PE_ORDER:
        sub = core[core.pe == pe]
        if sub.empty:
            continue
        onset = sub.loc[~sub.censored, "onset"]
        row = dict(
            PE=PE_LABEL[pe], n=len(sub), censored=int(sub.censored.sum()),
            median_onset=float(onset.median()) if len(onset) else None,
            median_sharpness=float(sub.loc[~sub.censored, "sharpness"].median()) if len(onset) else None,
        )
        if not sig.empty:
            fs = sig[(sig.pe == pe) & (sig.epoch == -1) & (~sig.ood)]
            if not fs.empty:
                row["embed_fourier_conc"] = round(float(fs["embed_fourier_conc"].median()), 3)
                row["uniform_abl_delta"] = round(float(fs["uniform_ablation_delta"].median()), 3)
                row["restricted_loss"] = round(float(fs["restricted_loss"].median()), 3)
        rows.append(row)
    t = pd.DataFrame(rows)
    RESULTS.mkdir(parents=True, exist_ok=True)
    t.to_csv(RESULTS / "table1_per_pe.csv", index=False)
    (RESULTS / "table1_per_pe.md").write_text(t.to_markdown(index=False))
    print("[figures] wrote table1")


def table2():
    h1 = collect_h1()
    abl = h1[h1.ood | (h1.layers == "2") | (h1.p == "97") | (h1.op != "add")]
    if abl.empty:
        return print("[figures] SKIP table2 (no ablation runs yet — Pass C)")
    g = abl.groupby(["p", "op", "layers", "ood", "pe"]).agg(
        n=("run", "count"), censored=("censored", "sum"), median_onset=("onset", "median"),
    ).reset_index()
    g.to_csv(RESULTS / "table2_ablations.csv", index=False)
    (RESULTS / "table2_ablations.md").write_text(g.to_markdown(index=False))
    print("[figures] wrote table2")


def main():
    global RESULTS, EXP, FIGDIR
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--exp-root", default=None, help="run-output root (GPU: experiments_gpu)")
    ap.add_argument("--out", default=None, help="results dir for tables/CSVs (GPU: research/week/results)")
    ap.add_argument("--figdir", default=None, help="figure output dir (GPU: research/week/results/figures)")
    args = ap.parse_args()
    if args.exp_root:
        EXP = analysis.EXP = Path(args.exp_root)  # collect_h1 reads analysis.EXP
    if args.out:
        RESULTS = analysis.RESULTS = Path(args.out)
    if args.figdir:
        FIGDIR = Path(args.figdir)
    elif args.out:
        FIGDIR = RESULTS / "figures"
    if args.all:
        fig1_dynamics()
        fig2_onset()
        fig3_trajectories()
        fig4_signatures()
        fig5_ood()
        table1()
        table2()


if __name__ == "__main__":
    main()
