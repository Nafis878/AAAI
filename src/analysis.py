"""Pre-registered analysis for the PE x grokking experiments (proposal §2 metrics).

H1: onset/sharpness extraction + group stats (Kruskal-Wallis, pairwise
    Mann-Whitney with Holm correction, k-group log-rank incl. censored runs).
H2: circuit signatures per checkpoint (embedding Fourier concentration,
    logit-level restricted/excluded loss [decisions D12], attention entropy,
    uniform-attention ablation delta).
H3: OOD accuracy vs signatures (leave-one-out, modest-n honest framing).

CLI:
    python -m src.analysis --h1                 # onset table + stats -> results dir
    python -m src.analysis --h2                 # signatures for all finished runs
    python -m src.analysis --h3                 # OOD analysis (needs _ood runs)
    python -m src.analysis --all
Outputs: research/day2/results/*.csv|md  (regenerable; raw data untouched)
"""

import argparse
import itertools
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from src.data import make_splits
from src.model import ModelConfig, Transformer
from src.train import load_ckpt

RESULTS = Path("research/day2/results")
VAL_TARGET = 0.99
SUSTAIN = 1000  # epochs, per proposal
TOPK = 5  # key frequencies

# ---------------------------------------------------------------- H1: dynamics


def read_log(run_dir: Path) -> pd.DataFrame:
    """Live runs keep log.csv open with an OS write buffer, so the file can be
    empty or end mid-row for ~15 min after launch — treat both as 'no data yet'."""
    cols = ["epoch", "train_loss", "train_acc", "val_loss", "val_acc", "ood_acc", "seconds"]
    try:
        df = pd.read_csv(run_dir / "log.csv", on_bad_lines="skip")
    except (pd.errors.EmptyDataError, OSError):
        return pd.DataFrame(columns=cols)
    return df.dropna(subset=["epoch", "val_acc"]) if "val_acc" in df else pd.DataFrame(columns=cols)


def onset_metrics(df: pd.DataFrame, cap: int) -> dict:
    """Onset = first logged epoch from which val_acc >= target holds through
    epoch+SUSTAIN (or through end-of-log for early-stopped runs). Censored if
    no such epoch and the run reached the cap."""
    ep = df["epoch"].to_numpy()
    va = df["val_acc"].to_numpy()
    onset = None
    ok = va >= VAL_TARGET
    for i in np.flatnonzero(ok):
        j = np.searchsorted(ep, ep[i] + SUSTAIN, side="right")
        if ok[i:j].all() and (j < len(ep) or ep[-1] >= ep[i] + SUSTAIN - 5 or _early_stopped(df, cap)):
            onset = int(ep[i])
            break
    # sharpness: epochs from 10% -> 99% crossing
    e10 = ep[np.argmax(va >= 0.10)] if (va >= 0.10).any() else None
    e99 = ep[np.argmax(va >= VAL_TARGET)] if ok.any() else None
    sharp = (int(e99) - int(e10)) if (onset is not None and e10 is not None) else None
    # max slope of val acc vs log10(epoch)
    with np.errstate(divide="ignore"):
        le = np.log10(np.maximum(ep, 1))
    dacc = np.diff(va) / np.maximum(np.diff(le), 1e-9)
    max_slope = float(dacc.max()) if len(dacc) else 0.0
    return dict(
        onset=onset,
        censored=onset is None,
        sharpness=sharp,
        max_slope=max_slope,
        final_val_acc=float(va[-1]),
        last_epoch=int(ep[-1]),
    )


def _early_stopped(df: pd.DataFrame, cap: int) -> bool:
    return df["epoch"].iloc[-1] < cap - 5


RUN_RE = re.compile(
    r"p(?P<p>\d+)_(?P<op>\w+?)_f(?P<frac>[\d.]+)_(?P<pe>\w+?)_L(?P<layers>\d)_wd(?P<wd>[\d.]+)_s(?P<seed>\d+)(?P<ood>_ood)?$"
)


def collect_h1(exp: Path = Path("experiments"), pattern: str = "") -> pd.DataFrame:
    rows = []
    for run_dir in sorted(exp.iterdir()):
        m = RUN_RE.match(run_dir.name)
        if not m or not (run_dir / "log.csv").exists():
            continue
        if pattern and pattern not in run_dir.name:
            continue
        cfg = json.loads((run_dir / "config.json").read_text()) if (run_dir / "config.json").exists() else {}
        df = read_log(run_dir)
        if df.empty:
            continue  # buffered fresh run: no usable rows yet
        met = onset_metrics(df, cap=int(cfg.get("epochs", 15000)))
        done = (run_dir / "checkpoints" / "ckpt_final.pt").exists()
        rows.append(dict(run=run_dir.name, done=done, **m.groupdict(), **met))
    cols = ["run", "done", "p", "op", "frac", "pe", "layers", "wd", "seed", "ood",
            "onset", "censored", "sharpness", "max_slope", "final_val_acc", "last_epoch"]
    out = pd.DataFrame(rows, columns=cols if not rows else None)
    if not out.empty:
        out["seed"] = out["seed"].astype(int)
        out["ood"] = out["ood"].notna() if "ood" in out else False
    return out


def holm(pvals: dict) -> dict:
    items = sorted(pvals.items(), key=lambda kv: kv[1])
    m = len(items)
    out, running = {}, 0.0
    for i, (k, p) in enumerate(items):
        running = max(running, min(1.0, (m - i) * p))
        out[k] = running
    return out


def logrank_k(times: np.ndarray, events: np.ndarray, groups: np.ndarray) -> tuple[float, float, int]:
    """K-group log-rank chi-square test. times = onset or cap; events = 1 if
    grokked (onset observed), 0 if censored."""
    from scipy.stats import chi2

    labels = np.unique(groups)
    k = len(labels)
    obs = np.zeros(k)
    exp = np.zeros(k)
    for t in np.unique(times[events == 1]):
        at_risk = times >= t
        n = at_risk.sum()
        d = ((times == t) & (events == 1)).sum()
        for gi, g in enumerate(labels):
            n_g = (at_risk & (groups == g)).sum()
            d_g = ((times == t) & (events == 1) & (groups == g)).sum()
            obs[gi] += d_g
            exp[gi] += d * n_g / n
    stat = float(np.sum((obs - exp) ** 2 / np.where(exp > 0, exp, 1)))
    df = k - 1
    return stat, float(chi2.sf(stat, df)), df


def h1_stats(df: pd.DataFrame) -> dict:
    from scipy.stats import kruskal, mannwhitneyu

    core = df[(~df.ood) & (df.layers == "1")]
    out = {"n_runs": len(core), "censored_by_pe": core.groupby("pe")["censored"].sum().to_dict(),
           "median_onset_by_pe": {}, "onset_ci_by_pe": {}}
    groups, labels = [], []
    for pe, sub in core.groupby("pe"):
        vals = sub.loc[~sub.censored, "onset"].to_numpy(float)
        if len(vals):
            groups.append(vals)
            labels.append(pe)
            out["median_onset_by_pe"][pe] = float(np.median(vals))
            boot = np.random.default_rng(0).choice(vals, (5000, len(vals))).__array__()
            med = np.median(boot, axis=1)
            out["onset_ci_by_pe"][pe] = [float(np.percentile(med, 2.5)), float(np.percentile(med, 97.5))]
        else:
            out["median_onset_by_pe"][pe] = None  # all censored
    if len(groups) >= 2 and all(len(g) >= 2 for g in groups):
        st, p = kruskal(*groups)
        out["kruskal"] = {"H": float(st), "p": float(p)}
        pw = {}
        for (i, a), (j, b) in itertools.combinations(enumerate(labels), 2):
            if len(groups[i]) >= 2 and len(groups[j]) >= 2:
                st2, p2 = mannwhitneyu(groups[i], groups[j], alternative="two-sided")
                pw[f"{a}_vs_{b}"] = float(p2)
        out["pairwise_mannwhitney_holm"] = holm(pw)
    # log-rank including censored
    times = np.where(core.censored, core.last_epoch, core.onset.fillna(core.last_epoch)).astype(float)
    events = (~core.censored).astype(int).to_numpy()
    if len(np.unique(core.pe)) >= 2 and events.sum() > 0:
        stat, p, dfree = logrank_k(times, events, core.pe.to_numpy())
        out["logrank"] = {"chi2": stat, "p": p, "df": dfree}
    return out


# ------------------------------------------------------- H2: circuit signatures


def model_from_run(run_dir: Path, ckpt: Path) -> tuple[Transformer, dict]:
    cfg = json.loads((run_dir / "config.json").read_text())
    mc = ModelConfig(vocab=cfg["p"] + 1, pe=cfg["pe"], n_layers=cfg["layers"], pre_ln=cfg["layers"] == 2)
    model = Transformer(mc)
    load_ckpt(model, ckpt)
    model.eval()
    return model, cfg


def embed_fourier(model: Transformer, p: int) -> tuple[float, np.ndarray]:
    """Fraction of non-DC embedding power in the top-TOPK frequencies + key freqs."""
    we = model.embed.weight.detach().numpy()[:p]  # (p, d)
    fft = np.fft.rfft(we, axis=0)
    power = (np.abs(fft) ** 2).sum(axis=1)[1:]  # drop DC; freqs 1..p//2
    order = np.argsort(power)[::-1]
    conc = float(power[order[:TOPK]].sum() / power.sum()) if power.sum() > 0 else 0.0
    return conc, order[:TOPK] + 1


def full_grid_logits(model: Transformer, p: int) -> torch.Tensor:
    a, b = np.meshgrid(np.arange(p), np.arange(p), indexing="ij")
    x = np.stack([a.ravel(), b.ravel(), np.full(p * p, p)], axis=1)
    with torch.no_grad():
        return model(torch.from_numpy(x).long())[:, -1, :]  # (p^2, p)


def logit_fourier_metrics(logits: torch.Tensor, p: int, train_idx: np.ndarray) -> dict:
    """Logit-level restricted/excluded loss (D12) + logit spectrum concentration.

    L(a,b,c) -> group by s=(a+b)%p -> M(s,c). FFT over s; key freqs = top-TOPK
    non-DC. restricted logits = DC + key components of M, broadcast to pairs;
    excluded logits = L - (M_key - DC broadcast).
    """
    a = np.repeat(np.arange(p), p)
    b = np.tile(np.arange(p), p)
    s = (a + b) % p
    y = torch.from_numpy(s).long()
    L = logits.numpy()  # (p^2, p)

    M = np.zeros((p, p))
    for r in range(p):
        M[r] = L[s == r].mean(axis=0)
    fft = np.fft.rfft(M, axis=0)
    power = (np.abs(fft) ** 2).sum(axis=1)[1:]
    order = np.argsort(power)[::-1]
    key = order[:TOPK] + 1
    conc = float(power[order[:TOPK]].sum() / power.sum()) if power.sum() > 0 else 0.0

    keep = np.zeros_like(fft)
    keep[0] = fft[0]
    keep[key] = fft[key]
    M_key = np.fft.irfft(keep, n=p, axis=0)  # (s, c) restricted component

    restricted = torch.from_numpy(M_key[s]).float()
    rest_loss = float(F.cross_entropy(restricted, y))
    excluded = torch.from_numpy(L - (M_key[s] - M.mean(axis=0))).float()
    exc_loss_train = float(F.cross_entropy(excluded[train_idx], y[train_idx]))
    full_loss_train = float(F.cross_entropy(torch.from_numpy(L[train_idx]).float(), y[train_idx]))
    return dict(
        logit_fourier_conc=conc,
        key_freqs=key.tolist(),
        restricted_loss=rest_loss,
        excluded_loss=exc_loss_train,
        train_loss_full=full_loss_train,
    )


def attention_metrics(model: Transformer, p: int, sample: int = 512) -> dict:
    rng = np.random.default_rng(0)
    a = rng.integers(0, p, sample)
    b = rng.integers(0, p, sample)
    x = torch.from_numpy(np.stack([a, b, np.full(sample, p)], axis=1)).long()
    attn0 = model.blocks[0].attn
    attn0.store_attn = True
    with torch.no_grad():
        logits = model(x)[:, -1, :]
        logits_uni = model(x, uniform_attn=True)[:, -1, :]
    attn0.store_attn = False
    pat = attn0.last_attn[:, :, -1, :]  # (B, H, T) attention from '='
    ent = -(pat * (pat + 1e-12).log()).sum(-1).mean().item()
    y = torch.from_numpy((a + b) % p).long()
    delta = float(F.cross_entropy(logits_uni, y) - F.cross_entropy(logits, y))
    return dict(attn_entropy_eq=ent, uniform_ablation_delta=delta)


SIG_KEYS = ["embed_fourier_conc", "logit_fourier_conc", "restricted_loss",
            "excluded_loss", "attn_entropy_eq", "uniform_ablation_delta"]


def signatures_for_ckpt(run_dir: Path, ckpt: Path) -> dict:
    model, cfg = model_from_run(run_dir, ckpt)
    p = cfg["p"]
    splits = make_splits(p=p, op=cfg["op"], frac=cfg["frac"], seed=cfg["seed"],
                         ood_a_range=tuple(cfg["ood_a_range"]) if cfg.get("ood_a_range") else None)
    a_tr = splits.x_train[:, 0].numpy()
    b_tr = splits.x_train[:, 1].numpy()
    train_idx = a_tr * p + b_tr
    conc, _ = embed_fourier(model, p)
    logits = full_grid_logits(model, p)
    out = dict(embed_fourier_conc=conc)
    out.update(logit_fourier_metrics(logits, p, train_idx))
    out.update(attention_metrics(model, p))
    return out


def collect_h2(exp: Path = Path("experiments"), trajectories: bool = False, pattern: str = "") -> pd.DataFrame:
    rows = []
    for run_dir in sorted(exp.iterdir()):
        m = RUN_RE.match(run_dir.name)
        ck_dir = run_dir / "checkpoints"
        if not m or not (ck_dir / "ckpt_final.pt").exists():
            continue
        if pattern and pattern not in run_dir.name:
            continue
        ckpts = [ck_dir / "ckpt_final.pt"]
        if trajectories:
            ckpts = sorted(ck_dir.glob("ckpt_0*.pt")) + ckpts
        for ck in ckpts:
            epoch = -1 if ck.name == "ckpt_final.pt" else int(ck.stem.split("_")[1])
            sig = signatures_for_ckpt(run_dir, ck)
            rows.append(dict(run=run_dir.name, pe=m["pe"], seed=int(m["seed"]),
                             ood=bool(m["ood"]), epoch=epoch, **sig))
    return pd.DataFrame(rows)


def permutation_test_signatures(df: pd.DataFrame, n_perm: int = 10000) -> dict:
    """PE label vs final-signature vectors: between/total variance statistic."""
    X = df[SIG_KEYS].to_numpy(float)
    X = (X - X.mean(0)) / np.where(X.std(0) > 0, X.std(0), 1)
    g = df["pe"].to_numpy()

    def stat(labels):
        grand = X.mean(0)
        between = sum((labels == u).sum() * ((X[labels == u].mean(0) - grand) ** 2).sum()
                      for u in np.unique(labels))
        total = ((X - grand) ** 2).sum()
        return between / total if total > 0 else 0.0

    obs = stat(g)
    rng = np.random.default_rng(0)
    null = [stat(rng.permutation(g)) for _ in range(n_perm)]
    p = float((np.sum(np.asarray(null) >= obs) + 1) / (n_perm + 1))
    return {"eta2": float(obs), "p_perm": p, "n": len(df)}


# ---------------------------------------------------------------------- H3: OOD


def h3_analysis(h1df: pd.DataFrame, h2df: pd.DataFrame) -> dict:
    """OOD accuracy predicted from signatures vs from PE label (leave-one-out).
    Modest-n: report R^2 comparison + rank correlation, flagged as supporting."""
    ood_runs = h1df[h1df.ood].copy()
    if ood_runs.empty:
        return {"note": "no OOD runs found"}
    acc = {}
    for _, r in ood_runs.iterrows():
        df = read_log(Path("experiments") / r["run"])
        if "ood_acc" in df and df["ood_acc"].notna().any():
            acc[r["run"]] = float(df["ood_acc"].dropna().iloc[-1])
    sig = h2df[h2df.ood & (h2df.epoch == -1)].set_index("run")
    common = [r for r in acc if r in sig.index]
    if len(common) < 4:
        return {"note": f"only {len(common)} OOD runs with signatures — insufficient"}
    y = np.array([acc[r] for r in common])
    X = sig.loc[common, SIG_KEYS].to_numpy(float)
    X = (X - X.mean(0)) / np.where(X.std(0) > 0, X.std(0), 1)
    pe = sig.loc[common, "pe"].to_numpy()

    def loo_pred(predict):
        preds = []
        for i in range(len(y)):
            mask = np.arange(len(y)) != i
            preds.append(predict(mask, i))
        return np.asarray(preds)

    def r2(pred):
        ss = ((y - pred) ** 2).sum()
        return 1 - ss / ((y - y.mean()) ** 2).sum()

    # ridge on signatures
    lam = 1.0

    def sig_pred(mask, i):
        Xm, ym = X[mask], y[mask]
        w = np.linalg.solve(Xm.T @ Xm + lam * np.eye(X.shape[1]), Xm.T @ (ym - ym.mean()))
        return float(X[i] @ w + ym.mean())

    # PE-label group mean
    def pe_pred(mask, i):
        same = mask & (pe == pe[i])
        return float(y[same].mean()) if same.any() else float(y[mask].mean())

    from scipy.stats import spearmanr

    r2_sig = float(r2(loo_pred(sig_pred)))
    r2_pe = float(r2(loo_pred(pe_pred)))
    rho = spearmanr(X @ np.linalg.lstsq(X, y - y.mean(), rcond=None)[0], y)
    return {"n": len(common), "loo_r2_signatures": r2_sig, "loo_r2_pe_label": r2_pe,
            "spearman_sig_vs_ood": float(rho.statistic), "ood_acc_by_run": acc,
            "note": "modest-n supporting evidence (proposal H3), not standalone"}


# ------------------------------------------------------------------------- CLI


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--h1", action="store_true")
    ap.add_argument("--h2", action="store_true")
    ap.add_argument("--h3", action="store_true")
    ap.add_argument("--trajectories", action="store_true", help="H2 per-checkpoint (slow)")
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--pattern", default="", help="substring filter on run names")
    args = ap.parse_args()
    RESULTS.mkdir(parents=True, exist_ok=True)

    if args.h1 or args.all or args.h3:
        h1df = collect_h1(pattern=args.pattern)
        h1df.to_csv(RESULTS / "h1_runs.csv", index=False)
        if not h1df.empty:
            stats = h1_stats(h1df)
            (RESULTS / "h1_stats.json").write_text(json.dumps(stats, indent=2, default=str))
            print("H1:", json.dumps(stats, indent=2, default=str))
    if args.h2 or args.all or args.h3:
        h2df = collect_h2(trajectories=args.trajectories, pattern=args.pattern)
        name = "h2_trajectories.csv" if args.trajectories else "h2_final_signatures.csv"
        h2df.to_csv(RESULTS / name, index=False)
        final = h2df[(h2df.epoch == -1) & (~h2df.ood)] if not h2df.empty else h2df
        if len(final) >= 8:
            perm = permutation_test_signatures(final)
            (RESULTS / "h2_permutation.json").write_text(json.dumps(perm, indent=2))
            print("H2 permutation:", perm)
    if args.h3 or args.all:
        res = h3_analysis(collect_h1(), collect_h2())
        (RESULTS / "h3_ood.json").write_text(json.dumps(res, indent=2, default=str))
        print("H3:", json.dumps(res, indent=2, default=str))


if __name__ == "__main__":
    main()
