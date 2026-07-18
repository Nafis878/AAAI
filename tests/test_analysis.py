import numpy as np
import pandas as pd
import torch

from src.analysis import (
    embed_fourier,
    holm,
    logit_fourier_metrics,
    logrank_k,
    onset_metrics,
    permutation_test_signatures,
    SIG_KEYS,
)
from src.model import build_model


def synth_log(onset: int, cap: int, step: int = 5) -> pd.DataFrame:
    eps = np.arange(0, cap, step)
    acc = np.where(eps < onset, 0.05 + 0.9 * (eps >= onset - 2000) * (eps - (onset - 2000)) / 2000 * 0.5, 1.0)
    acc = np.clip(acc, 0, 1.0)
    acc[eps >= onset] = 0.995
    return pd.DataFrame({"epoch": eps, "val_acc": acc})


def test_onset_detected():
    df = synth_log(onset=8000, cap=15000)
    met = onset_metrics(df, cap=15000)
    assert not met["censored"]
    assert abs(met["onset"] - 8000) <= 5


def test_onset_censored():
    eps = np.arange(0, 15000, 5)
    df = pd.DataFrame({"epoch": eps, "val_acc": np.full(len(eps), 0.3)})
    met = onset_metrics(df, cap=15000)
    assert met["censored"] and met["onset"] is None


def test_holm_known():
    adj = holm({"a": 0.01, "b": 0.04, "c": 0.03})
    assert adj["a"] == 0.03  # 3 * 0.01
    assert adj["b"] >= adj["c"] >= adj["a"]


def test_logrank_separates_groups():
    # group A groks early, group B all censored at 15000
    times = np.array([1000, 1100, 1200, 1300, 15000, 15000, 15000, 15000], float)
    events = np.array([1, 1, 1, 1, 0, 0, 0, 0])
    groups = np.array(["A"] * 4 + ["B"] * 4)
    stat, p, df = logrank_k(times, events, groups)
    assert p < 0.01
    same = np.array(["A", "B"] * 4)
    _, p2, _ = logrank_k(times, events, same)
    assert p2 > 0.05


def test_embed_fourier_single_frequency():
    p = 113
    model = build_model(p, "nope")
    k = 7
    pos = np.arange(p)
    emb = np.zeros((p + 1, 128), dtype=np.float32)
    emb[:p, 0] = np.cos(2 * np.pi * k * pos / p)
    emb[:p, 1] = np.sin(2 * np.pi * k * pos / p)
    with torch.no_grad():
        model.embed.weight.copy_(torch.from_numpy(emb))
    conc, key = embed_fourier(model, p)
    assert conc > 0.99
    assert k in key.tolist()


def test_logit_fourier_on_clock_logits():
    # Perfect 5-frequency 'clock' logits: L(a,b,c) = sum_k cos(w_k(a+b-c)).
    # (One frequency alone leaves near-tie logits — cos(2*pi*k*d/p) ~ 1 for the
    # d inverting k mod p — so, like real grokked nets, use several freqs.)
    p = 113
    keys = [7, 11, 23, 31, 44]
    a = np.repeat(np.arange(p), p)
    b = np.tile(np.arange(p), p)
    c = np.arange(p)
    L = sum(10 * np.cos(2 * np.pi * k / p * ((a + b)[:, None] - c[None, :])) for k in keys)
    train_idx = np.arange(0, p * p, 7)
    met = logit_fourier_metrics(torch.from_numpy(L).float(), p, train_idx)
    assert set(met["key_freqs"]) == set(keys)
    assert met["restricted_loss"] < 0.1  # key freqs alone solve the task
    assert met["excluded_loss"] > met["train_loss_full"] + 1.0  # removing them destroys it


def test_permutation_test_directions():
    rng = np.random.default_rng(0)
    n = 20
    base = {k: rng.normal(size=n) for k in SIG_KEYS}
    df = pd.DataFrame(base)
    df["pe"] = ["a", "b", "c", "d"] * 5
    res_null = permutation_test_signatures(df, n_perm=500)
    assert res_null["p_perm"] > 0.05
    df2 = df.copy()
    df2.loc[df2.pe == "a", SIG_KEYS[0]] += 10  # strong group effect
    res_eff = permutation_test_signatures(df2, n_perm=500)
    assert res_eff["p_perm"] < 0.05
