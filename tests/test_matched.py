import pandas as pd

from src.analysis import matched_postgrok_subset


def h1_row(run, censored=False, onset=8000, last=15000, ood=False, layers="1"):
    return dict(run=run, censored=censored, onset=onset if not censored else None,
                last_epoch=last, ood=ood, layers=layers)


def sig_row(run, epoch=-1, ood=False):
    return dict(run=run, epoch=epoch, ood=ood, pe="nope", seed=0)


def test_matched_selection_rules():
    h1 = pd.DataFrame([
        h1_row("resolved_ok"),                              # in
        h1_row("censored_run", censored=True),              # out: no onset
        h1_row("ood_run", ood=True),                        # out: OOD
        h1_row("depth2_run", layers="2"),                   # out: not core L1
        h1_row("short_tail", onset=14500, last=15000),      # out: <1000 post-onset
    ])
    sig = pd.DataFrame([sig_row(r) for r in h1.run] + [sig_row("resolved_ok", epoch=500)])
    sel = matched_postgrok_subset(h1, sig)
    assert set(sel.run) == {"resolved_ok"}
    assert (sel.epoch == -1).all()  # final checkpoints only


def test_matched_reproduces_paper_number():
    # Integration check against the committed CPU data (skip if results absent).
    from pathlib import Path

    if not Path("research/day2/results/h2_final_signatures.csv").exists():
        import pytest

        pytest.skip("no committed results")
    from src.analysis import matched_test

    res = matched_test(write=False)
    assert res["n"] == 20
    assert abs(res["eta2"] - 0.36788) < 1e-4
    assert res["p_perm"] < 0.0015
    assert all(v == 4 for v in res["per_pe_n"].values())