import numpy as np
import pandas as pd

from src.analysis import (
    RUN_RE,
    attention_split_delta,
    setting_of,
    sinusoidal_delay_rank,
    stratified_permutation,
    SIG_KEYS,
)


def test_run_re_new_variants():
    assert RUN_RE.match("p97_add_f0.3_rope_L1_wd1.0_s2")["p"] == "97"
    assert RUN_RE.match("p113_mul_f0.3_nope_L2_wd1.0_s0")["op"] == "mul"
    m = RUN_RE.match("p113_add_f0.3_learned_L1_wd1.0_s0_oodb")
    assert m and m["oodb"] == "_oodb"
    assert setting_of(RUN_RE.match("p97_sub_f0.3_alibi_L2_wd1.0_s1")) == "p97_sub_L2"


def test_sinusoidal_delay_rank():
    df = pd.DataFrame({
        "pe": ["nope", "rope", "alibi", "learned", "sinusoidal"],
        "onset": [10000, 11000, 9000, 12000, 21000],
        "censored": [False] * 5,
    })
    r = sinusoidal_delay_rank(df)
    assert r["sinusoidal_rank"] == 5 and r["sinusoidal_is_slowest"]


def test_attention_split_direction():
    df = pd.DataFrame({
        "pe": ["rope", "alibi", "nope", "sinusoidal", "learned"],
        "uniform_ablation_delta": [1.2, 1.9, 4.0, 4.2, 5.6],
    })
    r = attention_split_delta(df)
    assert r["direction_holds"]
    assert r["attention_level_median"] < r["embedding_level_median"]


def test_stratified_permutation_detects_and_nulls():
    rng = np.random.default_rng(0)
    rows = []
    for setting in ("A", "B"):
        for pe in ("nope", "learned", "sinusoidal", "rope", "alibi"):
            for _ in range(3):
                rows.append({"setting": setting, "pe": pe,
                             **{k: rng.normal() for k in SIG_KEYS}})
    df = pd.DataFrame(rows)
    assert stratified_permutation(df, n_perm=300)["p_perm"] > 0.05
    df2 = df.copy()
    for k in SIG_KEYS[:3]:  # consistent multi-metric effect in both settings
        df2.loc[df2.pe == "sinusoidal", k] += 12
    assert stratified_permutation(df2, n_perm=500)["p_perm"] < 0.05
