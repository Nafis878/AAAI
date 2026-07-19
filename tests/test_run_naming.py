from src.analysis import RUN_RE


def test_run_re_variants():
    base = RUN_RE.match("p113_add_f0.3_sinusoidal_L1_wd1.0_s2")
    assert base and base["pe"] == "sinusoidal" and not base["ext"] and not base["ood"]

    ext = RUN_RE.match("p113_add_f0.3_sinusoidal_L1_wd1.0_s2_x40")
    assert ext and ext["ext"] == "_x40" and not ext["ood"]

    ood = RUN_RE.match("p113_add_f0.3_rope_L1_wd1.0_s0_ood")
    assert ood and ood["ood"] == "_ood" and not ood["ext"]

    assert RUN_RE.match("smoke_learned_s0") is None


def test_extend40_preset_targets():
    from src.launch import jobs_for

    jobs = list(jobs_for("extend40"))
    assert len(jobs) == 9
    for j in jobs:
        assert "--run-name" in j and j[j.index("--run-name") + 1].endswith("_x40")
        assert j[j.index("--epochs") + 1] == "40000"
    sin = [j for j in jobs if "sinusoidal" in " ".join(j)]
    assert len(sin) == 4  # all sinusoidal seeds extended
