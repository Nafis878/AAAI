import json
from pathlib import Path

from src.run_week import expand_block, resolve_queue, run_name_of


def test_manifest_totals_and_priority_order():
    manifest = json.loads(Path("research/week/grid_manifest.json").read_text())
    queue = resolve_queue(manifest)
    assert len(queue) == 180
    names = [p for p, _ in queue]
    # priorities appear in non-decreasing order (P0 before P1 before ...)
    assert names == sorted(names, key=lambda n: int(n[1:]))
    assert names[:50] == ["P0"] * 50


def test_expand_block_names_and_flags():
    block = {"p": 97, "op": "add", "layers": 1, "seeds": [0, 1], "epochs": 40000,
             "traj_seeds": [0]}
    args = list(expand_block(block))
    assert len(args) == 10  # 5 PE x 2 seeds
    # traj seed 0 keeps log-spaced ckpts; seed 1 is final-only
    s0 = [a for a in args if a[a.index("--seed") + 1] == "0"][0]
    s1 = [a for a in args if a[a.index("--seed") + 1] == "1"][0]
    assert s0[s0.index("--ckpt-every") + 1] == "0"
    assert s1[s1.index("--ckpt-every") + 1] == "-1"
    assert run_name_of(s0, Path("experiments_gpu")) == "p97_add_f0.3_nope_L1_wd1.0_s0"


def test_ood_blocks_produce_tagged_names():
    manifest = json.loads(Path("research/week/grid_manifest.json").read_text())
    queue = resolve_queue(manifest)
    names = [run_name_of(a, Path("experiments_gpu")) for _, a in queue]
    assert any(n.endswith("_ood") for n in names)
    assert any(n.endswith("_oodb") for n in names)
    assert sum(n.endswith("_oodb") for n in names) == 10  # 5 PE x 2 seeds
