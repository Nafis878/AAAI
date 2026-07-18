import torch

from src.model import build_model
from src.train import ckpt_schedule, load_ckpt, save_ckpt


def test_schedule_density_and_shape():
    sched = sorted(ckpt_schedule(15000))
    assert 30 <= len(sched) <= 40
    assert sched[0] == 0
    assert sched[-1] >= 14000
    early = [e for e in sched if e <= 1000]
    late = [e for e in sched if e > 10000]
    assert len(early) > len(late)  # log-spaced: dense early, sparse late


def test_fp16_roundtrip(tmp_path):
    torch.manual_seed(0)
    model = build_model(113, "rope")
    tokens = torch.randint(0, 113, (4, 3))
    tokens[:, 2] = 113
    before = model(tokens)

    path = tmp_path / "ckpt.pt"
    save_ckpt(model, path)
    assert path.stat().st_size < 1_500_000  # fp16 keeps snapshots small (disk budget)

    torch.manual_seed(1)
    fresh = build_model(113, "rope")
    load_ckpt(fresh, path)
    after = fresh(tokens)
    assert torch.allclose(before, after, atol=2e-2)  # fp16 quantization tolerance
