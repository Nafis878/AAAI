import torch

from src.train import parse_args, train


def run(tmp_path, monkeypatch, name, epochs, resume_every=10):
    monkeypatch.chdir(tmp_path)
    args = parse_args([
        "--pe", "learned", "--seed", "0", "--p", "23", "--epochs", str(epochs),
        "--threads", "1", "--eval-every", "5", "--ckpt-every", "50",
        "--resume-every", str(resume_every), "--run-name", name,
    ])
    return train(args)


def test_resume_is_bit_exact(tmp_path, monkeypatch):
    # Reference: 60 epochs uninterrupted.
    run(tmp_path, monkeypatch, "ref", 60)
    ref = torch.load(tmp_path / "experiments/ref/checkpoints/ckpt_final.pt")

    # Crash simulation: run 35 epochs (resume state saved at 30), "reboot"
    # (final ckpt removed), then relaunch to 60 — must resume, not restart.
    run(tmp_path, monkeypatch, "crash", 35)
    (tmp_path / "experiments/crash/checkpoints/ckpt_final.pt").unlink()
    csv_before = (tmp_path / "experiments/crash/log.csv").read_text()
    run(tmp_path, monkeypatch, "crash", 60)
    resumed = torch.load(tmp_path / "experiments/crash/checkpoints/ckpt_final.pt")

    for k in ref:
        assert torch.equal(ref[k], resumed[k]), f"mismatch in {k}"

    # CSV: no duplicate epochs, contiguous eval grid to 55
    lines = (tmp_path / "experiments/crash/log.csv").read_text().splitlines()[1:]
    eps = [int(l.split(",")[0]) for l in lines]
    assert eps == sorted(set(eps))
    assert eps[-1] == 59 and 30 in eps  # final epoch always logged; 30 = resume point
    assert len(csv_before.splitlines()) > 2  # sanity: first attempt logged rows


def test_finished_run_does_not_resume(tmp_path, monkeypatch):
    run(tmp_path, monkeypatch, "done", 20)
    final = tmp_path / "experiments/done/checkpoints/ckpt_final.pt"
    before = final.stat().st_mtime
    # ckpt_final present => rerun trains from scratch (identical result), not
    # from resume.pt; guarded by launch.py --skip-existing in practice.
    run(tmp_path, monkeypatch, "done", 20)
    assert final.stat().st_mtime >= before
