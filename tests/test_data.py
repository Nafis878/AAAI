import numpy as np
import torch

from src.data import all_pairs, apply_op, make_splits


def as_pair_set(x: torch.Tensor) -> set[tuple[int, int]]:
    return {(int(a), int(b)) for a, b in x[:, :2].tolist()}


def test_determinism_across_calls():
    s1 = make_splits(p=113, op="add", frac=0.3, seed=7)
    s2 = make_splits(p=113, op="add", frac=0.3, seed=7)
    assert torch.equal(s1.x_train, s2.x_train)
    assert torch.equal(s1.y_train, s2.y_train)
    assert torch.equal(s1.x_val, s2.x_val)


def test_different_seeds_differ():
    s1 = make_splits(p=113, frac=0.3, seed=0)
    s2 = make_splits(p=113, frac=0.3, seed=1)
    assert not torch.equal(s1.x_train, s2.x_train)


def test_train_val_disjoint_and_full_coverage():
    p = 97
    s = make_splits(p=p, op="add", frac=0.3, seed=3)
    train, val = as_pair_set(s.x_train), as_pair_set(s.x_val)
    assert train.isdisjoint(val)
    assert len(train | val) == p * p
    assert len(s.x_train) == int(0.3 * p * p)


def test_label_correctness_all_ops():
    p = 113
    for op in ("add", "sub", "mul"):
        s = make_splits(p=p, op=op, frac=0.3, seed=0)
        for x, y in [(s.x_train, s.y_train), (s.x_val, s.y_val)]:
            a, b = x[:, 0].numpy(), x[:, 1].numpy()
            expected = apply_op(a, b, op, p)
            assert np.array_equal(y.numpy(), expected)
            assert (x[:, 2] == p).all()  # EQ token


def test_ood_holdout():
    p = 113
    s = make_splits(p=p, frac=0.3, seed=0, ood_a_range=(100, 112))
    ood = as_pair_set(s.x_ood)
    assert len(ood) == 13 * p
    assert all(100 <= a <= 112 for a, _ in ood)
    for split in (as_pair_set(s.x_train), as_pair_set(s.x_val)):
        assert split.isdisjoint(ood)
        assert all(a < 100 for a, _ in split)
    assert len(as_pair_set(s.x_train) | as_pair_set(s.x_val) | ood) == p * p


def test_all_pairs_shape():
    a, b = all_pairs(11)
    assert a.shape == (121,) and b.shape == (121,)
    assert len({(x, y) for x, y in zip(a, b)}) == 121
