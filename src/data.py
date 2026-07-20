"""Deterministic mod-p dataset generation for the PE x grokking experiments.

Sequences are [a, b, EQ] with target c = (a op b) mod p read at the EQ position.
Vocab: tokens 0..p-1 are residues, token p is the EQ marker (vocab size p+1).

All splits are pure functions of (p, op, frac, seed) — no files, no downloads.
"""

from dataclasses import dataclass

import numpy as np
import torch

OPS = ("add", "sub", "mul")


def apply_op(a: np.ndarray, b: np.ndarray, op: str, p: int) -> np.ndarray:
    if op == "add":
        return (a + b) % p
    if op == "sub":
        return (a - b) % p
    if op == "mul":
        return (a * b) % p
    raise ValueError(f"unknown op {op!r}, expected one of {OPS}")


def all_pairs(p: int) -> tuple[np.ndarray, np.ndarray]:
    """All p^2 (a, b) pairs in fixed row-major order."""
    a, b = np.meshgrid(np.arange(p), np.arange(p), indexing="ij")
    return a.ravel(), b.ravel()


@dataclass(frozen=True)
class Splits:
    """Token tensors (long): x_* are (N, 3) sequences, y_* are (N,) targets."""

    x_train: torch.Tensor
    y_train: torch.Tensor
    x_val: torch.Tensor
    y_val: torch.Tensor
    x_ood: torch.Tensor  # empty (0, 3) when no OOD holdout requested
    y_ood: torch.Tensor
    p: int
    op: str
    frac: float
    seed: int

    @property
    def eq_token(self) -> int:
        return self.p


def _to_tensors(a: np.ndarray, b: np.ndarray, c: np.ndarray, p: int):
    x = np.stack([a, b, np.full_like(a, p)], axis=1)
    return torch.from_numpy(x).long(), torch.from_numpy(c).long()


def make_splits(
    p: int = 113,
    op: str = "add",
    frac: float = 0.3,
    seed: int = 0,
    ood_a_range: tuple[int, int] | None = None,
    ood_b_range: tuple[int, int] | None = None,
) -> Splits:
    """Build deterministic train/val (and optional OOD) splits.

    ood_a_range=(lo, hi) holds out ALL pairs with lo <= a <= hi from both
    train and val; they become the OOD evaluation set (proposal §2: (100, 112)).
    ood_b_range does the same on the b operand (S-H3 alternative construction).
    The train set is `frac` of the remaining pairs, chosen by a seeded
    permutation; validation is everything else that is not OOD.
    """
    a, b = all_pairs(p)
    c = apply_op(a, b, op, p)

    ood_mask = np.zeros_like(a, dtype=bool)
    if ood_a_range is not None:
        lo, hi = ood_a_range
        if not (0 <= lo <= hi < p):
            raise ValueError(f"ood_a_range {ood_a_range} out of [0, {p})")
        ood_mask |= (a >= lo) & (a <= hi)
    if ood_b_range is not None:
        lo, hi = ood_b_range
        if not (0 <= lo <= hi < p):
            raise ValueError(f"ood_b_range {ood_b_range} out of [0, {p})")
        ood_mask |= (b >= lo) & (b <= hi)

    pool = np.flatnonzero(~ood_mask)
    rng = np.random.default_rng(seed)
    perm = rng.permutation(pool.size)
    n_train = int(frac * pool.size)
    train_idx = pool[perm[:n_train]]
    val_idx = pool[perm[n_train:]]
    ood_idx = np.flatnonzero(ood_mask)

    x_train, y_train = _to_tensors(a[train_idx], b[train_idx], c[train_idx], p)
    x_val, y_val = _to_tensors(a[val_idx], b[val_idx], c[val_idx], p)
    x_ood, y_ood = _to_tensors(a[ood_idx], b[ood_idx], c[ood_idx], p)

    return Splits(x_train, y_train, x_val, y_val, x_ood, y_ood, p, op, frac, seed)
