import pytest
import torch

from src.model import PE_TYPES, build_model


@pytest.mark.parametrize("pe", PE_TYPES)
def test_forward_shapes_all_pes(pe):
    p = 113
    model = build_model(p, pe, n_layers=1)
    tokens = torch.randint(0, p, (8, 3))
    tokens[:, 2] = p  # EQ
    logits = model(tokens)
    assert logits.shape == (8, 3, p)
    assert torch.isfinite(logits).all()


def test_param_counts_in_bracket():
    n1 = build_model(113, "learned", n_layers=1).n_params()
    n2 = build_model(113, "learned", n_layers=2).n_params()
    assert 200_000 <= n1 <= 1_200_000
    assert 400_000 <= n2 <= 1_200_000
    assert n2 > n1


def test_nope_learned_param_difference():
    # learned PE adds exactly seq_len * d_model parameters over nope
    n_nope = build_model(113, "nope").n_params()
    n_learned = build_model(113, "learned").n_params()
    assert n_learned - n_nope == 3 * 128
    # formula-based PEs add no parameters
    assert build_model(113, "rope").n_params() == n_nope
    assert build_model(113, "alibi").n_params() == n_nope
    assert build_model(113, "sinusoidal").n_params() == n_nope


def test_uniform_attention_changes_output():
    model = build_model(113, "learned")
    tokens = torch.randint(0, 113, (4, 3))
    tokens[:, 2] = 113
    normal = model(tokens)
    ablated = model(tokens, uniform_attn=True)
    assert not torch.allclose(normal, ablated)


def test_determinism_under_seed():
    torch.manual_seed(0)
    m1 = build_model(113, "rope")
    torch.manual_seed(0)
    m2 = build_model(113, "rope")
    tokens = torch.randint(0, 113, (4, 3))
    assert torch.equal(m1(tokens), m2(tokens))
