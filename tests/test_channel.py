"""CPU checks on tiny random models (no download needed)."""
import torch

from fvc.channel import MESSAGES, make_message, run_pair
from fvc.model import BITS, FrozenVocabLM

TINY = dict(vocab_size=97, hidden_size=640, intermediate_size=64, num_hidden_layers=2, num_attention_heads=1,
            num_key_value_heads=1, head_dim=16, sliding_window=8)


def tiny(arm, seed):
    torch.manual_seed(seed)
    m = FrozenVocabLM(arm, **TINY)
    m.table.copy_(torch.randint(0, 2, m.table.shape).float() * 2 - 1)
    with torch.no_grad():
        m.seal.copy_(torch.randn(m.seal.shape) * 0.1)
    return m.eval()


def test_shapes_and_all_messages():
    X, Y = tiny("B", 0), tiny("C", 1)
    ids = torch.randint(0, 97, (3, 4 + 5 + 6 + 1))
    res = run_pair(X, Y, ids, s=4, L=5, post=6)
    assert set(res) == set(MESSAGES)
    for v in res.values():
        assert v.shape == (3,) and torch.isfinite(v).all()


def test_message_forms():
    X, Y = tiny("B", 0), tiny("B", 1)
    ids = torch.randint(0, 97, (2, 12))
    soft = make_message("soft", X, Y, ids, 3, 4)
    assert torch.allclose(soft.norm(dim=-1), torch.full((2, 4), BITS ** 0.5), atol=1e-3)
    sign = make_message("sign", X, Y, ids, 3, 4)
    assert set(sign.unique().tolist()) <= {-1.0, 1.0}
    am = make_message("argmax", X, Y, ids, 3, 4)
    assert ((am.unsqueeze(2) == Y.table.view(1, 1, *Y.table.shape)).all(-1).any(-1)).all()
    assert torch.equal(make_message("true", X, Y, ids, 3, 4), Y.table[ids[:, 3:7]])


def test_true_span_matches_plain_forward():
    """With the true span, the receiver's NLL equals an ordinary forward pass over the tokens."""
    X, Y = tiny("B", 0), tiny("B", 1)
    ids = torch.randint(0, 97, (2, 3 + 4 + 5 + 1))
    res = run_pair(X, Y, ids, 3, 4, 5, kinds=("true",))["true"]
    h = Y.hidden(ids[:, :-1])[:, -5:]
    lg = Y.logits_from_codes(Y.out_codes(h))
    ref = torch.nn.functional.cross_entropy(lg.reshape(-1, 97), ids[:, -5:].reshape(-1), reduction="none").view(2, 5).mean(1)
    assert torch.allclose(res, ref, atol=1e-5)
