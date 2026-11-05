"""The code-space channel between two frozen-vocabulary models.

A sender X reads the true text. For each position t of a span, X's output vector out(h_X[t-1]) is its guess of
token t, written in the shared 512-d code space. The receiver Y reads the same text with the span replaced by a
message, and is scored on the tokens after the span. Messages:

  true    the real span tokens (table rows): upper reference
  soft    X's raw output vectors, rescaled to the norm of a code (sqrt 512): the zero-translator channel
  sign    sign(X's output vector): a 512-bit message, possibly not any token's code
  mix     X's next-token distribution averaged over the token codes, sum_v p_X(v) code(v): a soft token
  argmax  the code of X's top-scoring token: the ordinary text channel, the baseline to beat
  random  random +-1 codes: floor
  drop    no span at all (prefix and post are joined): floor without noise
"""
import math

import torch
import torch.nn.functional as F

from .model import BITS

MESSAGES = ("true", "soft", "sign", "mix", "argmax", "random", "drop")


@torch.no_grad()
def sender_codes(X, ids):
    """X's output vectors at every position: out[:, t] is X's guess of token t+1."""
    return X.out_codes(X.hidden(ids)).float()


def make_message(kind, X, Y, ids, s, L, gen=None):
    """Code vectors for span positions s..s+L-1, shape (B, L, 512). X sees ids[:, :s+L]."""
    if kind == "true":
        return Y.table[ids[:, s:s + L]]
    c = sender_codes(X, ids[:, :s + L])[:, s - 1:s + L - 1]          # guesses of tokens s..s+L-1
    if kind == "soft":
        return c * (math.sqrt(BITS) / c.norm(dim=-1, keepdim=True).clamp_min(1e-6))
    if kind == "sign":
        return torch.where(c >= 0, 1.0, -1.0)
    if kind == "mix":
        return torch.softmax(X.logits_from_codes(c).float(), -1) @ Y.table.float()
    if kind == "argmax":
        return Y.table[X.logits_from_codes(c).argmax(-1)]
    if kind == "random":
        r = torch.randint(0, 2, c.shape, generator=gen, device="cpu").to(c.device)
        return r.float() * 2 - 1
    raise ValueError(kind)


@torch.no_grad()
def receiver_nll(Y, ids, s, L, msg, post):
    """Mean NLL of Y on the `post` tokens after the span, given prefix ids[:, :s], the message, and the true post
    tokens as they arrive. msg=None drops the span."""
    pre = Y.table[ids[:, :s]]
    after = ids[:, s + L:s + L + post + 1]
    parts = [pre] + ([msg.to(pre.dtype)] if msg is not None else []) + [Y.table[after[:, :-1]]]
    codes = torch.cat(parts, 1)
    h = Y.hidden_from_codes(codes)[:, -post:]
    logits = Y.logits_from_codes(Y.out_codes(h)).float()
    return F.cross_entropy(logits.reshape(-1, logits.shape[-1]), after[:, 1:].reshape(-1), reduction="none") \
        .view(after.shape[0], post)


def run_pair(X, Y, ids, s, L, post, kinds=MESSAGES, seed=0):
    """Per-sequence mean NLL for each message kind. ids: (B, s+L+post+1)."""
    gen = torch.Generator().manual_seed(seed)
    out = {}
    for k in kinds:
        msg = None if k == "drop" else make_message(k, X, Y, ids, s, L, gen)
        out[k] = receiver_nll(Y, ids, s, L, msg, post).mean(1)
    return out
