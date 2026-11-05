"""Frozen-vocabulary language models (arms B and C of Bathula 2026), rebuilt for loading the released checkpoints.

Adapted from code/train.py and code/fastc.py in huggingface.co/QLNI/embeddings-are-overpriced
(QLNI Non-Commercial Research Licence; Sai Kiran Bathula). Only the parts needed for inference are kept: the
Gemma-3-270M-shaped body, the trainable 512->640 input map and 640->512 output map, the frozen table of 512-bit codes,
the per-token prior ("seal"), and BitNet b1.58 ternary linears for arm C.
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

BITS = 512

# Gemma 3 270M shape as stated in the paper (section 3): 18 layers, width 640, FFN 2,048, 4 query and 1 key-value head
# of size 256, five sliding-window (512) layers per full-attention layer, RoPE base 10k local and 1M global.
GEMMA3_270M = dict(
    vocab_size=262144, hidden_size=640, intermediate_size=2048, num_hidden_layers=18, num_attention_heads=4,
    num_key_value_heads=1, head_dim=256, sliding_window=512, query_pre_attn_scalar=256, rms_norm_eps=1e-6,
    max_position_embeddings=32768, hidden_activation="gelu_pytorch_tanh",
    rope_local_base_freq=10000.0, rope_theta=1000000.0,
)


def body_config(**over):
    from transformers import Gemma3TextConfig
    c = {**GEMMA3_270M, **over}
    n = c["num_hidden_layers"]
    c["layer_types"] = ["full_attention" if (i + 1) % 6 == 0 else "sliding_attention" for i in range(n)]
    local, glob = c.pop("rope_local_base_freq"), c.pop("rope_theta")
    c["rope_parameters"] = {"sliding_attention": {"rope_type": "default", "rope_theta": local},
                            "full_attention": {"rope_type": "default", "rope_theta": glob}}
    return Gemma3TextConfig(**c)


def _ste(x, q):
    return x + (q - x).detach()


class TLinear(nn.Linear):
    """BitNet b1.58 BitLinear, as in train.py: RMSNorm on the input, int8 absmax activations per token, absmean
    ternary weights."""

    def forward(self, x):
        xf = x.float()
        xn = xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + 1e-6)
        sx = 127.0 / xn.detach().abs().amax(-1, keepdim=True).clamp_min(1e-5)
        xq = _ste(xn, (xn * sx).round().clamp(-128, 127) / sx)
        w = self.weight
        sw = 1.0 / w.detach().abs().mean().clamp_min(1e-5)
        wq = _ste(w, (w * sw).round().clamp(-1, 1) / sw)
        return F.linear(xq.to(x.dtype), wq.to(x.dtype), self.bias)


class FrozenVocabLM(nn.Module):
    """Arm B (float body) or C (ternary body). Tokens enter as table[ids] -> inp; logits are out(h) . table / sqrt(512)
    + seal. The 512-bit code space at inp and out is the only space two such models share."""

    def __init__(self, arm="B", **cfg_over):
        super().__init__()
        from transformers import Gemma3TextModel
        cfg = body_config(**cfg_over)
        self.arm = arm
        self.body = Gemma3TextModel(cfg)
        self.body.embed_tokens = None
        V, d = cfg.vocab_size, cfg.hidden_size
        self.register_buffer("table", torch.zeros(V, BITS))
        if arm == "C":                      # C's prior was trained from its unigram start
            self.seal = nn.Parameter(torch.zeros(V))
        else:
            self.register_buffer("seal", torch.zeros(V))
        self.register_buffer("table_mu", torch.zeros(640))
        self.register_buffer("table_W", torch.zeros(640, BITS))
        self.inp = nn.Linear(BITS, d, bias=False)
        self.out = nn.Linear(d, BITS, bias=False)
        if arm == "C":
            for mod in list(self.body.layers.modules()):
                for name, child in list(mod.named_children()):
                    if type(child) is nn.Linear:
                        t = TLinear(child.in_features, child.out_features, bias=child.bias is not None)
                        t.weight, t.bias = child.weight, child.bias
                        setattr(mod, name, t)

    def hidden_from_codes(self, codes):
        """codes: (B, T, 512) vectors in code space, one per position (table rows for ordinary tokens)."""
        return self.body(inputs_embeds=self.inp(codes), use_cache=False).last_hidden_state

    def hidden(self, ids):
        return self.hidden_from_codes(self.table[ids])

    def out_codes(self, h):
        """The model's output vector in code space: what it scores against every token's code."""
        return self.out(h)

    def logits_from_codes(self, c):
        return (c @ self.table.T) / math.sqrt(BITS) + self.seal


def load_checkpoint(arm, path, device="cuda"):
    m = FrozenVocabLM(arm)
    st = torch.load(path, map_location="cpu", weights_only=True)
    sd = st["model"]
    missing, unexpected = m.load_state_dict(sd, strict=False)
    missing = [k for k in missing if "rotary" not in k]
    if missing or unexpected:
        raise RuntimeError(f"checkpoint mismatch: missing {missing[:5]}, unexpected {unexpected[:5]}")
    return m.to(device).eval(), {k: st[k] for k in ("step", "arm", "tag") if k in st}
