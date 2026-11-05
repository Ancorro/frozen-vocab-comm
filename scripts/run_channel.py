"""Run every message kind for sender -> receiver pairs on the paper's held-out validation tokens.

    python scripts/run_channel.py --n 256 --span 32 --post 64 --prefix 128
"""
import argparse
import json
import pathlib
import time

import numpy as np
import torch

from fvc.channel import MESSAGES, run_pair
from fvc.model import load_checkpoint

ap = argparse.ArgumentParser()
ap.add_argument("--data", default="data")
ap.add_argument("--n", type=int, default=256, help="sequences")
ap.add_argument("--prefix", type=int, default=128)
ap.add_argument("--span", type=int, default=32)
ap.add_argument("--post", type=int, default=64)
ap.add_argument("--batch", type=int, default=16)
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--pairs", default="B>C,C>B,B>B,C>C")
ap.add_argument("--out", default="results/channel.json")
a = ap.parse_args()

d = pathlib.Path(a.data)
models = {arm: load_checkpoint(arm, d / arm / "step_07820.pt")[0] for arm in "BC"}
for m in models.values():
    m.to(torch.bfloat16)
T = a.prefix + a.span + a.post + 1
val = np.load(d / "data" / "val.npy", mmap_mode="r")
starts = np.random.default_rng(a.seed).choice(len(val) // T, a.n, replace=False) * T
ids_all = torch.from_numpy(np.stack([val[s:s + T] for s in starts]).astype(np.int64))

res = {}
t0 = time.time()
for pair in a.pairs.split(","):
    xs, ys = pair.split(">")
    X, Y = models[xs], models[ys]
    acc = {k: [] for k in MESSAGES}
    for i in range(0, a.n, a.batch):
        ids = ids_all[i:i + a.batch].cuda()
        with torch.autocast("cuda", dtype=torch.bfloat16):
            r = run_pair(X, Y, ids, a.prefix, a.span, a.post, seed=a.seed + i)
        for k, v in r.items():
            acc[k] += v.float().cpu().tolist()
    res[pair] = acc
    print(pair, {k: round(float(np.mean(v)), 3) for k, v in acc.items()}, f"{time.time() - t0:.0f}s", flush=True)

out = pathlib.Path(a.out)
out.parent.mkdir(parents=True, exist_ok=True)
json.dump({"args": vars(a), "per_sequence_nll": res}, open(out, "w"))
