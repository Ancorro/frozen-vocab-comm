"""Check the rebuilt models against the paper's validation loss (B 3.358, C 3.439 nats on 512 x 1,024 held-out tokens)."""
import sys
import numpy as np, torch, torch.nn.functional as F
from fvc.model import load_checkpoint
n = int(sys.argv[1]) if len(sys.argv) > 1 else 64
val = np.load("data/data/val.npy", mmap_mode="r")
x = torch.from_numpy(val[:n * 1025].astype(np.int64).reshape(n, 1025))
for arm in "BC":
    m, meta = load_checkpoint(arm, f"data/{arm}/step_07820.pt")
    tot = 0.0
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        for i in range(n):
            ids = x[i:i + 1].cuda()
            c = m.out_codes(m.hidden(ids[:, :-1]))[0]
            for j in range(0, 1024, 256):
                lg = m.logits_from_codes(c[j:j + 256]).float()
                tot += F.cross_entropy(lg, ids[0, 1 + j:1 + j + 256], reduction="sum").item()
    print(arm, meta, round(tot / (n * 1024), 4), flush=True)
    del m; torch.cuda.empty_cache()
