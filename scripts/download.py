"""Download the paper's released files into data/ (not committed; QLNI Non-Commercial Research Licence, Gemma terms).

    python scripts/download.py            # final B and C checkpoints, validation tokens
"""
import argparse
import pathlib

from huggingface_hub import hf_hub_download

REPO = "QLNI/embeddings-are-overpriced"
FILES = ["B/step_07820.pt", "C/step_07820.pt", "data/val.npy", "LICENSE.md"]

ap = argparse.ArgumentParser()
ap.add_argument("--out", default="data")
ap.add_argument("files", nargs="*", default=FILES)
a = ap.parse_args()
out = pathlib.Path(a.out)
out.mkdir(exist_ok=True)
for f in a.files:
    print(hf_hub_download(REPO, f, local_dir=out))
