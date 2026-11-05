"""Goal check: a trained reader that lets model C read soft code-space messages, and carries over to a new sender.

    python scripts/check_goal.py

Reads results/reader/summary.json, written by a GPU run. Passes when:
  - the reader was trained on messages from sender B only, receiver C;
  - on held-out validation sequences, C's loss after the span with B's soft messages is below its loss with B's top
    tokens (the trained sender);
  - the same holds for messages from C itself, a sender the reader never saw in training;
  - the test used the settings of the untrained run in results/channel.json.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

SUMMARY = Path(__file__).resolve().parents[1] / "results" / "reader" / "summary.json"
SETTINGS = {"prefix": 128, "span": 32, "post": 64, "n": 256, "seed": 0}


def check(summary: Path = SUMMARY) -> list[str]:
    """Reasons the goal is not met; empty when it is."""
    if not summary.exists():
        return [f"no reader summary at {summary}"]
    s = json.loads(summary.read_text(encoding="utf-8"))
    errs = []
    hp = s.get("hyperparameters", {})
    errs += [f"setting {k} is {hp.get(k)!r}, need {v!r}" for k, v in SETTINGS.items() if hp.get(k) != v]
    if hp.get("train_senders") != ["B"]:
        errs.append(f"reader trained on senders {hp.get('train_senders')!r}, need ['B']")
    if hp.get("receiver") != "C":
        errs.append(f"receiver is {hp.get('receiver')!r}, need 'C'")
    for pair in ("B>C", "C>C"):
        r = s.get("results", {}).get(pair)
        if r is None or "soft" not in r or "argmax" not in r:
            errs.append(f"no soft and argmax loss for {pair}")
        elif not r["soft"] < r["argmax"]:
            errs.append(f"{pair}: soft {r['soft']:.3f} is not below argmax {r['argmax']:.3f}")
    return errs


def main() -> int:
    errs = check()
    print("PASS" if not errs else "FAIL")
    for e in errs:
        print(f"  - {e}")
    return int(bool(errs))


if __name__ == "__main__":
    sys.exit(main())
