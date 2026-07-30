"""Build the held-out CNN/DailyMail split that IE distillation trains on.

The evaluation corpus (dataset/cnn_dailymail/processed_cnn_dailymail.json) is
exactly the 200 rows we report on -- there is nothing after row 200 for the
tagger to learn from, which is why distillation died on an empty feature list.
IE's tagger must never see the text it will later be scored on, so the training
rows come from the dataset's TRAIN split (disjoint from test by construction),
with the 200 evaluation article ids excluded as a belt-and-braces check.

    python tools/build_cnndm_heldout.py [n_docs]
"""

import json
import os
import sys

from datasets import load_dataset

EVAL = "dataset/cnn_dailymail/processed_cnn_dailymail.json"
OUT = "dataset/cnn_dailymail/heldout_cnn_dailymail.json"


def main() -> None:
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 600
    with open(EVAL) as f:
        rows = [json.loads(line) for line in f if line.strip()]
    eval_ids = {r["id"] for r in rows if "id" in r}
    keys = list(rows[0])
    print(f"eval rows {len(rows)}, ids {len(eval_ids)}, schema {keys}")

    ds = load_dataset("abisee/cnn_dailymail", "3.0.0", split="train")
    out, seen = [], 0
    for r in ds:
        if r["id"] in eval_ids:
            seen += 1
            continue
        out.append({k: r[k] for k in keys if k in r})
        if len(out) >= n:
            break
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w") as f:
        for r in out:
            f.write(json.dumps(r) + "\n")
    print(f"wrote {len(out)} held-out docs -> {OUT} "
          f"(excluded {seen} that collided with the eval ids)")


if __name__ == "__main__":
    main()
