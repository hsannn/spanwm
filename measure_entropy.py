"""Per-(model, dataset) entropy calibration for the entropy-gated schemes.

SWEET and IE both gate on a threshold tau over the base model's next-token
Shannon entropy. The value we have been using (2.2) is Llama-3.2-3B's own
entropy statistic on C4 -- it carries no meaning for a different model or a
different domain, whose predictive distributions are differently peaked. A
fixed tau would therefore gate a different FRACTION of tokens in every cell,
making the schemes incomparable across the scale-up grid.

This measures the entropy distribution of one (model, dataset) pair and emits
the calibrated threshold, rounded to one decimal place:

    tau = round(mean entropy, 1)

Entropy is measured exactly as the schemes see it: the Shannon entropy of the
model's next-token distribution, teacher-forced over the dataset text the run
will actually condition on (the same first-200 rows the evaluation uses).
Median and the quartiles are reported alongside so the choice of statistic is
visible rather than implicit.

    python measure_entropy.py --model <path> --dataset c4 --out entropy_stats.json
"""

import argparse
import json
import os

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from utils.lm_compat import dtype_kwargs
from evaluation.dataset import (C4Dataset, CNN_DailyMailDataset,
                                WMT16DE_ENDataset)
# the continuation loaders live apart so evaluation/dataset.py (the
# collaborator's file) stays untouched
from evaluation.dataset_continuation import (CNNArticleDataset,
                                             WMT16ENDataset)

DATASETS = {
    "c4": (C4Dataset, "dataset/c4/processed_c4.json"),
    # continuation protocol (collaborator-style): first 30 words -> continue
    "cnn": (CNNArticleDataset,
            "dataset/cnn_dailymail/test-00000-of-00001.jsonl"),
    "cnn_dailymail": (CNN_DailyMailDataset,
                      "dataset/cnn_dailymail/processed_cnn_dailymail.json"),
    # collaborator protocol: en-side continuation, <10-word sentences skipped
    "wmt16": (WMT16ENDataset,
              "dataset/wmt16_de_en/processed_wmt16_de_en.json"),
    "wmt16_de_en": (WMT16DE_ENDataset,
                    "dataset/wmt16_de_en/processed_wmt16_de_en.json"),
}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--model_key", required=True)
    ap.add_argument("--dataset", required=True, choices=list(DATASETS))
    ap.add_argument("--num_samples", type=int, default=200)
    ap.add_argument("--max_tokens", type=int, default=256)
    ap.add_argument("--out", default="outputs/entropy_stats.json")
    args = ap.parse_args()

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(args.model)
    lm = AutoModelForCausalLM.from_pretrained(
        args.model, **dtype_kwargs("auto")).to(dev).eval()

    cls, path = DATASETS[args.dataset]
    ds = cls(path, max_samples=args.num_samples)
    n = min(args.num_samples, ds.prompt_nums)

    ents = []
    for i in range(n):
        # condition on exactly what the run conditions on: the prompt plus the
        # dataset's own continuation (C4 natural_text, else the reference)
        prompt = ds.get_prompt(i)
        if ds.natural_text_nums > i:
            cont = ds.get_natural_text(i)
        elif ds.reference_nums > i:
            cont = ds.get_reference(i)
        else:
            cont = ""
        text = (prompt + " " + cont).strip()
        ids = tok(text, return_tensors="pt", add_special_tokens=False,
                  truncation=True, max_length=args.max_tokens)["input_ids"].to(dev)
        if ids.shape[1] < 8:
            continue
        with torch.no_grad():
            probs = torch.softmax(lm(ids).logits.float(), dim=-1)
        e = -(probs * torch.log(probs.clamp_min(1e-12))).sum(-1)[0].cpu().numpy()
        ents.append(e)
        if (i + 1) % 50 == 0:
            print(f"  {i + 1}/{n} docs", flush=True)

    a = np.concatenate(ents)
    tau = round(float(a.mean()), 1)
    stats = {
        "model_key": args.model_key,
        "model_path": args.model,
        "dataset": args.dataset,
        "n_docs": len(ents),
        "n_positions": int(a.size),
        "mean": round(float(a.mean()), 4),
        "median": round(float(np.median(a)), 4),
        "p25": round(float(np.percentile(a, 25)), 4),
        "p75": round(float(np.percentile(a, 75)), 4),
        "std": round(float(a.std()), 4),
        # the calibrated gate: mean rounded to one decimal
        "tau": tau,
        "frac_above_tau": round(float((a >= tau).mean()), 4),
    }
    print(json.dumps(stats, indent=2), flush=True)

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    all_stats = {}
    if os.path.exists(args.out):
        with open(args.out) as f:
            all_stats = json.load(f)
    all_stats[f"{args.model_key}_{args.dataset}"] = stats
    with open(args.out, "w") as f:
        json.dump(all_stats, f, indent=2, sort_keys=True)
    print(f"tau({args.model_key}, {args.dataset}) = {tau} "
          f"(mean {stats['mean']}, median {stats['median']}, "
          f"gates {stats['frac_above_tau']:.1%} of tokens)", flush=True)
    print("ENTROPY MEASURE DONE")


if __name__ == "__main__":
    main()
