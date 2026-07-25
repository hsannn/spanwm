import argparse
import json

import numpy as np
import torch
from scipy.stats import norm
from sklearn.metrics import roc_auc_score, roc_curve
from transformers import AutoModelForCausalLM, AutoTokenizer

from watermark.auto_watermark import AutoWatermark
from utils.transformers_config import TransformersConfig

MODEL_ID = "meta-llama/Llama-3.2-3B"
# ROC score = z-score: higher = more watermarked. Text too short / no
# high-entropy tokens to score (ValueError from SWEETUtils.score_sequence)
# gets the lowest possible score.
FLOOR_SCORE = -1e9


def tpr_at_fpr(labels, scores, target_fpr: float) -> float:
    fpr, tpr, _ = roc_curve(labels, scores)
    return float(np.interp(target_fpr, fpr, tpr))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="jsonl from sweet_emb.py")
    ap.add_argument("--model", default=MODEL_ID)
    ap.add_argument("--config", default="config/SWEET.json")
    ap.add_argument("--negative", default="unwatermarked",
                    choices=["unwatermarked", "natural"],
                    help="which column is the non-watermarked (negative) class")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    # SWEET detection needs the model for per-token entropy, unlike SpanWM's
    # KGW greenlist-only detect -- must load real weights here.
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype="auto", device_map=device)
    model.eval()
    transformers_config = TransformersConfig(
        model=model, tokenizer=tokenizer, vocab_size=model.config.vocab_size, device=device,
    )
    watermark = AutoWatermark.load(
        "SWEET", algorithm_config=args.config, transformers_config=transformers_config)

    neg_key = f"{args.negative}_text"
    with open(args.input) as f:
        records = [json.loads(line) for line in f if line.strip()]

    def detect(text):
        if not text:
            return None, None
        try:
            r = watermark.detect_watermark(text)
        except ValueError:
            # not enough high-entropy tokens to score
            return None, None
        z = r["score"]
        p = norm.sf(z)  # normal-approx p-value (SWEET has no exact test)
        return z, p

    pos_z, pos_p = zip(*(detect(r["watermarked_text"]) for r in records)) if records else ((), ())
    neg_records = [r for r in records if r.get(neg_key, "")]
    neg_z, neg_p = zip(*(detect(r[neg_key]) for r in neg_records)) if neg_records else ((), ())

    pos_ok = [z for z in pos_z if z is not None]
    pos_ok_p = [p for p in pos_p if p is not None]
    neg_ok = [z for z in neg_z if z is not None]
    neg_ok_p = [p for p in neg_p if p is not None]
    n = len(records)

    pos_scores = [z if z is not None else FLOOR_SCORE for z in pos_z]
    neg_scores = [z if z is not None else FLOOR_SCORE for z in neg_z]
    labels = [1] * len(pos_scores) + [0] * len(neg_scores)
    scores = pos_scores + neg_scores

    print("=" * 66)
    print(f"input           : {args.input}")
    print(f"samples         : {n}   negative class: {args.negative}")
    print(f"scored          : pos {len(pos_ok)}/{len(pos_z)}   neg {len(neg_ok)}/{len(neg_z)}")
    print("-" * 66)
    if pos_ok:
        print(f"mean z (pos)    : {np.mean(pos_ok):+.4f}   mean p (pos): {np.mean(pos_ok_p):.3e}")
    if neg_ok:
        print(f"mean z (neg)    : {np.mean(neg_ok):+.4f}   mean p (neg): {np.mean(neg_ok_p):.3e}")
    print("-" * 66)

    if len(set(labels)) < 2:
        print("need both classes for AUROC / TPR@FPR")
        return

    auroc = roc_auc_score(labels, scores)
    print(f"AUROC           : {auroc:.4f}")
    for target in (0.10, 0.05, 0.01, 0.001):
        print(f"TPR@FPR={target*100:>5.1f}%  : {tpr_at_fpr(labels, scores, target):.4f}")
    print("=" * 66)


if __name__ == "__main__":
    main()
