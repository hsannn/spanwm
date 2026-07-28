"""SpanWM v7 detection + metrics (multi-span pooled test).

Same metrics as v5 (AUROC / mean z / mean exact-p / TPR@FPR); reconstruction
scans up to max_spans anchors and pools all window tokens into one exact
binomial test. Detection needs only the tokenizer (model=None).

Run:
    python spanwm_detect_v7.py --input outputs/spanwm_v7_c4_n200.jsonl
    python spanwm_detect_v7.py --input outputs/spanwm_v7_c4_n200.jsonl --negative natural
"""

import argparse
import json

import numpy as np
import torch
from sklearn.metrics import roc_auc_score, roc_curve
from transformers import AutoTokenizer

from watermark.spanwm_v7 import SpanWMV7
from utils.transformers_config import TransformersConfig

MODEL_ID = "meta-llama/Llama-3.2-3B"
FLOOR_SCORE = -1.0


def p_to_score(p):
    if p is None:
        return FLOOR_SCORE
    return -np.log10(max(p, 1e-300))


def tpr_at_fpr(labels, scores, target_fpr):
    fpr, tpr, _ = roc_curve(labels, scores)
    return float(np.interp(target_fpr, fpr, tpr))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--column", default="watermarked_text")
    ap.add_argument("--model", default=MODEL_ID)
    ap.add_argument("--config", default="config/SpanWM_v7.json")
    ap.add_argument("--negative", default="unwatermarked", choices=["unwatermarked", "natural"])
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    transformers_config = TransformersConfig(
        model=None, tokenizer=tokenizer, vocab_size=len(tokenizer), device=device)
    watermark = SpanWMV7(args.config, transformers_config)

    neg_key = f"{args.negative}_text"
    with open(args.input) as f:
        records = [json.loads(line) for line in f if line.strip()]

    def detect(text):
        if not text:
            return None, None, 0
        r = watermark.detect_watermark(text)
        return r["score"], r["p_value"], r["num_sites"]

    pos_z, pos_p, neg_z, neg_p = [], [], [], []
    pos_sites, neg_sites = [], []
    n_neg = 0
    for r in records:
        z, p, s = detect(r[args.column]); pos_z.append(z); pos_p.append(p); pos_sites.append(s)
        neg_text = r.get(neg_key, "")
        if neg_text:
            n_neg += 1
        z, p, s = detect(neg_text); neg_z.append(z); neg_p.append(p); neg_sites.append(s)

    pos_ok_z = [z for z in pos_z if z is not None]
    pos_ok_p = [p for p in pos_p if p is not None]
    neg_ok_z = [z for z in neg_z if z is not None]
    neg_ok_p = [p for p in neg_p if p is not None]
    n = len(records)

    pos_scores = [p_to_score(p) for p in pos_p]
    neg_scores = [p_to_score(p) for p, r in zip(neg_p, records) if r.get(neg_key, "")]
    labels = [1] * len(pos_scores) + [0] * len(neg_scores)
    scores = pos_scores + neg_scores

    print("=" * 66)
    print(f"input           : {args.input}")
    print(f"samples         : {n}   negative class: {args.negative}   [v7: multi-span pooled]")
    print(f"reconstruction  : pos {len(pos_ok_z)}/{len(pos_z)}   neg {len(neg_ok_z)}/{n_neg}")
    print(f"mean sites      : pos {np.mean(pos_sites):.2f}   neg {np.mean([s for s, r in zip(neg_sites, records) if r.get(neg_key, '')]):.2f}")
    print("-" * 66)
    if pos_ok_z:
        print(f"mean z (pos)    : {np.mean(pos_ok_z):+.4f}   mean p (pos): {np.mean(pos_ok_p):.3e}")
    if neg_ok_z:
        print(f"mean z (neg)    : {np.mean(neg_ok_z):+.4f}   mean p (neg): {np.mean(neg_ok_p):.3e}")
    print("-" * 66)
    if len(set(labels)) < 2:
        print("need both classes for AUROC / TPR@FPR"); return
    print(f"AUROC           : {roc_auc_score(labels, scores):.4f}")
    for target in (0.10, 0.05, 0.01, 0.001):
        print(f"TPR@FPR={target * 100:>5.1f}%  : {tpr_at_fpr(labels, scores, target):.4f}")
    print("=" * 66)


if __name__ == "__main__":
    main()
