"""KGW / SynthID baseline detection + metrics, matched to spanwm_detect_v6.py.

Reads the jsonl written by baseline_embed.py and reports the same metric block
as spanwm_detect_v6.py: mean z / mean exact-p over each class, AUROC and
TPR@FPR={10,5,1,0.1}% with positives ranked by -log10(p) (sklearn roc_curve +
interp, identical code path).

Protocol notes:
- The prompt is stripped from watermarked/unwatermarked texts before scoring
  (MarkLLM detectability protocol; the prompt itself is never watermarked).
  Default is exact char-level stripping text[len(prompt):] to preserve the
  generated string byte-for-byte (falls back to MarkLLM's word-level
  TruncatePromptTextEditor rule if the text does not start with the prompt).
  natural_text never contains the prompt and is scored as-is.
- KGW: z from KGWUtils.score_sequence (unchanged), plus the exact binomial
  p = P(Binom(N, gamma) >= G) — the same test SpanWM uses (binom.sf(G-1,N,g)),
  so the z / p-value columns are directly comparable across rows.
- SynthID: mean-detector score s = mean g-value over unmasked positions.
  z/p analog under H0 (each unmasked g ~ Bernoulli(0.5), K = num_unmasked *
  depth values): z = (s - 0.5) * 2 * sqrt(K), p = norm.sf(z). The raw-mean
  ranking AUROC is printed as a secondary line.

Detection needs only the tokenizer (model=None); GPU strongly recommended for
KGW (per-token randperm over the vocab).

Run:
    python baseline_detect.py --input outputs/kgw_c4_n200.jsonl --algorithm KGW
    python baseline_detect.py --input outputs/synthid_c4_n200.jsonl --algorithm SynthID --negative natural
"""

import argparse
import json
import os

import numpy as np
import torch
from scipy.stats import binom, norm
from sklearn.metrics import roc_auc_score, roc_curve
from transformers import AutoTokenizer

from watermark.auto_watermark import AutoWatermark
from utils.transformers_config import TransformersConfig

MODEL_ID = "meta-llama/Llama-3.2-3B"
FLOOR_SCORE = -1.0
DEFAULT_CONFIGS = {
    "KGW": "config/KGW_g0.25_d4.0.json",
    "SynthID": "config/SynthID.json",
    "SpARKP": "config/SpARKP_verb.json",
    "SpARKR": "config/SpARKR.json",
    "LemmaWM": "config/LemmaWM.json",
    "LemmaWMS": "config/LemmaWMS_k2.json",
    "ClusterWM": "config/ClusterWM_k2.json",
    "SentClusterWM": "config/SentClusterWM.json",
    "SWEET": "config/SWEET_g0.25_d4.0_t0.9.json",
    "EWD": "config/EWD_g0.25_d4.0.json",
    "Adaptive": "config/Adaptive.json",
    "IE": "config/IE_t2.2.json",
    "PivotWM": "config/PivotWM.json",
    "SpanCode": "config/SpanCode.json",
}


def p_to_score(p):
    if p is None:
        return FLOOR_SCORE
    return -np.log10(max(p, 1e-300))


def tpr_at_fpr(labels, scores, target_fpr):
    fpr, tpr, _ = roc_curve(labels, scores)
    return float(np.interp(target_fpr, fpr, tpr))


def strip_prompt(text, prompt, mode):
    """Remove the prompt prefix from a generated text."""
    if mode == "none" or not text:
        return text
    if mode == "char" and text.startswith(prompt):
        return text[len(prompt):]
    # MarkLLM TruncatePromptTextEditor rule (word-level) as fallback / "word"
    return " ".join(text.split()[len(prompt.split()):])


class KGWScorer:
    """z from KGWUtils.score_sequence + SpanWM-style exact binomial p."""

    def __init__(self, watermark, tokenizer, device):
        self.wm = watermark
        self.tokenizer = tokenizer
        self.device = device
        self.gamma = watermark.config.gamma
        self.prefix_length = watermark.config.prefix_length

    def score(self, text):
        """Returns dict(z, p, n, extra) or None if the text is unscorable."""
        ids = self.tokenizer(text, add_special_tokens=False)["input_ids"]
        if len(ids) - self.prefix_length < 1:
            return None
        encoded = torch.tensor(ids, device=self.device)
        z, flags = self.wm.utils.score_sequence(encoded)
        n = len(ids) - self.prefix_length
        g = sum(1 for f in flags if f == 1)
        p = float(binom.sf(g - 1, n, self.gamma))
        return {"z": float(z), "p": p, "n": n, "green": g,
                "green_frac": g / n,
                "decision": bool(z > self.wm.config.z_threshold)}


class SynthIDScorer:
    """Mean-detector score + z/p analog under the Bernoulli(0.5) null."""

    def __init__(self, watermark, tokenizer, device):
        self.wm = watermark
        self.tokenizer = tokenizer
        self.device = device
        self.cfg = watermark.config
        if self.cfg.detector_name not in ("mean", "weighted_mean"):
            raise ValueError("baseline_detect supports mean/weighted_mean detectors only")

    def score(self, text):
        enc = self.tokenizer(text, return_tensors="pt", add_special_tokens=False)["input_ids"].to(self.device)
        if enc.shape[1] < self.cfg.ngram_len:
            return None
        proc = self.wm.logits_processor
        g_values = proc.compute_g_values(enc)
        eos_mask = proc.compute_eos_token_mask(
            input_ids=enc, eos_token_id=self.tokenizer.eos_token_id
        )[:, self.cfg.ngram_len - 1:]
        if self.cfg.watermark_mode == "non-distortionary":
            rep_mask = proc.compute_context_repetition_mask(input_ids=enc)
            mask = rep_mask * eos_mask
        else:
            mask = eos_mask
        g_np = g_values.cpu().numpy()
        m_np = mask.cpu().numpy()
        num_unmasked = int(m_np.sum())
        if num_unmasked < 1:
            return None
        s = float(self.wm.detector.detect(g_np, m_np)[0])
        k = num_unmasked * g_values.shape[-1]
        z = (s - 0.5) * 2.0 * np.sqrt(k)
        p = float(norm.sf(z))
        return {"z": float(z), "p": p, "n": num_unmasked, "mean_g": s,
                "decision": bool(s > self.cfg.threshold)}


class DetectorScorer:
    """Methods whose own detect_watermark returns z / p / counts (LemmaWM)."""

    def __init__(self, watermark, tokenizer, device):
        self.wm = watermark

    def score(self, text):
        r = self.wm.detect_watermark(text)
        n = r.get("num_tested_tokens", 0)
        if not n:
            return None
        return {"z": float(r["score"]), "p": float(r["p_value"]), "n": n,
                "green": r["num_green_tokens"], "green_frac": r["num_green_tokens"] / n,
                "decision": bool(r["is_watermarked"])}


class SweetScorer:
    """SWEET: green test restricted to high-entropy tokens (entropy from the
    generation model). z from utils.score_sequence + exact binomial p."""

    def __init__(self, watermark, tokenizer, device):
        self.wm = watermark
        self.tokenizer = tokenizer
        self.device = device
        self.gamma = watermark.config.gamma
        self.prefix = watermark.config.prefix_length

    def score(self, text):
        ids = self.tokenizer(text, return_tensors="pt",
                             add_special_tokens=False)["input_ids"][0].to(self.device)
        if len(ids) - self.prefix < 1:
            return None
        ent = self.wm.utils.calculate_entropy(self.wm.config.generation_model, ids)
        z, flags, weights = self.wm.utils.score_sequence(ids, ent)
        # the entropy gate lives in `weights`; `flags` covers every token
        n = sum(1 for f, w in zip(flags, weights) if w == 1 and f in (0, 1))
        g = sum(1 for f, w in zip(flags, weights) if w == 1 and f == 1)
        if n < 1:
            return None
        p = float(binom.sf(g - 1, n, self.gamma))
        return {"z": float(z), "p": p, "n": n, "green": g, "green_frac": g / n,
                "decision": bool(z > self.wm.config.z_threshold)}


class EwdScorer:
    """EWD: entropy-WEIGHTED green statistic (detection-only novelty; the
    generation is standard KGW-style bias). p = norm.sf(z) for the weighted z."""

    def __init__(self, watermark, tokenizer, device):
        self.wm = watermark
        self.tokenizer = tokenizer
        self.device = device

    def score(self, text):
        ids = self.tokenizer(text, return_tensors="pt",
                             add_special_tokens=False)["input_ids"][0].to(self.device)
        if len(ids) - self.wm.config.prefix_length < 1:
            return None
        ent = self.wm.utils.calculate_entropy(self.wm.config.generation_model, ids)
        z, flags, _ = self.wm.utils.score_sequence(ids, ent)
        scored = [f for f in flags if f in (0, 1)]
        n, g = len(scored), sum(scored)
        p = float(norm.sf(float(z)))
        return {"z": float(z), "p": p, "n": n, "green": g,
                "green_frac": (g / n) if n else 0.0,
                "decision": bool(z > self.wm.config.z_threshold)}


class IEScorer:
    """IE: KGW green test gated to tokens the entropy tagger predicts as
    HIGH-entropy (P(low)<0.5). Detection needs NO base LM -- the gate comes
    from SimCSE features + the (retrained, in-domain) tagger MLP. z from
    utils.score_sequence + exact binomial p over gated tokens, mirroring
    SweetScorer (gate lives in `weights`; `flags` covers every token)."""

    def __init__(self, watermark, tokenizer, device):
        self.wm = watermark
        self.tokenizer = tokenizer
        self.gamma = watermark.config.gamma

    def score(self, text):
        ids = self.tokenizer(text, return_tensors="pt",
                             add_special_tokens=False)["input_ids"][0].to(self.wm.config.device)
        if len(ids) < 2:
            return None
        ent = self.wm.utils.calculate_entropy(
            None, self.wm.feature_extractor, self.wm.entropy_tagger, ids)
        try:
            z, flags, weights = self.wm.utils.score_sequence(ids, ent)
        except ValueError:
            return None
        n = sum(1 for f, w in zip(flags, weights) if w == 1 and f in (0, 1))
        g = sum(1 for f, w in zip(flags, weights) if w == 1 and f == 1)
        if n < 1:
            return None
        p = float(binom.sf(g - 1, n, self.gamma))
        return {"z": float(z), "p": p, "n": n, "green": g, "green_frac": g / n,
                "decision": bool(z > self.wm.config.z_threshold)}


class AdaptiveScorer:
    """Adaptive (ATW): detect_watermark returns a normalized score in [0,1]
    (fraction of measured tokens whose logits-scaling entry is positive).
    There is no analytic null, so the score is used directly for ranking and
    a Bernoulli(0.5) normal approximation gives a comparable z."""

    def __init__(self, watermark, tokenizer, device):
        self.wm = watermark
        self.tokenizer = tokenizer

    def score(self, text):
        r = self.wm.detect_watermark(text)
        s = float(r.get("score", 0.0))
        n = len(self.tokenizer(text, add_special_tokens=False)["input_ids"])
        if n < 1:
            return None
        z = (s - 0.5) * 2.0 * np.sqrt(n)
        return {"z": z, "p": float(norm.sf(z)), "n": n, "green": s * n,
                "green_frac": s,
                "decision": bool(r.get("is_watermarked", s > 0.5))}


class SpARKPScorer:
    """SpARK-P: z over reconstructed trigger positions + exact binomial p.
    A text with zero tested positions carries no evidence (z=0, p=1)."""

    def __init__(self, watermark, tokenizer, device):
        self.wm = watermark
        self.gamma = watermark.config.gamma

    def score(self, text):
        bits = self.wm.utils.decode_bits(text)
        t, g = len(bits), sum(bits)
        if t == 0:
            return {"z": 0.0, "p": 1.0, "n": 0, "green": 0, "green_frac": 0.0,
                    "decision": False}
        z = (g - self.gamma * t) / np.sqrt(t * self.gamma * (1 - self.gamma))
        p = float(binom.sf(g - 1, t, self.gamma))
        return {"z": float(z), "p": p, "n": t, "green": g, "green_frac": g / t,
                "decision": bool(z > self.wm.config.z_threshold)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--algorithm", required=True, choices=list(DEFAULT_CONFIGS))
    ap.add_argument("--model", default=MODEL_ID)
    ap.add_argument("--config", default=None)
    ap.add_argument("--negative", default="unwatermarked", choices=["unwatermarked", "natural"])
    ap.add_argument("--truncation", default="char", choices=["char", "word", "none"])
    ap.add_argument("--scores_out", default=None,
                    help="sidecar json with per-sample scores (default <input>.<neg>.scores.json)")
    args = ap.parse_args()

    config_path = args.config or DEFAULT_CONFIGS[args.algorithm]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    # SWEET/EWD compute per-token entropy at detection -> need the LM loaded
    if args.algorithm in ("SWEET", "EWD"):
        from transformers import AutoModelForCausalLM
        det_model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype="auto").to(device)
        det_model.eval()
    else:
        det_model = None
    transformers_config = TransformersConfig(
        model=det_model, tokenizer=tokenizer, vocab_size=len(tokenizer), device=device)
    if args.algorithm == "SynthID":
        transformers_config.temperature = 0.8
        transformers_config.top_k = -1
    watermark = AutoWatermark.load(args.algorithm, algorithm_config=config_path,
                                   transformers_config=transformers_config)

    scorer_cls = {"KGW": KGWScorer, "SynthID": SynthIDScorer,
                  "SpARKP": SpARKPScorer, "SpARKR": SpARKPScorer,
                  "LemmaWM": DetectorScorer, "LemmaWMS": DetectorScorer,
                  "ClusterWM": DetectorScorer, "SentClusterWM": DetectorScorer,
                  "SpanCode": DetectorScorer,
                  "PivotWM": DetectorScorer,
                  "Adaptive": AdaptiveScorer,
                  "IE": IEScorer,
                  "SWEET": SweetScorer, "EWD": EwdScorer}[args.algorithm]
    scorer = scorer_cls(watermark, tokenizer, device)

    with open(args.input) as f:
        records = [json.loads(line) for line in f if line.strip()]

    neg_key = f"{args.negative}_text"
    pos_res, neg_res = [], []
    n_neg = 0
    for r in records:
        pos_text = strip_prompt(r["watermarked_text"], r["prompt"], args.truncation)
        pos_res.append(scorer.score(pos_text) if pos_text else None)

        neg_text = r.get(neg_key, "")
        if args.negative == "unwatermarked":
            neg_text = strip_prompt(neg_text, r["prompt"], args.truncation)
        if neg_text:
            n_neg += 1
            neg_res.append(scorer.score(neg_text))
        else:
            neg_res.append(None)

    pos_ok = [x for x in pos_res if x is not None]
    neg_ok = [x for x in neg_res if x is not None]

    pos_scores = [p_to_score(x["p"] if x else None) for x in pos_res]
    neg_scores = [p_to_score(x["p"]) if x else FLOOR_SCORE
                  for x, r in zip(neg_res, records) if r.get(neg_key, "")]
    labels = [1] * len(pos_scores) + [0] * len(neg_scores)
    scores = pos_scores + neg_scores

    print("=" * 66)
    print(f"input           : {args.input}")
    print(f"algorithm       : {args.algorithm}   config: {config_path}")
    print(f"samples         : {len(records)}   negative class: {args.negative}   truncation: {args.truncation}")
    print(f"scorable        : pos {len(pos_ok)}/{len(pos_res)}   neg {len(neg_ok)}/{n_neg}")
    if args.algorithm in ("KGW", "SpARKP", "SpARKR", "LemmaWM", "LemmaWMS", "ClusterWM", "SentClusterWM", "SWEET", "EWD", "SpanCode", "Adaptive", "IE", "PivotWM"):
        print(f"tokens scored   : pos mean {np.mean([x['n'] for x in pos_ok]):.1f}   "
              f"green_frac pos {np.mean([x['green_frac'] for x in pos_ok]):.3f} "
              f"neg {np.mean([x['green_frac'] for x in neg_ok]):.3f}")
    else:
        print(f"g-vals unmasked : pos mean {np.mean([x['n'] for x in pos_ok]):.1f}   "
              f"mean_g pos {np.mean([x['mean_g'] for x in pos_ok]):.4f} "
              f"neg {np.mean([x['mean_g'] for x in neg_ok]):.4f}")
    print("-" * 66)
    if pos_ok:
        print(f"mean z (pos)    : {np.mean([x['z'] for x in pos_ok]):+.4f}   "
              f"mean p (pos): {np.mean([x['p'] for x in pos_ok]):.3e}")
    if neg_ok:
        print(f"mean z (neg)    : {np.mean([x['z'] for x in neg_ok]):+.4f}   "
              f"mean p (neg): {np.mean([x['p'] for x in neg_ok]):.3e}")
    # fixed-threshold classification at the method's native threshold
    # (KGW/SpARKP: z > z_threshold, SynthID: mean_g > threshold);
    # unscorable texts count as a negative decision.
    pos_dec = [bool(x and x["decision"]) for x in pos_res]
    neg_dec = [bool(x and x["decision"])
               for x, r in zip(neg_res, records) if r.get(neg_key, "")]
    if pos_dec and neg_dec:
        tpr = float(np.mean(pos_dec))
        fpr = float(np.mean(neg_dec))
        print(f"fixed threshold : TPR {tpr:.4f}   TNR {1 - fpr:.4f}   "
              f"FPR {fpr:.4f}   FNR {1 - tpr:.4f}")
    print("-" * 66)
    if len(set(labels)) < 2:
        print("need both classes for AUROC / TPR@FPR")
        return
    print(f"AUROC           : {roc_auc_score(labels, scores):.4f}")
    for target in (0.10, 0.05, 0.01, 0.001):
        print(f"TPR@FPR={target * 100:>5.1f}%  : {tpr_at_fpr(labels, scores, target):.4f}")
    if args.algorithm == "SynthID":
        raw = [x["mean_g"] if x else -1.0 for x in pos_res] + \
              [x["mean_g"] if x else -1.0 for x, r in zip(neg_res, records) if r.get(neg_key, "")]
        print(f"AUROC (raw mean): {roc_auc_score(labels, raw):.4f}   [secondary: unnormalized mean-g ranking]")
    else:
        raw = [x["z"] if x else -1e9 for x in pos_res] + \
              [x["z"] if x else -1e9 for x, r in zip(neg_res, records) if r.get(neg_key, "")]
        print(f"AUROC (rank by z): {roc_auc_score(labels, raw):.4f}   [secondary]")
    print("=" * 66)

    scores_out = args.scores_out or f"{os.path.splitext(args.input)[0]}.{args.negative}.scores.json"
    with open(scores_out, "w") as f:
        json.dump({
            "input": args.input, "algorithm": args.algorithm, "config": config_path,
            "negative": args.negative, "truncation": args.truncation,
            "pos": pos_res, "neg": neg_res,
        }, f, indent=1)
    print(f"per-sample scores -> {scores_out}")


if __name__ == "__main__":
    main()
