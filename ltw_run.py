"""LTW (Learning to Watermark, NeurIPS 2025) reproduction driver.

Generation keeps the paper's own decoding protocol (their custom loop:
top_k=100, top_p=0.95, implicit temperature 1.0, no_repeat_ngram 8,
min/max_new_tokens 175/200) and the paper's watermark strength gamma=0.25 /
delta=3.0, with the released selector checkpoint (trained on OPT-1.3B; the
paper itself reuses it across OPT-6.7B / GPT-J, we extend to Llama-3.2-3B
and report the transfer honestly). Variants: ltw1 = KGW-style context hash
green list, ltw0 = unigram (fixed) green list -- the paper's more robust arm.

Detection follows the paper: the selector replays its gating decision over
the text (entropy from the BASE LM -> detection is NOT model-free), and the
z test runs over selected tokens only. We add the exact binomial p over
(green, scored) for our -log10(p) ranking protocol, identical to every other
row in the results table.

Pitfall 18 applies (CUDA randperm differs across GPU archs): generation and
detection MUST run on the same GPU family (base_suma_rtx3090).

    python ltw_run.py --model <path> --variant ltw1
    python ltw_run.py --model <path> --variant ltw0 --skip_generate
"""

import argparse
import json
import os

import numpy as np
import torch
from scipy.stats import binom
from sklearn.metrics import roc_curve

from transformers import AutoModelForCausalLM, AutoTokenizer

from evaluation.dataset import C4Dataset
from watermark.ltw.watermark import Detector, Watermark

FLOOR_SCORE = -1.0
DEFAULT_CONFIG = "config/LTW.json"


def p_to_score(p):
    if p is None:
        return FLOOR_SCORE
    return -np.log10(max(p, 1e-300))


def tpr_at_fpr(labels, scores, target_fpr):
    fpr, tpr, _ = roc_curve(labels, scores)
    return float(np.interp(target_fpr, fpr, tpr))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--config", default=DEFAULT_CONFIG)
    # CLI values override the config file; leaving them unset uses the config.
    ap.add_argument("--variant", choices=["ltw1", "ltw0"], default=None)
    ap.add_argument("--num_samples", type=int, default=200)
    ap.add_argument("--gamma", type=float, default=None)
    ap.add_argument("--delta", type=float, default=None)
    ap.add_argument("--max_new_tokens", type=int, default=None)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--output", default=None)
    ap.add_argument("--skip_generate", action="store_true")
    ap.add_argument("--force", action="store_true",
                    help="overwrite an existing output file")
    args = ap.parse_args()

    with open(args.config) as f:
        cfg = json.load(f)
    variant = args.variant or cfg["variant"]
    gamma = args.gamma if args.gamma is not None else float(cfg["gamma"])
    delta = args.delta if args.delta is not None else float(cfg["delta"])
    max_new = args.max_new_tokens or int(cfg["max_new_tokens"])
    k_ctx = int(cfg["k"])          # selector context window: generation == detection
    semantic_model = cfg["semantic_model"]
    checkpoint = cfg["checkpoint"]

    device = "cuda" if torch.cuda.is_available() else "cpu"
    tag = cfg.get("output_tag") if args.variant is None else variant
    out = args.output or f"outputs/{tag or variant}_c4_n{args.num_samples}.jsonl"
    unigram = variant == "ltw0"
    torch.manual_seed(args.seed)

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype="auto").to(device)
    model.eval()
    vocab_size = model.config.vocab_size
    print(f"variant={variant} unigram={unigram} gamma={gamma} delta={delta} "
          f"k={k_ctx} vocab={vocab_size} device={device} config={args.config}",
          flush=True)

    ds = C4Dataset("dataset/c4/processed_c4.json", max_samples=args.num_samples)
    n = min(args.num_samples, ds.prompt_nums)

    if not args.skip_generate:
        import os
        if os.path.exists(out) and not args.force:
            raise SystemExit(f"refusing to overwrite {out} — pass --output or --force")
        wm = Watermark(device=torch.device(device), model=model, tokenizer=tokenizer,
                       semantic_model_path=semantic_model, checkpoint_path=checkpoint,
                       top_k=int(cfg["top_k"]), top_p=float(cfg["top_p"]),
                       repetition_penalty=cfg["repetition_penalty"],
                       no_repeat_ngram_size=int(cfg["no_repeat_ngram_size"]),
                       max_new_tokens=max_new,
                       min_new_tokens=int(cfg["min_new_tokens"]),
                       k=k_ctx, embed_unigram_wm=unigram)
        with open(out, "w") as f:
            for i in range(n):
                prompt = ds.get_prompt(i)
                natural = ds.get_natural_text(i) if ds.natural_text_nums > i else ""
                wm_gen = wm.generate_watermark(prompt, gamma, delta)[0]
                un_gen = wm.generate_unwatermark(prompt)[0]
                rec = {"index": i, "prompt": prompt,
                       "watermarked_text": prompt + wm_gen,
                       "unwatermarked_text": prompt + un_gen,
                       "natural_text": natural}
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                f.flush()
                print(f"[{i + 1:>4}/{n}] wm_tok="
                      f"{len(tokenizer(wm_gen, add_special_tokens=False)['input_ids'])} "
                      f"un_tok={len(tokenizer(un_gen, add_special_tokens=False)['input_ids'])}",
                      flush=True)
        print(f"wrote {n} records -> {out}", flush=True)

    # ---------------- detection ----------------
    det = Detector(vocab=list(range(vocab_size)), gamma=gamma, delta=delta,
                   seeding_scheme="simple_1", tokenizer=tokenizer, model=model,
                   checkpoint_path=checkpoint, semantic_model_path=semantic_model,
                   z_threshold=float(cfg["z_threshold"]),
                   k=k_ctx,                       # same window as generation
                   embed_unigram_wm=unigram)

    recs = [json.loads(l) for l in open(out)]

    def score_text(text, prompt=None):
        ids = tokenizer(text, return_tensors="pt")["input_ids"][0].to(device)
        if prompt is not None:
            pref = tokenizer(prompt, return_tensors="pt")["input_ids"][0]
        else:
            pref = ids[:1]
        if len(ids) - max(1, len(pref)) < 5:
            return None
        with torch.no_grad():
            r = det.detect(tokenized_text=ids, tokenized_prefix=pref,
                           return_prediction=False)
        if r.get("invalid"):
            return None
        n_sc = int(r.get("num_tokens_scored", 0))
        g = int(r.get("num_green_tokens", 0))
        if n_sc < 1:
            return {"z": -100.0, "p": 1.0, "n": 0, "green": 0, "green_frac": 0.0,
                    "wm_frac": 0.0}
        return {"z": float(r["z_score"]), "p": float(binom.sf(g - 1, n_sc, gamma)),
                "n": n_sc, "green": g, "green_frac": g / n_sc,
                "wm_frac": float(r.get("watermarking_fraction", 0.0))}

    cls = {"watermarked": [], "unwatermarked": [], "natural": []}
    for i, r in enumerate(recs):
        cls["watermarked"].append(score_text(r["watermarked_text"], r["prompt"]))
        cls["unwatermarked"].append(score_text(r["unwatermarked_text"], r["prompt"]))
        nat = r.get("natural_text", "")
        cls["natural"].append(score_text(nat) if nat and len(nat) > 50 else None)
        if (i + 1) % 25 == 0:
            print(f"  detect {i + 1}/{len(recs)}", flush=True)

    pos = [s for s in cls["watermarked"] if s]
    for neg_name in ("unwatermarked", "natural"):
        neg = [s for s in cls[neg_name] if s]
        if not neg:
            continue
        y = [1] * len(pos) + [0] * len(neg)
        sc = [p_to_score(s["p"]) for s in pos] + [p_to_score(s["p"]) for s in neg]
        fpr, tpr, _ = roc_curve(y, sc)
        auroc = float(np.trapezoid(tpr, fpr)) if hasattr(np, "trapezoid") else float(np.trapz(tpr, fpr))
        print(f"\nsamples         : {len(pos)}   negative class: {neg_name}   variant: {variant}")
        print(f"tokens scored   : pos mean {np.mean([s['n'] for s in pos]):.1f}   "
              f"green_frac pos {np.mean([s['green_frac'] for s in pos]):.3f} "
              f"neg {np.mean([s['green_frac'] for s in neg]):.3f}   "
              f"wm_frac pos {np.mean([s['wm_frac'] for s in pos]):.3f}")
        print(f"mean z (pos)    : {np.mean([s['z'] for s in pos]):+.4f}   "
              f"mean p (pos): {np.mean([s['p'] for s in pos]):.3e}")
        print(f"mean z (neg)    : {np.mean([s['z'] for s in neg]):+.4f}")
        print(f"AUROC           : {auroc:.4f}")
        for t in (0.10, 0.05, 0.01, 0.001):
            print(f"TPR@FPR={t * 100:5.1f}%  : {tpr_at_fpr(y, sc, t):.4f}")
    print("\nLTW RUN DONE", flush=True)


if __name__ == "__main__":
    main()
