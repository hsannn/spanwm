"""Oracle-model perplexity via vLLM prompt_logprobs (fast batched scoring).

Texts are tokenized EXPLICITLY with the oracle tokenizer using
add_special_tokens=True and passed as token ids, so BOS is guaranteed —
gemma-family models produce garbage logprobs without <bos> (relying on vLLM's
implicit tokenization produced exactly that failure).

Reported per (file, which):
  corpus_ppl  = exp(total NLL / total scored tokens)   [token-weighted, primary]
  median_ppl  = median of per-text ppl
  gmean_ppl   = exp(mean log per-text ppl)
  mean_ppl    = arithmetic mean of per-text ppl        [outlier-dominated]

Run (vllm env):
    python baseline_ppl_vllm.py --oracle_model <path-or-id> \
        --input outputs/kgw_c4_n200.jsonl ...
"""

import argparse
import json
import os

import numpy as np
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams

def strip_prompt(text, prompt, mode):
    if mode == "none" or not text:
        return text
    if mode == "char" and text.startswith(prompt):
        return text[len(prompt):]
    return " ".join(text.split()[len(prompt.split()):])


def nll_from_output(out):
    """(total_nll, n_scored) of one RequestOutput's prompt."""
    lps = out.prompt_logprobs
    tids = out.prompt_token_ids
    if not lps or not tids:
        return None
    vals = []
    for tid, d in zip(tids, lps):
        if d is None:
            continue
        item = d.get(tid)
        if item is not None:
            vals.append(item.logprob)
    if len(vals) < 1:
        return None
    return -float(np.sum(vals)), len(vals)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", nargs="+", required=True)
    ap.add_argument("--oracle_model", required=True)
    ap.add_argument("--which", nargs="+", default=["watermarked", "unwatermarked", "natural"],
                    choices=["watermarked", "unwatermarked", "natural"])
    ap.add_argument("--truncation", default="char", choices=["char", "word", "none"])
    ap.add_argument("--out", default="outputs/ppl_summary.json")
    ap.add_argument("--tensor_parallel", type=int, default=1)
    ap.add_argument("--max_model_len", type=int, default=2048)
    ap.add_argument("--gpu_memory_utilization", type=float, default=0.90)
    args = ap.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.oracle_model)

    tasks = []  # (path, which, text)
    for path in args.input:
        with open(path) as f:
            records = [json.loads(line) for line in f if line.strip()]
        for which in args.which:
            for r in records:
                text = r.get(f"{which}_text", "")
                if which in ("watermarked", "unwatermarked"):
                    text = strip_prompt(text, r["prompt"], args.truncation)
                if text and len(text.split()) >= 2:
                    tasks.append((path, which, text))
    print(f"scoring {len(tasks)} texts with {args.oracle_model}", flush=True)

    # explicit tokenization; gemma-4's tokenizer does NOT prepend <bos> even
    # with add_special_tokens=True, and gemma is catastrophically miscalibrated
    # without it — force-prepend when missing.
    bos = tokenizer.bos_token_id
    prompts = []
    for _, _, text in tasks:
        ids = tokenizer(text, add_special_tokens=True)["input_ids"][:args.max_model_len - 8]
        if bos is not None and (not ids or ids[0] != bos):
            ids = [bos] + ids
        prompts.append({"prompt_token_ids": ids})
    print(f"first task ids[:5]={prompts[0]['prompt_token_ids'][:5]} "
          f"(bos_token_id={tokenizer.bos_token_id})", flush=True)

    llm = LLM(model=args.oracle_model, tokenizer=args.oracle_model,
              tensor_parallel_size=args.tensor_parallel,
              max_model_len=args.max_model_len,
              gpu_memory_utilization=args.gpu_memory_utilization,
              dtype="auto", seed=42)
    sp = SamplingParams(max_tokens=1, prompt_logprobs=0, temperature=0.0)
    outs = llm.generate(prompts, sp, use_tqdm=True)

    per_group = {}
    for (path, which, _), out in zip(tasks, outs):
        r = nll_from_output(out)
        if r is not None:
            per_group.setdefault(path, {}).setdefault(which, []).append(r)

    summary = {"oracle_model": args.oracle_model, "truncation": args.truncation,
               "bos_token_id": tokenizer.bos_token_id, "files": {}}
    for path, groups in per_group.items():
        summary["files"][path] = {}
        for which, rs in groups.items():
            total_nll = sum(nll for nll, _ in rs)
            total_tok = sum(n for _, n in rs)
            ppls = [np.exp(nll / n) for nll, n in rs]
            summary["files"][path][which] = {
                "corpus_ppl": float(np.exp(total_nll / total_tok)),
                "median_ppl": float(np.median(ppls)),
                "gmean_ppl": float(np.exp(np.mean(np.log(ppls)))),
                "mean_ppl": float(np.mean(ppls)),
                "n": len(rs), "tokens": int(total_tok),
            }
            s = summary["files"][path][which]
            print(f"{os.path.basename(path):34s} {which:14s} "
                  f"corpus={s['corpus_ppl']:9.3f}  median={s['median_ppl']:9.3f}  "
                  f"gmean={s['gmean_ppl']:9.3f}  mean={s['mean_ppl']:12.3f}  n={s['n']}", flush=True)

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(summary, f, indent=1)
    print(f"summary -> {args.out}")


if __name__ == "__main__":
    main()
