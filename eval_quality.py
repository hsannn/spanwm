"""
eval_quality.py — measure quality of watermarked text.

Two independent metrics on the `watermarked_text` field of an embed jsonl:
  * PPL       : perplexity under an LM (default google/gemma-4-12B).
  * GPT judge : style / consistency / ethics scores (1-10) from an OpenAI
                judge model (default gpt-5-mini-2025-08-07), prompted with
                gpt_prompt.txt.

The two metrics live in different environments (PPL wants a GPU + transformers;
the judge wants the `openai` package + network), so each is its own `--mode`
and results are merged (by sample index) into one output json. Run them
separately or together:

    # PPL  — spanwm env, on a GPU node (srun --overlap ...)
    python eval_quality.py --mode ppl

    # judge — seqwm env (has openai), login node (has network)
    python eval_quality.py --mode judge

    # both  — needs GPU + network + openai all in one env
    python eval_quality.py --mode both

Output (default outputs/quality_<inputstem>.json) is re-read and updated on
each run, so the ppl pass and the judge pass accumulate into the same file.
"""

import os
import re
import sys
import json
import time
import argparse

REPO = os.path.dirname(os.path.abspath(__file__))


# --------------------------------------------------------------------------- #
# io helpers
# --------------------------------------------------------------------------- #
def load_jsonl(path):
    rows = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def load_results(path):
    """Existing output keyed by index (str) -> per-sample dict."""
    if os.path.exists(path):
        with open(path) as f:
            data = json.load(f)
        return {str(r["index"]): r for r in data.get("per_sample", [])}
    return {}


def save_results(path, results, meta):
    per_sample = [results[k] for k in sorted(results, key=lambda x: int(x))]
    summary = summarize(per_sample)
    with open(path, "w") as f:
        json.dump({"meta": meta, "summary": summary, "per_sample": per_sample},
                  f, indent=2, ensure_ascii=False)
    return summary


def load_env_file(path):
    """Minimal .env loader: set KEY=VALUE lines into os.environ (no override)."""
    if not path or not os.path.exists(path):
        return
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            k, v = line.split("=", 1)
            k, v = k.strip(), v.strip().strip('"').strip("'")
            os.environ.setdefault(k, v)


# --------------------------------------------------------------------------- #
# PPL
# --------------------------------------------------------------------------- #
def load_ppl_model(model_name):
    import torch
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(model_name)
    model = None
    # gemma-4-12B is a Gemma4UnifiedForConditionalGeneration (multimodal); it is
    # not always registered under AutoModelForCausalLM, so fall back.
    last_err = None
    for loader_name in ("AutoModelForCausalLM", "AutoModelForMultimodalLM"):
        try:
            import transformers
            Loader = getattr(transformers, loader_name)
            model = Loader.from_pretrained(
                model_name, device_map="auto", dtype=torch.bfloat16)
            print(f"[ppl] loaded {model_name} via {loader_name}")
            break
        except Exception as e:  # noqa: BLE001
            last_err = e
            print(f"[ppl] {loader_name} failed: {e}")
    if model is None:
        raise RuntimeError(f"could not load {model_name}: {last_err}")
    model.eval()
    return tok, model


def compute_ppl(text, tok, model):
    """Perplexity, matching MarkLLM's PPLCalculator (add_special_tokens=False)."""
    import torch

    ids = tok(text, return_tensors="pt", add_special_tokens=False)["input_ids"]
    if ids.shape[1] < 2:
        return None
    device = next(model.parameters()).device
    ids = ids.to(device)
    with torch.inference_mode():
        logits = model(input_ids=ids).logits[0].float()
    loss = torch.nn.functional.cross_entropy(logits[:-1], ids[0][1:])
    return float(torch.exp(loss).item())


def run_ppl(rows, results, args):
    tok, model = load_ppl_model(args.ppl_model)
    n = len(rows)
    for i, row in enumerate(rows):
        idx = str(row["index"])
        text = (row.get(args.field) or "").strip()
        entry = results.setdefault(idx, {"index": row["index"]})
        if not text:
            entry["ppl"] = None
        else:
            entry["ppl"] = compute_ppl(text, tok, model)
        print(f"[ppl] {i + 1}/{n} idx={idx} ppl={entry['ppl']}")
        if (i + 1) % 20 == 0 or i + 1 == n:
            save_results(args.output, results, build_meta(args))
    return results


# --------------------------------------------------------------------------- #
# GPT judge
# --------------------------------------------------------------------------- #
def extract_scores(obj):
    """Pull style/consistency/ethics numeric scores from the judge's json."""
    scores = {}
    for k, v in obj.items():
        kl = k.lower()
        if not isinstance(v, dict) or "score" not in v:
            continue
        try:
            s = float(str(v["score"]).strip())
        except (ValueError, TypeError):
            continue
        if "style" in kl:
            scores["style"] = s
        elif "consist" in kl:
            scores["consistency"] = s
        elif "ethic" in kl:
            scores["ethics"] = s
    return scores


def parse_judge_response(content):
    """Return (scores_dict, parsed_json_or_None)."""
    obj = None
    try:
        obj = json.loads(content)
    except (json.JSONDecodeError, TypeError):
        m = re.search(r"\{.*\}", content or "", re.DOTALL)
        if m:
            try:
                obj = json.loads(m.group())
            except json.JSONDecodeError:
                obj = None
    if isinstance(obj, dict):
        return extract_scores(obj), obj
    return {}, None


def judge_text(client, model, system_prompt, text, max_completion_tokens,
               temperature, retries=5):
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": text},
    ]
    kwargs = dict(model=model, messages=messages,
                  max_completion_tokens=max_completion_tokens,
                  response_format={"type": "json_object"})
    if temperature is not None:
        kwargs["temperature"] = temperature
    last_err = None
    for attempt in range(retries):
        try:
            resp = client.chat.completions.create(**kwargs)
            return resp.choices[0].message.content
        except Exception as e:  # noqa: BLE001
            last_err = e
            msg = str(e)
            # gpt-5* reject a non-default temperature; drop it and retry.
            if "temperature" in msg and "temperature" in kwargs:
                kwargs.pop("temperature")
                continue
            # json response_format unsupported -> retry without it.
            if "response_format" in msg and "response_format" in kwargs:
                kwargs.pop("response_format")
                continue
            print(f"[judge] api error (attempt {attempt + 1}): {e}")
            time.sleep(min(10, 2 ** attempt))
    raise RuntimeError(f"judge failed after {retries} retries: {last_err}")


def run_judge(rows, results, args):
    from openai import OpenAI

    load_env_file(args.env_file)
    if not os.environ.get("OPENAI_API_KEY"):
        raise RuntimeError("OPENAI_API_KEY not set (checked env + --env-file).")
    with open(args.prompt_file) as f:
        system_prompt = f.read()
    client = OpenAI()

    n = len(rows)
    for i, row in enumerate(rows):
        idx = str(row["index"])
        entry = results.setdefault(idx, {"index": row["index"]})
        if not args.overwrite and isinstance(entry.get("judge"), dict) \
                and entry["judge"].get("scores"):
            print(f"[judge] {i + 1}/{n} idx={idx} (cached, skip)")
            continue
        text = (row.get(args.field) or "").strip()
        if not text:
            entry["judge"] = {"scores": {}, "raw": None}
        else:
            content = judge_text(client, args.judge_model, system_prompt, text,
                                 args.max_completion_tokens, args.temperature)
            scores, obj = parse_judge_response(content)
            entry["judge"] = {"scores": scores, "raw": obj if obj else content}
            print(f"[judge] {i + 1}/{n} idx={idx} scores={scores}")
        save_results(args.output, results, build_meta(args))
    return results


# --------------------------------------------------------------------------- #
# summary / reporting
# --------------------------------------------------------------------------- #
def _mean(xs):
    xs = [x for x in xs if x is not None]
    return sum(xs) / len(xs) if xs else None


def summarize(per_sample):
    ppls = [r.get("ppl") for r in per_sample]
    def jscore(key):
        return [r.get("judge", {}).get("scores", {}).get(key)
                for r in per_sample if isinstance(r.get("judge"), dict)]
    style = jscore("style")
    consistency = jscore("consistency")
    ethics = jscore("ethics")
    # per-sample style+consistency mean, then averaged
    combined = []
    for s, c in zip(style, consistency):
        if s is not None and c is not None:
            combined.append((s + c) / 2)
    return {
        "num_samples": len(per_sample),
        "num_ppl": len([p for p in ppls if p is not None]),
        "num_judged": len([s for s in style if s is not None]),
        "mean_ppl": _mean(ppls),
        "mean_style": _mean(style),
        "mean_consistency": _mean(consistency),
        "mean_ethics": _mean(ethics),
        "mean_gpt_quality": _mean(combined),  # style+consistency
    }


def print_summary(summary, args):
    def fmt(x):
        return f"{x:.4f}" if isinstance(x, float) else str(x)
    print("\n" + "=" * 60)
    print(f"QUALITY SUMMARY  ({args.field})")
    print(f"  input        : {args.input}")
    print(f"  ppl model    : {args.ppl_model}")
    print(f"  judge model  : {args.judge_model}")
    print("-" * 60)
    print(f"  samples          : {summary['num_samples']}")
    print(f"  ppl computed     : {summary['num_ppl']}")
    print(f"  judged           : {summary['num_judged']}")
    print(f"  mean PPL         : {fmt(summary['mean_ppl'])}")
    print(f"  mean style       : {fmt(summary['mean_style'])}  (1-10)")
    print(f"  mean consistency : {fmt(summary['mean_consistency'])}  (1-10)")
    print(f"  mean ethics      : {fmt(summary['mean_ethics'])}  (1-10)")
    print(f"  mean GPT quality : {fmt(summary['mean_gpt_quality'])}  "
          f"(style+consistency avg)")
    print("=" * 60)


# --------------------------------------------------------------------------- #
# main
# --------------------------------------------------------------------------- #
def build_meta(args):
    return {
        "input": args.input, "field": args.field,
        "ppl_model": args.ppl_model, "judge_model": args.judge_model,
    }


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--input", default=os.path.join(REPO, "outputs/spanwm_v7_c4_n200.jsonl"))
    p.add_argument("--output", default=None,
                   help="results json (default outputs/quality_<inputstem>.json)")
    p.add_argument("--mode", choices=["ppl", "judge", "both"], default="both")
    p.add_argument("--field", default="watermarked_text",
                   help="which text field to score")
    p.add_argument("--ppl-model", default="google/gemma-4-12B")
    p.add_argument("--judge-model", default="gpt-5-mini-2025-08-07")
    p.add_argument("--prompt-file", default=os.path.join(REPO, "gpt_prompt.txt"))
    p.add_argument("--env-file", default="/scratch2/sunny5574/.env",
                   help=".env to source OPENAI_API_KEY from")
    p.add_argument("--temperature", type=float, default=None,
                   help="judge temperature (default: omit; gpt-5* need default)")
    p.add_argument("--max-completion-tokens", type=int, default=3000)
    p.add_argument("--limit", type=int, default=None, help="only first N samples")
    p.add_argument("--overwrite", action="store_true",
                   help="re-judge samples that already have scores")
    args = p.parse_args()
    if args.output is None:
        stem = os.path.splitext(os.path.basename(args.input))[0]
        args.output = os.path.join(REPO, "outputs", f"quality_{stem}.json")
    return args


def main():
    args = parse_args()
    rows = load_jsonl(args.input)
    if args.limit:
        rows = rows[:args.limit]
    print(f"[eval] {len(rows)} samples from {args.input} | mode={args.mode} "
          f"| field={args.field} | output={args.output}")

    results = load_results(args.output)

    if args.mode in ("ppl", "both"):
        results = run_ppl(rows, results, args)
    if args.mode in ("judge", "both"):
        results = run_judge(rows, results, args)

    summary = save_results(args.output, results, build_meta(args))
    print_summary(summary, args)
    print(f"\n[eval] wrote {args.output}")


if __name__ == "__main__":
    main()
