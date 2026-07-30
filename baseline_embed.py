"""KGW / SynthID baseline embedding, matched to the SpanWM v6 experiment.

Generation settings are IDENTICAL to spanwm_embed_v6.py so the resulting jsonl
is directly comparable to the SpanWM v6 run:
    meta-llama/Llama-3.2-3B (base, no chat template), C4 first N prompts,
    max_new_tokens=200, do_sample=True, top_p=0.9, temperature=0.8.
Watermarked / unwatermarked texts INCLUDE the prompt (raw completion decode),
exactly like SpanWM's draft path; baseline_detect.py strips it before scoring.

One config per algorithm (config/KGW.json, config/SWEET.json, ...). A
reported row that only changes a value is that same file with the field
edited, not a second config file:
    KGW gamma 0.5/delta 2.0 (as shipped) or 0.25/4.0
    SpARK-P pos_tags ["V"] (Verb) / ["N"] (Noun) / ["DT"] (Det)
    SWEET, IE  entropy_threshold = the cell's calibrated tau
    LTW     variant ltw0 (as shipped) or ltw1
tools/make_cfg.py writes the per-cell tau/tagger variants into
config/generated/, which is not committed.
SynthID uses MarkLLM defaults (config/SynthID.json, mean detector,
non-distortionary). Its internal pre-tournament temperature is aligned to the
sampling temperature (0.8) below; HF's temperature/top-p warpers still apply
after the tournament reweighting — that is MarkLLM's standard behavior and
does not affect g-value detection.

NOTE (SynthID): MarkLLM's SynthIDLogitsProcessor keeps generation state across
generate() calls (context + repeated-context history bleed between samples).
We reset it before every sample so each text is watermarked independently,
matching the per-sequence semantics of the reference implementation.

Run (GPU node, wmattack env):
    python baseline_embed.py --algorithm KGW
    python baseline_embed.py --algorithm SynthID
"""

import argparse
import json
import os
import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from watermark.auto_watermark import AutoWatermark
from utils.transformers_config import TransformersConfig
from utils.lm_compat import dtype_kwargs, vocab_size as lm_vocab_size
from evaluation.dataset import (C4Dataset, CNN_DailyMailDataset,
                                WMT16DE_ENDataset)
# the continuation loaders live apart so evaluation/dataset.py (the
# collaborator's file) stays untouched
from evaluation.dataset_continuation import (CNNArticleDataset,
                                             WMT16ENDataset)

MODEL_ID = "meta-llama/Llama-3.2-3B"
# All three loaders take lines[:max_samples] -> the FIRST N rows, in file
# order, no shuffling. Same selection convention across datasets.
DATASETS = {
    "c4": (C4Dataset, "dataset/c4/processed_c4.json"),
    # CNN and Daily Mail are reported SEPARATELY (200 each). The merged
    # corpus mixes two outlets and the HF release strips bylines, so the two
    # are split by SHA1(source URL) -- see tmp_sync/split_by_url.py.
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
DEFAULT_CONFIGS = {
    "KGW": "config/KGW.json",
    "SynthID": "config/SynthID.json",
    "SpARKP": "config/SpARKP.json",
    "SpARKR": "config/SpARKR.json",
    "SWEET": "config/SWEET.json",
    "EWD": "config/EWD.json",
    "Adaptive": "config/Adaptive.json",
    "IE": "config/IE.json",
}


def load_dataset(name, num_samples):
    if name not in DATASETS:
        raise ValueError(f"unknown dataset '{name}'. choices: {list(DATASETS)}")
    cls, path = DATASETS[name]
    return cls(path, max_samples=num_samples)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--algorithm", required=True, choices=list(DEFAULT_CONFIGS))
    ap.add_argument("--dataset", default="c4", choices=list(DATASETS))
    ap.add_argument("--num_samples", type=int, default=200)
    ap.add_argument("--model", default=MODEL_ID)
    ap.add_argument("--config", default=None)
    ap.add_argument("--output", default=None)
    ap.add_argument("--force", action="store_true",
                    help="overwrite an existing output file")
    ap.add_argument("--max_new_tokens", type=int, default=200)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    config_path = args.config or DEFAULT_CONFIGS[args.algorithm]
    # Output name, in precedence order:
    #   1. --output                     explicit, wins over everything
    #   2. "output_tag" in the config    set this when sweeping a value INSIDE
    #                                    one config file (e.g. IE's tau), since
    #                                    the file name alone cannot distinguish
    #                                    those runs
    #   3. the config file stem          distinguishes different config FILES
    #                                    of the same algorithm (SpARKR.json vs
    #                                    SpARKR_softfix.json), which naming by
    #                                    algorithm alone did not
    with open(config_path) as _f:
        cfg_dict = json.load(_f)  # read once; also recorded in the meta json
    tag = cfg_dict.get("output_tag")
    stem = tag or os.path.splitext(os.path.basename(config_path))[0]
    output = args.output or f"outputs/{stem.lower()}_{args.dataset}_n{args.num_samples}.jsonl"
    os.makedirs(os.path.dirname(output), exist_ok=True)
    # Last line of defence: never silently replace a corpus that cost GPU-hours.
    if os.path.exists(output) and not args.force:
        raise SystemExit(
            f"refusing to overwrite {output}\n"
            f"  set \"output_tag\" in {config_path} (or pass --output) to give "
            f"this run its own file,\n"
            f"  or pass --force to overwrite")

    torch.manual_seed(args.seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device={device}  model={args.model}  algorithm={args.algorithm}  "
          f"config={config_path}  dataset={args.dataset}  N={args.num_samples}  seed={args.seed}", flush=True)

    ds = load_dataset(args.dataset, args.num_samples)
    n = min(args.num_samples, ds.prompt_nums)
    gen_times, unwm_times = [], []

    _load0 = time.perf_counter()

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    # torch_dtype (not the 4.56+ `dtype` alias): the cluster env runs
    # transformers 4.55.4. "auto" -> bf16 for Llama-3.2 (matches the SpanWM run).
    model = AutoModelForCausalLM.from_pretrained(
        args.model, **dtype_kwargs("auto")).to(device)
    model.eval()
    _load_s = time.perf_counter() - _load0
    print(f"model dtype={next(model.parameters()).dtype}", flush=True)

    # Same sampling setup as spanwm_embed_v6.py.
    transformers_config = TransformersConfig(
        model=model, tokenizer=tokenizer,
        vocab_size=lm_vocab_size(model, tokenizer),
        device=device, max_new_tokens=args.max_new_tokens,
        do_sample=True, top_p=0.9, temperature=0.8,
    )
    if args.algorithm == "SynthID":
        # SynthIDConfig reads these via getattr on the TransformersConfig object
        # (they are NOT picked up from gen_kwargs): pre-tournament temperature
        # scaling and tournament top-k restriction.
        transformers_config.temperature = 0.8
        transformers_config.top_k = -1  # tournament over the full vocab

    watermark = AutoWatermark.load(args.algorithm, algorithm_config=config_path,
                                   transformers_config=transformers_config)

    n_params = sum(p.numel() for p in model.parameters())
    meta = {
        "model_path": args.model,
        "model_name": getattr(model.config, "_name_or_path", args.model),
        "model_type": getattr(model.config, "model_type", None),
        "n_params": int(n_params),
        "n_params_b": round(n_params / 1e9, 2),
        "dtype": str(next(model.parameters()).dtype),
        # The greenlist is randperm(vocab_size), so detection has to key on the
        # SAME value generation used. len(tokenizer) differs from
        # config.vocab_size on Qwen (151669 vs 151936) though not on Llama, so
        # record both and let baseline_detect.py bind to them (--vocab_size
        # model|tokenizer) instead of guessing.
        "vocab_size": lm_vocab_size(model, tokenizer),
        "tokenizer_len": len(tokenizer),
        # measured once per run, never per sample: tools/paper_timing_table.py
        # reports it separately so load cost can never inflate a per-sample figure
        "model_load_s": round(_load_s, 2),
        "algorithm": args.algorithm,
        "config_file": config_path,
        "config": {k: v for k, v in cfg_dict.items() if not k.startswith("_")},
        "dataset": args.dataset,
        "num_samples": n,
        "max_new_tokens": args.max_new_tokens,
        "seed": args.seed,
        "sampling": {"do_sample": True, "top_p": 0.9, "temperature": 0.8},
    }
    print(f"meta: {meta['model_name']} ({meta['n_params_b']}B, {meta['dtype']})  "
          f"vocab {meta['vocab_size']} (tokenizer {meta['tokenizer_len']})  "
          f"load {meta['model_load_s']}s  "
          f"{args.algorithm} <- {config_path}  dataset={args.dataset}", flush=True)

    print(f"gen_kwargs={transformers_config.gen_kwargs}", flush=True)
    print(f"model.generation_config={model.generation_config}", flush=True)

    with open(output, "w") as fout:
        for i in range(n):
            prompt = ds.get_prompt(i)
            # C4 carries natural_text; CNN/DM and WMT16 carry references
            if ds.natural_text_nums > i:
                natural = ds.get_natural_text(i)
            elif ds.reference_nums > i:
                natural = ds.get_reference(i)
            else:
                natural = ""

            if args.algorithm == "SynthID":
                watermark.logits_processor.state = None  # fresh per-sample state
            t0 = time.perf_counter()
            wm_text = watermark.generate_watermarked_text(prompt)
            t_wm = time.perf_counter() - t0
            t0 = time.perf_counter()
            unwm_text = watermark.generate_unwatermarked_text(prompt)
            t_unwm = time.perf_counter() - t0
            gen_times.append(t_wm)
            unwm_times.append(t_unwm)

            record = {
                "index": i,
                "prompt": prompt,
                "watermarked_text": wm_text,
                "unwatermarked_text": unwm_text,
                "natural_text": natural,
                # end-to-end wall time for THIS sample (seconds)
                "gen_time_watermarked": round(t_wm, 4),
                "gen_time_unwatermarked": round(t_unwm, 4),
                "meta": meta,
            }
            fout.write(json.dumps(record, ensure_ascii=False) + "\n")
            fout.flush()

            n_wm = len(tokenizer(wm_text, add_special_tokens=False)["input_ids"])
            n_un = len(tokenizer(unwm_text, add_special_tokens=False)["input_ids"])
            print(f"[{i + 1:>4}/{n}] wm_tokens={n_wm} unwm_tokens={n_un}", flush=True)

    import statistics as _st
    summary = dict(meta)
    summary["timing"] = {
        "gen_watermarked_mean_s": round(_st.mean(gen_times), 4),
        "gen_watermarked_median_s": round(_st.median(gen_times), 4),
        "gen_watermarked_total_s": round(sum(gen_times), 2),
        "gen_unwatermarked_mean_s": round(_st.mean(unwm_times), 4),
        "gen_unwatermarked_total_s": round(sum(unwm_times), 2),
        "overhead_ratio": round(_st.mean(gen_times) / max(_st.mean(unwm_times), 1e-9), 3),
        "n": len(gen_times),
    }
    meta_path = output.replace(".jsonl", ".meta.json")
    with open(meta_path, "w") as mf:
        json.dump(summary, mf, indent=2)
    t = summary["timing"]
    print(f"\ntiming: watermarked {t['gen_watermarked_mean_s']:.3f}s/sample "
          f"(median {t['gen_watermarked_median_s']:.3f}, total {t['gen_watermarked_total_s']:.1f}s)  "
          f"| unwatermarked {t['gen_unwatermarked_mean_s']:.3f}s/sample  "
          f"| overhead {t['overhead_ratio']:.2f}x", flush=True)
    print(f"wrote {n} records -> {output}", flush=True)
    print(f"wrote metadata     -> {meta_path}", flush=True)


if __name__ == "__main__":
    main()
