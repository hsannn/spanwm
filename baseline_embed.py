"""KGW / SynthID baseline embedding, matched to the SpanWM v6 experiment.

Generation settings are IDENTICAL to spanwm_embed_v6.py so the resulting jsonl
is directly comparable to the SpanWM v6 run:
    meta-llama/Llama-3.2-3B (base, no chat template), C4 first N prompts,
    max_new_tokens=200, do_sample=True, top_p=0.9, temperature=0.8.
Watermarked / unwatermarked texts INCLUDE the prompt (raw completion decode),
exactly like SpanWM's draft path; baseline_detect.py strips it before scoring.

KGW config is the SpanWM-matched strength (config/KGW_g0.25_d4.0.json:
gamma=0.25, delta=4.0, same hash_key / prefix_length / schemes as SpanWM).
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

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from watermark.auto_watermark import AutoWatermark
from utils.transformers_config import TransformersConfig
from evaluation.dataset import C4Dataset

MODEL_ID = "meta-llama/Llama-3.2-3B"
DATASETS = {"c4": (C4Dataset, "dataset/c4/processed_c4.json")}
DEFAULT_CONFIGS = {
    "KGW": "config/KGW_g0.25_d4.0.json",
    "SynthID": "config/SynthID.json",
    "SpARKP": "config/SpARKP.json",
    "SpARKR": "config/SpARKR.json",
    "LemmaWM": "config/LemmaWM.json",
    "LemmaWMS": "config/LemmaWMS_k2.json",
    "ClusterWM": "config/ClusterWM_k2.json",
    "SentClusterWM": "config/SentClusterWM.json",
    "SWEET": "config/SWEET_g0.25_d4.0_t0.9.json",
    "EWD": "config/EWD_g0.25_d4.0.json",
    "Adaptive": "config/Adaptive.json",
    "IE": "config/IE.json",
    "PivotWM": "config/PivotWM.json",
    "SpanCode": "config/SpanCode.json",
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
    # Name the output after the CONFIG, not the algorithm: several configs of
    # the same algorithm (e.g. SpARKR.json vs SpARKR_softfix.json) would
    # otherwise write to the same file and silently overwrite each other.
    cfg_stem = os.path.splitext(os.path.basename(config_path))[0].lower()
    output = args.output or f"outputs/{cfg_stem}_{args.dataset}_n{args.num_samples}.jsonl"
    os.makedirs(os.path.dirname(output), exist_ok=True)
    # A value swept INSIDE one config file (e.g. IE's entropy_threshold) does
    # not change the config name, so refuse to clobber instead of silently
    # replacing a corpus that took GPU-hours to make.
    if os.path.exists(output) and not args.force:
        raise SystemExit(
            f"refusing to overwrite {output}\n"
            f"  sweeping a value inside {config_path}? pass an explicit "
            f"--output outputs/<name>_{args.dataset}_n{args.num_samples}.jsonl\n"
            f"  or pass --force to overwrite")

    torch.manual_seed(args.seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device={device}  model={args.model}  algorithm={args.algorithm}  "
          f"config={config_path}  dataset={args.dataset}  N={args.num_samples}  seed={args.seed}", flush=True)

    ds = load_dataset(args.dataset, args.num_samples)
    n = min(args.num_samples, ds.prompt_nums)

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    # torch_dtype (not the 4.56+ `dtype` alias): the cluster env runs
    # transformers 4.55.4. "auto" -> bf16 for Llama-3.2 (matches the SpanWM run).
    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype="auto").to(device)
    model.eval()
    print(f"model dtype={next(model.parameters()).dtype}", flush=True)

    # Same sampling setup as spanwm_embed_v6.py.
    transformers_config = TransformersConfig(
        model=model, tokenizer=tokenizer, vocab_size=model.config.vocab_size,
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

    print(f"gen_kwargs={transformers_config.gen_kwargs}", flush=True)
    print(f"model.generation_config={model.generation_config}", flush=True)

    with open(output, "w") as fout:
        for i in range(n):
            prompt = ds.get_prompt(i)
            natural = ds.get_natural_text(i) if ds.natural_text_nums > i else ""

            if args.algorithm == "SynthID":
                watermark.logits_processor.state = None  # fresh per-sample state
            wm_text = watermark.generate_watermarked_text(prompt)
            unwm_text = watermark.generate_unwatermarked_text(prompt)

            record = {
                "index": i,
                "prompt": prompt,
                "watermarked_text": wm_text,
                "unwatermarked_text": unwm_text,
                "natural_text": natural,
            }
            fout.write(json.dumps(record, ensure_ascii=False) + "\n")
            fout.flush()

            n_wm = len(tokenizer(wm_text, add_special_tokens=False)["input_ids"])
            n_un = len(tokenizer(unwm_text, add_special_tokens=False)["input_ids"])
            print(f"[{i + 1:>4}/{n}] wm_tokens={n_wm} unwm_tokens={n_un}", flush=True)

    print(f"\nwrote {n} records -> {output}", flush=True)


if __name__ == "__main__":
    main()
