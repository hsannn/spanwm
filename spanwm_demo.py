"""SpanWM demo: watermark one syntactic span via local regeneration, then
detect the watermark by reconstructing that span from the final text.

No vLLM. Plain HuggingFace transformers with Qwen3-4B-Instruct-2507.

Run (on a GPU node, dlmwm env):
    python spanwm_demo.py
    python spanwm_demo.py --prompt "Your prompt here."
    python spanwm_demo.py --model Qwen/Qwen3-4B-Instruct-2507
"""

import argparse
import json

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from watermark.auto_watermark import AutoWatermark
from utils.transformers_config import TransformersConfig

MODEL_ID = "Qwen/Qwen3-4B-Instruct-2507"


def load_prompt(cli_prompt: str | None) -> tuple[str, str]:
    if cli_prompt:
        return cli_prompt, ""
    with open("dataset/c4/processed_c4.json") as f:
        item = json.loads(f.readline())
    return item["prompt"], item.get("natural_text", "")


def show(title: str, text: str, result: dict) -> None:
    print(f"\n{'=' * 78}\n{title}\n{'-' * 78}")
    print(text)
    print("-" * 78)
    if result.get("reconstructed"):
        print(f"  role            : {result['selected_role']}")
        print(f"  span            : {result['selected_span']!r}  {result.get('span_char_range')}")
        print(f"  tested tokens N : {result['num_tested_tokens']}   green G : {result['num_green_tokens']}")
        z = result["score"]
        print(f"  z-score         : {z:.4f}" if z is not None else "  z-score         : n/a")
    else:
        print(f"  role            : {result['selected_role']}  (no eligible span -> skipped)")
    print(f"  is_watermarked  : {result['is_watermarked']}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--prompt", default=None)
    ap.add_argument("--model", default=MODEL_ID)
    ap.add_argument("--max_new_tokens", type=int, default=200)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device={device}  model={args.model}")

    prompt, natural_text = load_prompt(args.prompt)

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype="auto", device_map=device,
    )
    model.eval()

    transformers_config = TransformersConfig(
        model=model,
        tokenizer=tokenizer,
        vocab_size=model.config.vocab_size,
        device=device,
        max_new_tokens=args.max_new_tokens,
        do_sample=True,
        top_p=0.9,
        temperature=0.8,
    )

    watermark = AutoWatermark.load(
        "SpanWM",
        algorithm_config="config/SpanWM.json",
        transformers_config=transformers_config,
    )

    print(f"\nprompt: {prompt!r}")

    # 1) watermarked (draft -> span regen)
    wm_text = watermark.generate_watermarked_text(prompt)
    info = watermark.last_embedding_info
    if not info["skipped"]:
        sp = info["selected_span"]
        print(f"\n[embed] role={sp.role}  span={sp.text!r}  chars={sp.start_char}:{sp.end_char}")
        print(f"[embed] draft span text was regenerated under watermark.")
    else:
        print("\n[embed] no eligible span in draft -> returned unwatermarked draft.")

    # 2) unwatermarked baseline
    unwm_text = watermark.generate_unwatermarked_text(prompt)

    # 3) detect on all three
    r_wm = watermark.detect_watermark(wm_text)
    r_unwm = watermark.detect_watermark(unwm_text)

    show("WATERMARKED (span-regenerated)", wm_text, r_wm)
    show("UNWATERMARKED (plain generation)", unwm_text, r_unwm)
    if natural_text:
        r_nat = watermark.detect_watermark(natural_text)
        show("NATURAL TEXT (human)", natural_text, r_nat)


if __name__ == "__main__":
    main()
