# Environment & Licenses

Recorded 2026-07-29. Everything below was read off the machine the experiments
ran on, not off a spec — the pip versions come from
`/home/ssgyejin/miniconda3/envs/spanwm/bin/python`, the GPU line from
`nvidia-smi` inside the standing SLURM allocation.

## 1. Hardware & OS

| | |
|---|---|
| OS | Rocky Linux 9.4 (Blue Onyx), kernel 5.14.0-427.22.1.el9_4.x86_64 |
| Scheduler | Slurm 23.11.6 |
| GPU | 4 × NVIDIA RTX A6000, 49140 MiB (48 GB) each |
| Driver | 580.126.09 |

**One GPU per model, always.** Peer-to-peer copies between this node's A6000s
are silently corrupt (measured on node26: all 12 ordered pairs; each GPU's own
memory is fine), so an `accelerate`-sharded model emits fluent garbage. Every
model in the pipeline had to fit in 48 GB — that is why gemma-4-31B-it runs from
Google's QAT w4a16 checkpoint (~23 GB) instead of the bf16 repo (~63 GB).
See the header of [run_paraphrase_attack.sh](attacks/run_paraphrase_attack.sh).

The login node has no GPU; GPU work attaches to the standing allocation with
`srun --jobid=<id> --overlap`.

## 2. Python environment

conda env **`spanwm`** — `/home/ssgyejin/miniconda3/envs/spanwm`, **Python 3.12.13**.

Note: a `.venv` on `PATH` shadows this env, so `conda run -n spanwm python`
resolves to `/home/ssgyejin/.venv/bin/python` (Python 3.9.18). Call the
interpreter by absolute path, as the run scripts do.

### Core

| Package | Version |
|---|---|
| torch | 2.13.0+cu130 |
| CUDA (torch build) | 13.0 |
| cuDNN | 9.20.0 |
| NCCL | 2.29.7 |
| triton | 3.7.1 |
| transformers | 5.14.1 |
| tokenizers | 0.22.2 |
| accelerate | 1.14.0 |
| huggingface_hub | 1.24.0 |
| safetensors | 0.8.0 |
| compressed-tensors | 0.17.1 |

`compressed-tensors` is what lets the gemma-4-31B QAT w4a16 checkpoint load; it
is pulled in indirectly by `transformers` and is **not pinned in
requirements.txt**.

### Numeric / NLP

| Package | Version |
|---|---|
| numpy | 2.5.1 |
| scipy | 1.18.0 |
| scikit-learn | 1.9.0 |
| pandas | 3.0.5 |
| nltk | 3.10.0 |
| spacy | 3.8.14 |
| en_core_web_sm | 3.8.0 |
| textacy | 0.13.0 |
| sentence-transformers | 5.6.1 |
| tiktoken | 0.13.0 |
| openai | 2.48.0 |
| python-dotenv | 1.2.2 |
| translate / libretranslatepy | 3.8.1 / 2.1.1 |

NLTK corpora present in `~/nltk_data/corpora`: `wordnet`, `omw-1.4`, `cmudict`
— `wordnet` is what the synonym-substitution attack needs.

### Not installed in `spanwm` (code paths that will not run here)

| Missing | Used by |
|---|---|
| matplotlib | [attacks/plot_attack_auroc.py](attacks/plot_attack_auroc.py), [attacks/plot_tpr_by_fpr.py](attacks/plot_tpr_by_fpr.py), [visualize/color_scheme.py](visualize/color_scheme.py) — run these with the `diffuguard` env |
| vllm | [MarkvLLM_demo.py](MarkvLLM_demo.py), [baseline_ppl_vllm.py](baseline_ppl_vllm.py), [watermark/kgw/kgw_logits_processor_for_vllm.py](watermark/kgw/kgw_logits_processor_for_vllm.py) |
| sacrebleu | [evaluation/tools/text_quality_analyzer.py](evaluation/tools/text_quality_analyzer.py) (BLEU path) |
| datasets | [dataset/c4-train/](dataset/c4-train/) is an Arrow dir; loading it needs `datasets` |

`requirements.txt` pins agree with the installed env for every package it lists.
It omits `compressed-tensors`, `tiktoken`, `translate`, and the `en_core_web_sm`
model wheel, all of which the experiment path touches.

## 3. Models used

### Generators (watermark embedding)

| Model | License |
|---|---|
| Qwen/Qwen3-4B, Qwen3-8B | Apache-2.0 |
| meta-llama/Llama-3.2-3B | Llama 3.2 Community License |
| meta-llama/Llama-3.1-8B | Llama 3.1 Community License |
| google/gemma-4-12B | Gemma Terms of Use |

### Paraphrase attackers

| Model | License |
|---|---|
| google/gemma-4-12B-it, gemma-4-31B-it-qat-w4a16-ct | Gemma Terms of Use |
| Qwen/Qwen3-14B | Apache-2.0 |
| openai/gpt-oss-20b | Apache-2.0 |
| gpt-5-mini (hosted) | OpenAI API terms — commercial, needs `OPENAI_API_KEY` in `.env` |

### Evaluation & auxiliary

| Model | Used by | License |
|---|---|---|
| gpt-5-mini-2025-08-07 | GPT judge in [eval_quality.py](eval_quality.py), [judge_baselines.py](judge_baselines.py) | OpenAI API terms |
| google/gemma-4-12B | PPL model, `--ppl-model` default in [eval_quality.py](eval_quality.py) | Gemma Terms of Use |
| princeton-nlp/sup-simcse-roberta-base | IE embedding, LTW semantic model, IE tagger training | MIT |
| gpt2-large | Adaptive measurement model | MIT |
| sentence-transformers/all-mpnet-base-v2 | Adaptive embedding model | Apache-2.0 |
| en_core_web_sm 3.8.0 | spaCy parse for SpanWM / SpARK | MIT |

`facebook/opt-1.3b` and `facebook/nllb-200-distilled-600M` appear only in
inherited MarkLLM demo code ([evaluation/examples/](evaluation/examples/),
[test/](test/)) and were **not** used in these experiments. Worth noting if that
ever changes: OPT is research-only and NLLB-200 is CC-BY-NC-4.0, both
non-commercial.

### Local checkpoints shipped in-tree

- [watermark/ie/model/](watermark/ie/model/) — `entropy_tagger_{0_9,2_2,3_5}.pt`,
  trained here by [train_ie_tagger.py](train_ie_tagger.py) on SimCSE features.
- [watermark/adaptive/model/semantic_mapping_model.pth](watermark/adaptive/model/) —
  from the Adaptive baseline, inherited with the MarkLLM tree.

## 4. Datasets

Local copies under [dataset/](dataset/), n=200 prompts per run.

| Dataset | License |
|---|---|
| C4 (`realnewslike`) | ODC-BY 1.0; underlying Common Crawl content under its own terms |
| CNN/DailyMail | Apache-2.0 (script); articles remain © their publishers |
| WMT16 de-en | mixed per sub-corpus; Europarl/news-commentary generally research-permissive |
| HumanEval | MIT |

## 5. Code licenses

This repository is **Apache License 2.0** ([LICENSE](LICENSE)).

### Third-party code in-tree

| Component | Origin | License |
|---|---|---|
| [watermark/base.py](watermark/base.py), [auto_watermark.py](watermark/auto_watermark.py), [auto_config.py](watermark/auto_config.py), [watermark/kgw/](watermark/kgw/), [sweet/](watermark/sweet/), [ewd/](watermark/ewd/), [adaptive/](watermark/adaptive/), [evaluation/tools/](evaluation/tools/), [visualize/](visualize/), [exceptions/](exceptions/) | THU-BPM **MarkLLM** | Apache-2.0 (headers intact) |
| [watermark/ltw/detect_utils.py](watermark/ltw/detect_utils.py) | Kirchenbauer et al., *A Watermark for LLMs* (arXiv 2301.10226) | Apache-2.0 (header intact) |
| [watermark/sparkr/](watermark/sparkr/), [watermark/sparkp/](watermark/sparkp/) | reimplementation from `mail-research/SpARK-llm-watermarking` (upstream not runnable as committed) | upstream **MIT** |
| [watermark/ltw/watermark.py](watermark/ltw/watermark.py) and the rest of [watermark/ltw/](watermark/ltw/) | vendored from `fattyray/learning-to-watermark` (NeurIPS 2025), commit of 2025-10-11, reproduction fixes only | ⚠ **no license file upstream** |
| [watermark/ie/](watermark/ie/) | no provenance header | ⚠ **unverified** |
| [watermark/spanwm*/](watermark/) (v4–v8) | ours | Apache-2.0 |

### Two things to resolve before release

1. **LTW has no upstream license.** `fattyray/learning-to-watermark` ships no
   LICENSE file, so by default all rights are reserved and the vendored copy in
   [watermark/ltw/](watermark/ltw/) is not redistributable. Either get written
   permission from the authors, or drop the directory and ship a patch/fetch
   script that pulls it from upstream at setup time.
2. **IE has no attribution header.** [watermark/ie/](watermark/ie/) carries no
   copyright or upstream pointer, unlike every other baseline. Trace where it
   came from and add the header, or confirm it is an original implementation.

Everything else is Apache-2.0 or MIT and is fine to redistribute under this
repo's Apache-2.0 with attribution preserved.

Model licenses above are from the model cards as generally published; re-check
each card at release time, particularly the Llama and Gemma terms, which carry
use restrictions that ordinary open-source licenses do not.
