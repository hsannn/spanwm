# Baseline reproduction package: ATW / IE / LTW (+ EWD-default)

This branch contains exactly what is needed to **reproduce our baseline
experiments** (the generated corpora, detection, and quality measurement) for
the three reproduced watermarks and the EWD default setting. Attack code is
NOT included — attacks are developed separately; any attacked text saved in
the same record schema can be re-scored with the harness here (see
"Re-detecting modified texts").

## What is in here

| Piece | Files | Notes |
|---|---|---|
| ATW (ICML'24) | `watermark/adaptive/` (+ `model/semantic_mapping_model.pth`, 4.5 MB) | MarkLLM port (authored by the paper's first author). Uses `len(tokenizer)` so the Llama-3.2 vocab bug of the original repo does not apply. |
| IE (EMNLP'25) | `watermark/ie/` (+ `model/entropy_tagger_{0_9,2_2,3_5}.pt`, 2.3 MB each) | Taggers are **our in-domain re-distillations** (see "Trained components"). |
| LTW (NeurIPS'25) | `watermark/ltw/` (+ `selective_network_epoch0_step2000.pth`, 825 KB) | Vendored from the official repo with three documented reproduction fixes (see module docstring: generation/detection vocab-size unification, SimCSE mask shape, `skip_special_tokens`). Driver: `ltw_run.py` (pass `k=6`; the class default 10 mismatches generation). |
| EWD (ACL'24) | `watermark/ewd/`, `config/EWD.json` (γ=.5, δ=2 MarkLLM default), `config/EWD_g0.25_d4.0.json` | |
| Harness | `baseline_embed.py`, `baseline_detect.py` | Generation + detection with AUROC / TPR@{10,5,1,0.1}% by exact-p ranking. |
| Quality (optional) | `judge_baselines.py` + `GPT_EVAL_PROMPT.md` (GPT judge), `baseline_ppl_vllm.py` (PPL; forces BOS — required for gemma oracles) | |
| Corpora (the texts to attack) | `outputs/{atw_d1.5,ie_t0.9,ie_t2.2,ie_t3.5,ltw1,ltw0,ewd_std}_c4_n200.jsonl` | 200 records each: `{index, prompt, watermarked_text, unwatermarked_text, natural_text}`; texts include the prompt (strip char-level before scoring — `baseline_detect` does this). **Attack these files; do not regenerate** (sampling will not reproduce them). |
| IE re-distillation | `train_ie_tagger.py` | Only needed if you change base model or domain. |

## Environment

Two conda envs (they conflict; do not merge):

- **detection/generation env** — python 3.12, `torch 2.10`, `transformers 4.55.4`
  (use `torch_dtype=`, not `dtype=`), `scipy`, `scikit-learn`, `sentencepiece`,
  `sentence-transformers>=3` (ATW).
- **vllm env** — `vllm 0.25.x`, `transformers 5.x`: only for the gemma PPL
  quality oracle. gemma-4-12B needs ≥40 GB (A6000-class); it does not fit a
  24 GB card.

Models downloaded on first run: `meta-llama/Llama-3.2-3B` (gated — needs an HF
account with access), `gpt2-large` + `sentence-transformers/all-mpnet-base-v2`
(ATW detection), `princeton-nlp/sup-simcse-roberta-base` (IE + LTW).

## ⚠ GPU-family rule for re-detection (silent-failure risk)

`torch.randperm` on CUDA gives **different permutations on different GPU
families for the same seed** (measured: A6000 vs RTX 3090 disagree at
n=128256; CPU differs from both). Green lists built on one family do not
reproduce on another — detection silently degrades toward chance while
looking plausible.

- **LTW and IE detection: run on the same GPU family that generated the
  corpora — RTX 3090.** (Both derive green lists via CUDA `randperm`.)
- ATW: device-independent (Python `random` mapping). EWD: device-independent
  (large-integer seeded hashing).

## Sanity gate before attacking (do not skip)

Re-detect each **clean** corpus first and compare against our measured
numbers. If yours differ by more than ~0.01, the environment (usually the GPU
family) is wrong, and any attack numbers would be invalid.

```
python baseline_detect.py --input outputs/atw_d1.5_c4_n200.jsonl --algorithm Adaptive --config config/Adaptive.json --model <llama-3.2-3b path> --negative unwatermarked
python baseline_detect.py --input outputs/ie_t0.9_c4_n200.jsonl  --algorithm IE --config config/IE_t0.9.json  --model <path> --negative unwatermarked
python ltw_run.py --model <path> --variant ltw1 --skip_generate      # detects outputs/ltw1_c4_n200.jsonl
python baseline_detect.py --input outputs/ewd_std_c4_n200.jsonl --algorithm EWD --config config/EWD.json --model <path> --negative unwatermarked
```

Expected (TPR@0.1% / AUROC, negative = paired unwatermarked, n=200):

| arm | TPR@0.1% | AUROC | mean z |
|---|---|---|---|
| ATW δ=1.5 | 0.975 | 0.9999 | +8.95 |
| IE τ=0.9 | 0.985 | 0.9945 | +10.17 |
| IE τ=2.2 | 0.910 | 0.9924 | +7.97 |
| IE τ=3.5 | 0.601 | 0.9601 | +3.80 |
| LTW-1 | 0.995 | 1.0000 | +12.70 |
| LTW-0 | 0.995 | 0.9998 | +12.42 |
| EWD γ.5/δ2 | 0.990 | 0.9987 | +8.85 |

## Re-detecting modified texts

Detection is decoupled from how a text was produced: any jsonl that keeps the
record schema (`index, prompt, watermarked_text, unwatermarked_text,
natural_text`) can be scored with the same commands by pointing `--input` at
it — e.g., write attacked variants of a corpus into a new jsonl and re-run
`baseline_detect.py` unchanged. For LTW, use
`ltw_run.py --variant ltw{0,1} --skip_generate --output <file>` (its detector
recomputes entropy with the base LM, and the GPU-family rule above applies).

## Trained components: provenance and reuse rules

Three learned artifacts ship in-repo (total ≈ 7.4 MB — small enough that git
is the right distribution channel; no LFS needed):

1. **ATW semantic-mapping model** (`watermark/adaptive/model/`): the original
   authors' released checkpoint, LM-agnostic by construction (maps sentence
   embeddings; author-confirmed usable with any tokenizer size). Reusable
   as-is with any base model.
2. **LTW selector** (`watermark/ltw/selective_network_epoch0_step2000.pth`):
   the official released checkpoint (trained on OPT-1.3B). We measured that
   it transfers to Llama-3.2-3B (gating rate 0.38–0.41 vs 0.45–0.49 in the
   paper). Reusable as-is; if you switch to a very different model family,
   check the gating rate first.
3. **IE entropy taggers** (`watermark/ie/model/entropy_tagger_*.pt`): **our
   re-distillations, NOT the released ones.** The published checkpoints
   (HF `Carol0110/IE`) are distilled from StarCoder on MBPP code and score
   AUROC 0.45–0.50 (chance) on Llama-3.2-3B/C4 — unusable here. Ours are
   distilled from Llama-3.2-3B true next-token entropy on C4 rows 1000+
   (τ = 0.9/2.2/3.5 = our corpus entropy P25/P50/P75; held-out AUROC
   0.79–0.83). **They are just as model/domain-bound as the originals**: if
   you change the base model or the text domain, re-distill with
   `train_ie_tagger.py --model <new model>` (~1–2 h on one GPU) — do not
   reuse these taggers.

## Protocol constants (do not change if numbers should stay comparable)

C4 first 200 prompts (`dataset/c4/processed_c4.json`), Llama-3.2-3B base,
`max_new_tokens=200, do_sample, top_p=0.9, temperature=0.8`, seed 42.
Decoding exceptions (baked into the corpora): ATW and LTW use their papers'
own sampling loops; IE and EWD use the shared protocol. Detection strips the
prompt char-level and scores the continuation; negatives = paired
unwatermarked (same file) and `natural_text`.
