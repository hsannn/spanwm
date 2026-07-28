# SpanWM — Syntax-Guided Span Watermarking

A **span-aware** watermarking method on top of MarkLLM. `README.md` is the design
spec; **this file records what is actually implemented and why.**

## Core idea

Don't spread the watermark over the whole text. Anchor it to a **reproducible
syntactic position** (a keyed grammatical role), watermark a fixed number of
tokens there, and at detection re-find that anchor from the text alone and test
only those tokens — so the statistic isn't diluted by unwatermarked tokens.

- **Where** to watermark (span anchor selection / extraction / mapping) →
  **our logic** (`watermark/spanwm/span_ops.py`)
- **How** to watermark & score (green list / logit bias / z-score) →
  **reused KGW** (`from ..kgw.kgw import KGWUtils, KGWLogitsProcessor`)

Separated on purpose: swapping the in-span watermark for EXP/SynthID is just an
import change in `spanwm.py`.

## THE WORKING RECIPE (current)

> **base model (Llama-3.2-3B) + left-AR free generation + fixed K=20 window +
> γ=0.25 + exact binomial p-value**

Each piece is there to fix a specific failure we hit (see "How we got here").

| Piece | What it means | Why it's needed |
|---|---|---|
| **base model** (Llama-3.2-3B) | raw completion, **no chat template** | chat scaffolding caused a leading-space tokenization drift so detection scored *different* tokens than were watermarked. Pure left-to-right removes it. |
| **left-AR free generation** | feed only the **left context**, freely generate the replacement (no right-context constraint, no "fill the blank" instruction) | a watermark needs **entropy** to bias tokens green. Constraining the span (infilling with left+right) makes it low-entropy → the bias can't land (green rate dropped to ~0–29%). Free continuation is high-entropy → bias lands (green rate ~75–90%). |
| **fixed K=20 window** | watermark exactly **K tokens** from the anchor at embed; test exactly **K tokens** from the reconstructed anchor at detect | (a) guarantees N=K so the test has power (z_max = √(N·(1−γ)/γ)); (b) sidesteps the "reconstruction extent" problem — the parser's constituent **end** is unstable after regeneration, so we ignore it and use a fixed length from the (stable) **start**. |
| **γ=0.25** | green list = 25% of vocab (not 50%) | lower γ packs more evidence per green token: z_max = √N·√((1−γ)/γ). γ=0.5→√N, γ=0.25→1.73·√N. Standard KGW choice. |
| **exact binomial p** | decide by `p = P(Binom(N,γ) ≥ G) < p_threshold`, not by the normal z≈4 | the z-score is a large-N approximation; for a span's small N the exact binomial tail is the correct significance. |

**6-sample probe result:** N=20, G≈15–18 (green rate 75–90%), z≈5–7,
p≈1e-6…1e-9, all detected. (Earlier recipes failed — see below.)

**Cost:** the K=20 free-generated tokens overflow the original short constituent
and ramble into new content, so the watermarked region is **semantically off**
(quality trade-off accepted for the detectability PoC). Tunable via K and δ.

## Pipeline

**Embedding** — `SpanWM.generate_watermarked_text(prompt)`
1. **draft** = raw base-model completion of the prompt (watermark off).
2. Parse the draft (spaCy+textacy) → role-tagged spans; key selects one role →
   first eligible span of that role = the **anchor** (its `start_char`).
3. **left-AR regenerate**: feed `draft[:anchor_start]`, generate exactly K
   watermarked tokens (KGW `LogitsProcessor`), splice `draft[anchor_end:]` back.
4. **verify**: re-parse the final text with the same key; check the reconstructed
   anchor's `start_char` aligns with the watermark start (`anchor_dist ≤ 4`).
   Retry up to `max_regen_attempts`; mark `verified` accordingly.
5. Return final text. Debug info on `self.last_embedding_info` (analysis only —
   detection never reads it).

**Detection** — `SpanWM.detect_watermark(text)`
1. Parse the final text **independently** (no embed-time info).
2. Same key → same role → reconstruct the anchor.
3. `window_positions`: take **K tokens from the anchor** (fixed window).
4. Green-list test → `z`, exact `p`; `is_watermarked = p < p_threshold`.
   Returns `{is_watermarked, score(z), p_value, num_tested_tokens,
   num_green_tokens, selected_role, selected_span, reconstructed}`.

## How we got here (failed recipes — don't repeat)

1. **v1: left-AR, force n_span tokens, test the reconstructed constituent** →
   AUROC ≈ 0.55. Regeneration changed the parse so detection reconstructed a
   **shorter sub-span** (start anchored 84%, but IoU 0.33); tiny N → z can't
   clear threshold.
2. **v2: instruction infilling with a chat model** ("fill the blank",
   left+[BLANK]+right) → AUROC ≈ 0.47 (worse than chance). Two independent
   killers, found by tracing green-rate at generation vs detection:
   - **low entropy**: right-context-constrained fills are near-deterministic
     (`Middletown`), so δ=5 still gave 0–29% green **at generation**.
   - **tokenization drift**: fill generated as `"performance"` but spliced as
     `" performance"` → different BPE id → detection scores different tokens.
3. **v3 (current): base + left-AR + fixed-K** → fixes both (high entropy + pure
   left-to-right tokenization) → z≈5–7.

Lesson: **reconstructable spans (short, syntactic) are low-entropy = hard to
watermark.** Free generation restores entropy; the fixed-K window restores a
usable N without depending on the unstable constituent boundary.

## Where things live

```
watermark/spanwm/
├── span_ops.py     # [OURS] StructuralSpan, SpanExtractor, RoleSelector,
│                   #        TokenSpanMapper (map_positions + window_positions)
└── spanwm.py       # SpanWMConfig, SpanWMUtils (KGW greenlist + select_span +
                    #   score_span), SpanWM (generate/detect/visualize)
config/SpanWM.json                            # configuration
watermark/auto_watermark.py, auto_config.py   # "SpanWM" registered in both
spanwm_embed.py   # [ENTRY] embed over a dataset -> jsonl (needs GPU)
spanwm_detect.py  # [ENTRY] read jsonl -> detect + metrics (same-device as embed!)
_probe_infill.py, _probe_greenrate.py         # diagnostics (keep)
_probe_greenrate_pos.py                       # per-window-position green rate +
                                              #   splice-artifact counts (v6/v7)
parser_testing/spacy_textacy_probe.py         # parser exploration tool
```

### Versions (v4–v7) — separate modules, earlier ones never touched

Each version is its own package `watermark/spanwm_vN/` (subclassing the
previous), with its own `config/SpanWM_vN.json`, `spanwm_embed_vN.py`,
`spanwm_detect_vN.py`, `_probe_vN.py`. History + results: `history.md`.

- **v4**: base-model few-shot infilling (constituent-scale N; weaker).
- **v5**: v3 + per-span PRF role selection (`role = PRF(master_key ‖ 16 chars
  before the anchor)`; the n-gram lies left of the anchor so it is identical
  at embed and detect).
- **v6**: multi-span — up to `max_spans`=4 sites, each a `span_window_tokens`=6
  window (v5 PRF rule per site), embed and detect share the same left→right
  scan (`scan_sites`), green counts of all windows pooled into ONE exact
  binomial test.
- **v8 (current)**: v7 + **two-role anchor PRF**. The per-span PRF now indexes
  the fixed list of `roles_per_anchor`-subsets of `roles` (default k=2, so 3
  pairs out of SUBJECT/OBJECT/PREDICATE) and a span is a site if its role is
  **in** that subset — v5–v7 required equality with a single role. ~2/3 of
  parsed spans become role-eligible instead of ~1/3, so the left→right scan
  reaches `max_spans` sites in more drafts (larger pooled N). Same key material
  and same left-of-anchor n-gram, so the embed/detect invariant is unchanged;
  `roles_per_anchor: 1` reproduces v7's selection byte-for-byte (verified).
  Splicing and the pooled binomial detector are inherited from v7 untouched.
- **v7**: v6 + **splice fixes** in `_regen_at` (detector unchanged
  from v6):
  1. *Drift-free left boundary*: if `left` ends with a space, generate from
     the space-stripped context so the model emits the leading-space token
     itself → the spliced text retokenizes into exactly the generated ids and
     the detector recomputes the same greenlist seeds. (v6 generated after a
     dangling `" "` token; retokenization then merged/shifted the window so
     its tail slid off the fill — measured green rate fell from 0.75 at pos 0
     to 0.53 at pos 5, overall 0.627.) Fallback: fill not starting with a
     space → restore the space (old splice).
  2. *Punctuation-aware right splice*: no separator before `,.;:!?)]}…` etc.
     (v6 stamped `' ,'` into 133/200 watermarked texts vs 1/200 unwatermarked
     — a regex-detectable fingerprint); a trailing-space fill is rstripped
     when the right context supplies its own boundary (kills the double-space
     fingerprint, 23/200 in v6); a fill that OPENS with punctuation is
     resampled up to 3×, else attached directly without a space (that site is
     sacrificed — natural text over fingerprint).

### Key implementation points
- **RoleSelector**: `roles[ int.from_bytes(sha256(master_key||b"role-selection")[:8]) % len(roles) ]`. Fixed role set, **not** a mutable-candidate-list index (which shifts after regeneration). Deterministic; never Python `hash()`.
- **SpanExtractor**: textacy SVO heads + spaCy dep labels, unioned; subject/object head → enclosing `noun_chunk`, predicate → contiguous verb group. Offsets from spaCy (`Token.idx`), never `str.find()`. Used only to locate the **anchor**; the watermarked region is the fixed-K window, not this span.
- **TokenSpanMapper.window_positions(text, anchor_char, K)**: tokenize with offsets, find first real token reaching `anchor_char`, return K consecutive positions. (`map_positions` = trim-then-overlap over a constituent, used only in the legacy chat mode.)
- **KGW reuse**: `SpanWMUtils.kgw = KGWUtils(config)` (green list); `KGWLogitsProcessor(config, kgw)` biases generation; `score_span` computes `z=(G−γN)/√(Nγ(1−γ))` and exact `p=binom.sf(G−1,N,γ)` per token using its true preceding context. `SpanWMConfig` exposes KGW's attribute names so `KGWUtils(config)` works without a `KGWConfig`.
- **Two modes** via `use_chat_template`: **false** (default) = base/left-AR/fixed-K (the working recipe); **true** = the legacy chat-infilling path, kept for reference but known-bad.

## Configuration (`config/SpanWM.json`)

Watermark: `gamma` 0.25, `delta` 4.0, `hash_key`, `prefix_length` 1,
`f_scheme` "time", `window_scheme` "left", `z_threshold` 4.0 (legacy),
`p_threshold` 0.01 (decision).
Span/window: `roles` `["SUBJECT","OBJECT","PREDICATE"]`, `spacy_model`
"en_core_web_sm", `min_span_tokens` 2, `max_span_tokens` 40,
**`detect_window_tokens` 20 (= K)**, `master_key` (hex; role-selection key).
Verification: `max_regen_attempts` 4, `min_reconstruct_overlap` 0.8 (chat mode).
Mode: **`use_chat_template` false**. (`draft/regen_prompt_template`,
`blank_marker`, `system_prompt` only used when true.)

## Metrics (`spanwm_detect.py`)

Positive = watermarked, negative = unwatermarked (default) or `--negative natural`.
- **AUROC**, **TPR@FPR = 10/5/1/0.1%** — ranked by **−log10(exact p)**
  (higher = more watermarked; a failed reconstruction floors to the lowest score).
- **mean z**, **mean p (exact)** over reconstructed samples; reconstruction rate
  printed separately.
- mean p = arithmetic mean of per-sample exact p (descriptive). `sf(mean_z)` is
  wrong (ignores N); a single combined test would be Stouffer `sf(Σz/√N)`.

### jsonl schema (embed → detect)
`index, prompt, watermarked_text, unwatermarked_text, natural_text,
embed_skipped, embed_verified, embed_attempts, embed_anchor_dist, embed_role,
embed_span_text, embed_reconstructed_span, embed_wm_char_range`.
(All `embed_*` are analysis-only; detection re-parses `*_text` and ignores them.)

## Running

- **Model**: `meta-llama/Llama-3.2-3B` (base, gated — the `hsannn` HF login has
  access; downloaded to cache).
- **Env — ALWAYS conda `spanwm`**, by path
  `/home/sunny5574/miniconda3/envs/spanwm/bin/python` (spaCy 3.8.14 + textacy
  0.13.0 + `en_core_web_sm` installed there).
- **GPU — slurm, needs `--overlap`** (an interactive job holds a step; without
  `--overlap` you get "step creation temporarily disabled … nodes busy"):
  ```bash
  srun --jobid=<JOBID> --overlap --chdir=/scratch2/sunny5574/spanwm \
       /home/sunny5574/miniconda3/envs/spanwm/bin/python <script> ...
  ```
- **Detection MUST run on the same device type as embedding.** The KGW
  greenlist is `torch.randperm(vocab, generator=rng)`, and CPU vs CUDA
  generators produce **different sequences from the same seed** — detecting on
  CPU what was embedded on GPU silently yields green rate ≈ γ and AUROC ≈ 0.5
  (measured). So always run `spanwm_detect*.py` under `srun` on a GPU node.

### Workflow policy
- **The user supplies the slurm JOBID** each session (it changes; never reuse an
  old one). Find the node with `squeue -j <JOBID>`.
- **A JOBID from the user = the signal to run the experiment** on it via
  `srun --overlap`.

```bash
python spanwm_embed.py  --dataset c4 --num_samples 200 --output outputs/run.jsonl
python spanwm_detect.py --input outputs/run.jsonl                 # neg = unwatermarked
python spanwm_detect.py --input outputs/run.jsonl --negative natural
```

## Open issues / next

- **Quality**: K=20 free tokens overflow the constituent → semantically off text.
  Tune K down / δ down, or make the window track a coherent unit.
- **Verification rate** ≈ 80% (anchor sometimes doesn't reconstruct); unverified
  samples detect weakly. Improve anchor stability or selection.
- **Low-FPR TPR** is quantized by small N; larger K or multiple spans (Stouffer)
  for FPR=0.1%.
- **Entropy-aware** selection (SWEET/EWD/IE-style) is the principled way to keep
  the watermark localized *and* watermarkable — a natural follow-up.

## References
- Base contract: `watermark/base.py`; registration in `auto_watermark.py` +
  `auto_config.py`. Reused core: `watermark/kgw/kgw.py`.
