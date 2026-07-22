# Syntax-Guided Span Watermarking

## Overview

This project explores a **syntax-guided span watermarking** method for machine-generated text.

The main idea is simple:
1. Generate a normal, unwatermarked draft.
2. Parse the draft and locate a fixed set of syntactic roles (e.g. SUBJECT, OBJECT, PREDICATE).
3. Use a secret key to select one role, then expand that role's head into a watermarkable span.
4. Regenerate the text at that span's anchor while applying a text watermark.
5. During detection, parse the final text again, reconstruct the **anchor** from the same key, and run the watermark detector on a fixed-length token window from that anchor.

## Core Hypothesis
A watermark does not need to be distributed across the entire text.

If the generator and detector share:
- the same parser,
- the same span extraction policy,
- the same secret seed or key,
- and the same watermark algorithm,

then the detector can reconstruct the intended watermark support from the final text and compute the detection statistic only over the selected span.

This avoids diluting the watermark statistic with unwatermarked tokens outside the selected span.

### Empirical design constraints (learned from v1–v5)

Two constraints, discovered experimentally, now shape the method:

1. **The watermarked region needs entropy.** A short, tightly-constrained
   constituent (e.g. an infilled object phrase) is nearly deterministic, so a
   green-list bias cannot land on it (observed green rates ~0–29%). The
   replacement must be **freely generated** (left-context continuation), which
   restores high entropy (green rates ~75–90%).
2. **The detection unit is an anchor + fixed-K token window, not the
   constituent itself.** After regeneration the constituent's *end* boundary is
   unstable under re-parsing (its start is stable ~84–92%), and constituent
   lengths (~2–5 tokens) are statistically too small anyway. Both sides
   therefore use "K tokens from the reconstructed anchor" (K≈20), which fixes
   N and ignores the unstable end.

The syntactic span thus serves as a **reproducible anchor**, while the
watermark support is the fixed-length window that starts there.


## High-Level Pipeline

### Watermark Embedding

```text
Prompt
  ↓
Generate an unwatermarked draft
  ↓
Parse the draft with 'spaCy + textacy' and locate role heads (SUBJECT / OBJECT / PREDICATE)
  ↓
Use the secret key to select one role, then expand its head into a span
(subject/object -> enclosing noun_chunk; predicate -> verb dependency subtree)
  ↓
Freely regenerate K tokens from that span's anchor with watermarking enabled,
then splice the remaining right context back
  ↓
Verify: re-parse the result; the same key must reconstruct the same anchor
(retry the regeneration if not)
  ↓
Return the final watermarked text
```

### Watermark Detection

```text
Input text
  ↓
Parse the text using the same parser
  ↓
Select the same role using the same key, then reconstruct the span's anchor
  ↓
Take a fixed window of K model tokens starting at the anchor
  ↓
Run the watermark detector only on those tokens
  ↓
Compute the exact binomial p-value and detection decision
```

Formally:

```text
target = select_span(input_text, secret_key)            # keyed role -> anchored span
tokens = window_from_anchor(input_text, target.start, K)  # fixed K model tokens
p = detect_watermark(tokens, secret_key)                # exact binomial tail
is_watermarked = p < p_threshold
```

Note: selection is over a **fixed role set**, not over a mutable candidate list.
A `key % len(candidates)` scheme is avoided on purpose, because the candidate
count and ordering can shift after regeneration and break reconstruction.


## Initial Technical Scope

The first prototype should be intentionally simple.

### Span Parser

Use:

- **spaCy** for tokenization, sentence segmentation, dependency parsing, and character offsets
- **textacy** for initial subject–verb–object or dependency-based structural extraction

The only requirement is that selected spans:

- reflect sentence structure,
- are reproducible from the final text,
- are contiguous character spans,
- can be mapped to model-token indices,
- and contain enough model tokens for watermark detection.

textacy is used only to locate the **role heads** (subject / verb / object).
Raw textacy S/V/O tokens are too short to watermark (empirically ~1-2 model
tokens each), so the head is expanded into a longer contiguous span:

- subject / object head -> the enclosing spaCy `noun_chunk`,
- predicate (verb) -> the verb-centered dependency subtree (VP-like span),
- then apply minimum / maximum model-token length filters.

Whole-triple spans are avoided: they can include unrelated intervening text
or be discontinuous.

The exact span policy should be isolated behind a replaceable interface.


## Main Components

### 1. Draft Generator

Responsibilities:

- Generate an unwatermarked draft from a prompt.
- Return both the generated text and generation metadata when available.

Suggested interface:

```python
class DraftGenerator:
    def generate(self, prompt: str) -> str:
        ...
```

---

### 2. Structural Span Extractor

Responsibilities:

- Parse the input text.
- Extract candidate structural spans.
- Preserve original character offsets.
- Filter malformed, discontinuous, trivial, or unsupported spans.
- Return candidates in a deterministic canonical order.

Suggested data model:

```python
from dataclasses import dataclass

@dataclass(frozen=True)
class StructuralSpan:
    sentence_index: int
    role: str
    start_char: int
    end_char: int
    text: str
    parser_token_start: int
    parser_token_end: int
```

Suggested interface:

```python
class SpanExtractor:
    def extract(self, text: str) -> list[StructuralSpan]:
        ...
```

Initial implementation:

```text
spaCy + textacy
```

Important requirements:

- Use spaCy token and span offsets.
- Do not recover locations with string search such as `str.find()`.
- Preserve `Token.idx`, `Span.start_char`, and `Span.end_char`.
- Convert textacy outputs back to their original spaCy token objects when needed.
- Only expose contiguous spans in the first prototype.
- Keep the extraction policy deterministic and versioned.

---

### 3. Span Eligibility Filter

Not every parser output should be considered a watermark target.

Possible initial filters:

- contiguous span only,
- no punctuation-only spans,
- no full-sentence span,
- minimum model-token length,
- maximum model-token length,
- no overlap with special tokens,
- supported syntactic role only,
- one sentence at a time.

Suggested interface:

```python
class SpanFilter:
    def filter(
        self,
        text: str,
        spans: list[StructuralSpan],
    ) -> list[StructuralSpan]:
        ...
```

For the initial experiment, it is acceptable to skip a sentence when no eligible span exists.

---

### 4. Keyed Role Selector

Selection is over a **fixed role set** (e.g. `SUBJECT`, `OBJECT`, `PREDICATE`),
not over a mutable candidate list. This is the key robustness decision: the
number and ordering of parser candidates can change after regeneration, so a
`key % len(candidates)` index is not reproducible from the final text. A fixed
role, by contrast, is stable as long as regeneration preserves that role.

Responsibilities:

- Derive a role deterministically from the secret key.
- Produce the same role given the same key (independent of text length or candidate count).
- Avoid relying on mutable Python hash values.

Suggested interface:

```python
class RoleSelector:
    def select_role(self, secret_key: bytes) -> str:  # -> "SUBJECT" | "OBJECT" | "PREDICATE"
        ...
```

Two policies, both over the fixed role set:

```text
(a) Global keyed role (v3):
    role = ROLES[PRF(secret_key) % len(ROLES)]
    -> one role for every text; simplest, most stable reconstruction.

(b) Per-span keyed role (v5):
    scan candidate spans in canonical order; for each span,
    target = ROLES[PRF(secret_key, n-gram preceding the span) % len(ROLES)]
    -> the first span whose actual role matches its target is the site.
    The n-gram must come from the LEFT of the span (regeneration never
    touches it), so embedding and detection derive identical targets.
    Roles now vary per position/text, which removes the "attacker learns
    the one fixed role" weakness at a small reconstruction cost.
```

If no eligible span matches in a given text, the sample is skipped
(and this is logged as a reconstruction failure).

The role-selection key and watermark key should be derived separately from a master key.

```text
role_key = KDF(master_key, "role-selection")
watermark_key = KDF(master_key, "watermark")
```

---

### 5. Character-to-Model-Token Mapper

spaCy and the generation model may use different tokenizations.

Character offsets should be the shared coordinate system.

Responsibilities:

- Tokenize the full text with the generation model tokenizer.
- Request token offset mappings.
- Select model tokens that overlap the chosen character span (see boundary rule below).
- Use the exact same boundary rule during embedding and detection.

Suggested interface:

```python
class TokenSpanMapper:
    def map(
        self,
        text: str,
        span: StructuralSpan,
    ) -> list[int]:
        ...
```

Recommended initial boundary policy: **trim-then-overlap**, not strict containment.

Strict containment drops any token whose char interval is not fully inside the
span. With BPE tokenizers (e.g. OPT/GPT-2) the leading space is part of the
token offset (`' model'` = `[31:37]`), so a span starting at the word's first
letter (`32`) loses that token and N is roughly halved. Instead:

```text
For each model token with char offset [a, b):
  1. Trim leading/trailing whitespace from [a, b) -> [ta, tb).
  2. Include the token iff  tb > span_start  and  ta < span_end.
```

This keeps `' model'` (trimmed `[32:37]`) inside a span `[32:45]`. Verified on
`facebook/opt-1.3b`: strict N=1 vs trim-overlap N=2 for the span `'model weights'`.

Special tokens and zero-length offsets must be ignored.

---

### 6. Span Regenerator

Responsibilities:

- Preserve the text before the anchor exactly; splice the right context back after.
- **Freely** generate K replacement tokens from the anchor (left-context
  continuation — constrained infilling kills the entropy the watermark needs;
  the generated window may overrun the original constituent, which is accepted).
- Apply watermarking only while generating those K tokens.
- Verify reconstructability (re-parse; same key must land on the same anchor)
  and retry the generation a few times if it fails.
- Return the full final text.

Suggested interface:

```python
class SpanRegenerator:
    def regenerate(
        self,
        original_text: str,
        target_span: StructuralSpan,
        watermark_key: bytes,
    ) -> str:
        ...
```

The initial prototype does not need to guarantee perfect semantic equivalence. It should log the original and regenerated span for later analysis.

---

### 7. Watermark Embedder

Responsibilities:

- Modify token generation probabilities only for tokens generated inside the selected span.
- Keep the implementation independent from the parser.

Suggested interface:

```python
class WatermarkLogitsProcessor:
    def __call__(self, input_ids, logits):
        ...
```

For a green-list/red-list prototype, configurable parameters may include:

```yaml
gamma: 0.25
delta: 4.0
context_width: 1
```

---

### 8. Detector

Responsibilities:

- Parse the final input text.
- Reconstruct the target span with the same parser and selection policy.
- Map the selected span to model-token indices.
- Compute the watermark statistic only over those tokens.
- Return the z-score, token counts, and decision.

Suggested output:

```python
@dataclass(frozen=True)
class DetectionResult:
    is_watermarked: bool
    z_score: float
    threshold: float
    num_tested_tokens: int
    num_green_tokens: int
    selected_span: StructuralSpan | None
```

For a simple green-list detector:

\[
z =
\frac{G - \gamma N}
{\sqrt{N\gamma(1-\gamma)}}
\]

where:

- \(N\) is the number of model tokens in the detection window (fixed K),
- \(G\) is the number of green-list tokens,
- \(\gamma\) is the expected green-list probability under the null hypothesis.

**Decision rule:** because span-scale N is small, the normal-approximation
z-score is only reported descriptively; the decision uses the **exact one-sided
binomial tail** \(p = P(\mathrm{Binom}(N,\gamma) \ge G)\) with a p-threshold
(e.g. 0.01). A lower \(\gamma\) (0.25 rather than 0.5) packs more evidence per
green token: \(z_{max} = \sqrt{N}\cdot\sqrt{(1-\gamma)/\gamma}\).

The detector must not include tokens outside the reconstructed window.

---

## Determinism Requirements

Embedding and detection must use identical versions of:

- spaCy,
- the spaCy language model,
- textacy,
- the span extraction policy,
- the candidate filtering policy,
- the candidate sorting policy,
- the model tokenizer,
- Unicode normalization,
- the key derivation method,
- and the watermark detector.

Record these values in experiment metadata.

Suggested metadata:

```json
{
  "spacy_version": "...",
  "spacy_model": "...",
  "spacy_model_version": "...",
  "textacy_version": "...",
  "span_policy_version": "v0",
  "tokenizer_name": "...",
  "tokenizer_revision": "...",
  "watermark_type": "kgw",
  "watermark_config": {
    "gamma": 0.25,
    "delta": 4.0
  }
}
```

---

## Initial Experiment

The first experiment should answer one question:

> Can a parser-derived structural span be selected, watermarked through local regeneration, reconstructed from the final text, and detected using only the reconstructed span?

### Minimal Dataset

Use a small collection of English prompts that produce:

- simple declarative sentences,
- one or more noun phrases,
- clear dependency structures,
- and sufficiently long candidate spans.

Start with tens or hundreds of examples, not a large benchmark.

### Conditions

At minimum, compare:

1. **Unwatermarked**
   - Generate the draft.
   - Do not regenerate or watermark any span.

2. **Structural-span watermark**
   - Generate the draft.
   - Select a parser-derived span.
   - Regenerate only the span with watermarking enabled.
   - Detect only over the reconstructed span.

Optional baselines:

3. **Random span**
   - Select a random contiguous span with the same approximate token length.

4. **Whole-text watermark**
   - Apply the same watermark over the whole generated text.

### Metrics

Primary metrics:

- detection true positive rate,
- false positive rate,
- z-score,
- number of tested tokens,
- selected span length,
- percentage of text regenerated,
- parser success rate,
- detector span reconstruction success rate.

Secondary metrics:

- semantic similarity between draft and final text,
- edit distance,
- perplexity or fluency score,
- generation latency,
- parse latency,
- regeneration latency.



## Recommended Implementation Order

### Phase 1: Parsing and Offset Validation

- Load spaCy and textacy.
- Extract structural span candidates.
- Verify character offsets against the original text.
- Verify that all returned spans are contiguous.
- Add unit tests for repeated words and identical phrases.

### Phase 2: Token Alignment

- Map spaCy character spans to generation-model token indices.
- Test subword boundaries, punctuation, whitespace, and Unicode text.
- Ensure embedding and detection use the same mapping rule.

### Phase 3: Local Regeneration

- Mask or mark one selected span.
- Regenerate only that span.
- Confirm that text outside the span remains unchanged.

### Phase 4: Watermark Integration

- Add a simple token-level watermark during span regeneration.
- Implement the corresponding z-test detector.

### Phase 5: End-to-End Experiment

- Run generation, parsing, selection, regeneration, reconstruction, and detection.
- Save detailed traces.
- Measure detection performance and failure cases.

---

## Unit Tests Required for the Prototype

At minimum, test:

- repeated identical phrases in one sentence,
- repeated words,
- subject and object spans with the same surface text,
- contractions,
- punctuation adjacent to the selected span,
- multi-token noun phrases,
- subword tokenization,
- empty candidate sets,
- one-token spans,
- discontinuous parser outputs,
- parser/model version determinism,
- identical selection from identical text and key,
- different selection behavior under different keys.

Do not use string search to recover span positions.

---

## Current Assumptions

- The detector knows the parser configuration and secret key.
- The detector does not receive the embedding-time span directly.
- The final text is parsed independently at detection time.
- The first experiment does not focus on adversarial paraphrasing.
- The first experiment may skip unsupported sentences.
- The first experiment uses one selected span per sentence or text sample.
- The initial watermark detector uses a z-test over selected model tokens.
- spaCy and textacy are provisional choices and may be replaced later.

---

## Non-Goals for the First Prototype

The initial implementation does not need to:

- support all sentence structures,
- implement a full five-pattern English grammar classifier,
- support multiple languages,
- resist sentence insertion or deletion attacks,
- solve parser synchronization under aggressive rewriting,
- optimize the best syntactic span type,
- prove theoretical optimality,
- or implement multiple watermark algorithms.

The first target is a reliable end-to-end proof of concept.

---

## Success Criteria

The prototype is successful when it can:

1. Generate an unwatermarked draft.
2. Extract at least one valid structural span.
3. Select the span deterministically from a secret key.
4. Regenerate only the selected span.
5. Apply a watermark only during that regeneration.
6. Parse the final text and reconstruct the target span.
7. Map the reconstructed span to model-token indices.
8. Compute a span-only z-score.
9. Separate watermarked and unwatermarked samples better than chance.
10. Produce enough debug metadata to explain failures.

---

## Research Direction After the Prototype

Possible follow-up studies include:

- constituency spans versus dependency spans,
- noun chunks versus full dependency subtrees,
- syntactic spans versus random spans,
- different span-length constraints,
- multiple spans per document,
- adaptive span eligibility,
- robustness to paraphrasing,
- parser disagreement,
- multilingual parsing,
- diffusion-based iterative span infilling,
- and compatibility with different watermark algorithms.

---

## One-Sentence Summary

> Generate first, identify a reproducible syntactic anchor, freely regenerate a fixed-length token window there under a watermark constraint, and detect the watermark by reconstructing the same anchor and testing the same window.
