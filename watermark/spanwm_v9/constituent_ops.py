# ==============================================================
# constituent_ops.py
# ConstituentExtractor for SpanWM v9: benepar (Berkeley Neural Parser) as a
# spaCy pipeline component -> constituency-labeled StructuralSpans.
#
# Why benepar-via-spaCy and not constituent-treelib: treelib's phrase
# extraction returns strings without char offsets (would force str.find(),
# banned); the spaCy component exposes constituents as spacy Spans, so
# start_char/end_char come from the same offset coordinate system every other
# SpanWM version uses.
#
# The "role" of a span is its constituency label (e.g. NP/VP/PP), so the
# v5 per-span PRF role rule and the v6 scan work unchanged on top of this
# extractor — only the notion of what a span IS changes.
# ==============================================================

from collections import OrderedDict

import spacy

# --- transformers 5.x compat shim (must precede benepar model loading) -----
# transformers 5 removed build_inputs_with_special_tokens from the tokenizer
# base classes; benepar 0.2.0 still calls it on its T5 tokenizer during
# Retokenizer.__init__. T5's rule is `ids + [eos]`.
from transformers.tokenization_utils_base import PreTrainedTokenizerBase

if not hasattr(PreTrainedTokenizerBase, "build_inputs_with_special_tokens"):
    def _build_inputs_with_special_tokens(self, token_ids_0, token_ids_1=None):
        if token_ids_1 is not None:
            raise NotImplementedError("pair inputs not supported by the shim")
        if "T5" in type(self).__name__:
            return list(token_ids_0) + [self.eos_token_id]
        raise NotImplementedError(
            f"no build_inputs_with_special_tokens shim for {type(self).__name__}")
    PreTrainedTokenizerBase.build_inputs_with_special_tokens = \
        _build_inputs_with_special_tokens

import benepar  # noqa: E402  (import after the shim on purpose)

from watermark.spanwm.span_ops import StructuralSpan


def _collapse_whitespace(text: str) -> tuple[str, list[int]]:
    """Collapse every whitespace run to a single ' ' (and drop leading runs).

    benepar's T5 retokenizer asserts on whitespace-only spaCy tokens, which
    '\\n', '\\t' and double spaces all produce — and C4 texts / model drafts
    contain them routinely. Parsing a collapsed copy avoids that; `idx_map`
    maps each collapsed-text index back to its ORIGINAL index so constituent
    offsets stay in the original coordinate system (no str.find()).
    """
    out: list[str] = []
    idx_map: list[int] = []
    in_space = True                      # True at start: drop leading whitespace
    for i, ch in enumerate(text):
        if ch.isspace():
            if in_space:
                continue
            out.append(' ')
            idx_map.append(i)
            in_space = True
        else:
            out.append(ch)
            idx_map.append(i)
            in_space = False
    return ''.join(out), idx_map


class ConstituentExtractor:
    """Parse with spaCy+benepar and emit constituent spans as StructuralSpans.

    role = the first constituency label of the node that is in `labels`
    (a unary chain like ('S','VP') can carry several labels). Spans NEST —
    an NP inside a VP is emitted alongside it; the caller's scan/dedupe
    handles overlap exactly as it did for dep-based spans.
    """

    def __init__(self, spacy_model: str, benepar_model: str, labels: list[str],
                 cache_size: int = 32) -> None:
        self.nlp = spacy.load(spacy_model)
        self.nlp.add_pipe("benepar", config={"model": benepar_model})
        self.labels = set(labels)
        self._cache: OrderedDict[str, list[StructuralSpan]] = OrderedDict()
        self._cache_size = cache_size
        self.n_parse_failures = 0

    def extract(self, text: str) -> list[StructuralSpan]:
        if text in self._cache:
            self._cache.move_to_end(text)
            return self._cache[text]
        spans = self._extract_uncached(text)
        self._cache[text] = spans
        if len(self._cache) > self._cache_size:
            self._cache.popitem(last=False)
        return spans

    def _extract_uncached(self, text: str) -> list[StructuralSpan]:
        collapsed, idx_map = _collapse_whitespace(text)
        if not collapsed:
            return []
        try:
            doc = self.nlp(collapsed)
        except Exception:
            # benepar raises on sentences longer than its subword limit (512);
            # treat the text as having no reconstructable sites
            self.n_parse_failures += 1
            return []

        seen: set[tuple] = set()
        spans: list[StructuralSpan] = []
        for si, sent in enumerate(doc.sents):
            for c in sent._.constituents:
                role = next((lb for lb in c._.labels if lb in self.labels), None)
                if role is None:
                    continue
                if c.end_char <= c.start_char or not collapsed[c.start_char:c.end_char].strip():
                    continue
                # map collapsed offsets back to the original text; constituents
                # start and end on non-space chars, so both endpoints are exact
                s = idx_map[c.start_char]
                e = idx_map[c.end_char - 1] + 1
                key = (role, s, e)
                if key in seen:
                    continue
                seen.add(key)
                spans.append(StructuralSpan(
                    sentence_index=si, role=role,
                    start_char=s, end_char=e, text=text[s:e],
                    parser_token_start=c.start, parser_token_end=c.end,
                ))
        # same canonical order as SpanExtractor (deterministic scan)
        spans.sort(key=lambda sp: (sp.start_char, sp.role, sp.end_char))
        return spans
