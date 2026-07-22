# ==============================================================
# span_ops.py
# Description: Syntactic-span operations for SpanWM.
#              - StructuralSpan data model
#              - SpanExtractor   (spaCy + textacy -> role spans)
#              - RoleSelector    (secret key -> one fixed role)
#              - TokenSpanMapper (char span -> model-token indices)
#
# Design decisions (README v0):
#   * Selection is over a FIXED role set, not a mutable candidate list.
#   * textacy locates S/V/O role heads; the head is EXPANDED into a
#     longer contiguous span (subject/object -> enclosing noun_chunk,
#     predicate -> contiguous verb group). Raw S/V/O tokens are too short.
#   * Char offsets are the shared coordinate system. Never str.find().
#   * Token mapping uses trim-then-overlap, not strict containment,
#     so BPE leading-space tokens (e.g. ' model') are not dropped.
# ==============================================================

from __future__ import annotations

import hashlib
from dataclasses import dataclass

import spacy
from textacy import extract as textacy_extract

# ---- role -> spaCy dependency labels of the head token --------------------
SUBJECT_DEPS = {"nsubj", "nsubjpass", "csubj", "csubjpass"}
OBJECT_DEPS = {"dobj", "obj", "dative", "attr", "oprd", "pobj"}
# tokens that may be glued onto a verb to form a contiguous predicate span
PREDICATE_CHILD_DEPS = {"aux", "auxpass", "neg", "prt", "advmod"}

ROLE_SUBJECT = "SUBJECT"
ROLE_OBJECT = "OBJECT"
ROLE_PREDICATE = "PREDICATE"


@dataclass(frozen=True)
class StructuralSpan:
    """A reproducible, contiguous syntactic span with character offsets."""
    sentence_index: int
    role: str
    start_char: int
    end_char: int
    text: str
    parser_token_start: int  # spaCy token index (inclusive)
    parser_token_end: int    # spaCy token index (exclusive)


def _trim(a: int, b: int, text: str) -> tuple[int, int]:
    """Strip leading/trailing whitespace from a char interval."""
    while a < b and text[a].isspace():
        a += 1
    while b > a and text[b - 1].isspace():
        b -= 1
    return a, b


def _noun_chunk_for(token, chunks) -> object | None:
    """Return the noun_chunk (spaCy Span) that contains a head token."""
    for c in chunks:
        if c.start <= token.i < c.end:
            return c
    return None


def _contiguous_run(tokens) -> list | None:
    """If the spaCy token indices are contiguous, return them sorted; else None."""
    toks = sorted(tokens, key=lambda t: t.i)
    idxs = [t.i for t in toks]
    if idxs == list(range(idxs[0], idxs[-1] + 1)):
        return toks
    return None


class SpanExtractor:
    """Parse text with spaCy + textacy and emit role-tagged structural spans."""

    def __init__(self, spacy_model: str = "en_core_web_sm") -> None:
        self.spacy_model = spacy_model
        self.nlp = spacy.load(spacy_model)

    def extract(self, text: str) -> list[StructuralSpan]:
        doc = self.nlp(text)
        # map each sentence start-char to a sentence index for metadata
        sent_bounds = [(s.start_char, s.end_char, i) for i, s in enumerate(doc.sents)]

        def sent_index(char_pos: int) -> int:
            for a, b, i in sent_bounds:
                if a <= char_pos < b:
                    return i
            return -1

        chunks = list(doc.noun_chunks)
        raw: list[tuple[str, int, int, int, int]] = []  # role, s, e, tok_start, tok_end

        # --- 1) textacy S/V/O triples give us the role heads ---------------
        for tri in textacy_extract.subject_verb_object_triples(doc):
            subj_head = self._head(tri.subject, SUBJECT_DEPS)
            obj_head = self._head(tri.object, OBJECT_DEPS)
            if subj_head is not None:
                raw.append(self._expand_noun(ROLE_SUBJECT, subj_head, chunks))
            if obj_head is not None:
                raw.append(self._expand_noun(ROLE_OBJECT, obj_head, chunks))
            pred = self._expand_predicate(list(tri.verb))
            if pred is not None:
                raw.append(pred)

        # --- 2) supplement from spaCy dep labels (covers intransitive /
        #        prepositional-object cases textacy misses) -----------------
        for tok in doc:
            if tok.dep_ in SUBJECT_DEPS:
                raw.append(self._expand_noun(ROLE_SUBJECT, tok, chunks))
            elif tok.dep_ in OBJECT_DEPS:
                raw.append(self._expand_noun(ROLE_OBJECT, tok, chunks))
            elif tok.pos_ in ("VERB",) and tok.dep_ in ("ROOT", "conj", "ccomp", "advcl", "relcl"):
                pred = self._expand_predicate([tok] + [c for c in tok.children
                                                       if c.dep_ in PREDICATE_CHILD_DEPS])
                if pred is not None:
                    raw.append(pred)

        # --- 3) dedupe by (role, char span), drop degenerate spans ---------
        seen: set[tuple] = set()
        spans: list[StructuralSpan] = []
        for role, s, e, ts, te in raw:
            if s is None or e <= s:
                continue
            key = (role, s, e)
            if key in seen:
                continue
            seen.add(key)
            span_text = text[s:e]
            if not span_text.strip():
                continue
            spans.append(StructuralSpan(
                sentence_index=sent_index(s),
                role=role,
                start_char=s,
                end_char=e,
                text=span_text,
                parser_token_start=ts,
                parser_token_end=te,
            ))

        # canonical order: by start, then role, then end (deterministic)
        spans.sort(key=lambda sp: (sp.start_char, sp.role, sp.end_char))
        return spans

    # -- helpers -----------------------------------------------------------
    @staticmethod
    def _head(token_span, dep_set):
        """Pick the head token of a textacy subject/object token list."""
        toks = list(token_span)
        if not toks:
            return None
        for t in toks:
            if t.dep_ in dep_set:
                return t
        return toks[-1]  # fallback: rightmost (usually the head noun)

    @staticmethod
    def _expand_noun(role, head, chunks):
        """Expand a subject/object head into its enclosing noun_chunk."""
        c = _noun_chunk_for(head, chunks)
        if c is not None:
            return (role, c.start_char, c.end_char, c.start, c.end)
        # pronoun / no chunk -> use the head token itself
        return (role, head.idx, head.idx + len(head), head.i, head.i + 1)

    @staticmethod
    def _expand_predicate(verb_tokens):
        """Predicate span = the contiguous verb group (verb + aux/neg/prt)."""
        run = _contiguous_run(verb_tokens)
        if run is None:
            return None
        s = run[0].idx
        e = run[-1].idx + len(run[-1])
        return (ROLE_PREDICATE, s, e, run[0].i, run[-1].i + 1)


class RoleSelector:
    """Deterministically map a secret key to one fixed role."""

    def __init__(self, roles: list[str]) -> None:
        if not roles:
            raise ValueError("roles must be non-empty")
        self.roles = list(roles)

    def select_role(self, master_key: bytes) -> str:
        # role_key = KDF(master_key, "role-selection"); avoid Python hash()
        digest = hashlib.sha256(master_key + b"role-selection").digest()
        idx = int.from_bytes(digest[:8], "big") % len(self.roles)
        return self.roles[idx]


class TokenSpanMapper:
    """Map a character span to model-token indices via trim-then-overlap."""

    def __init__(self, tokenizer) -> None:
        self.tokenizer = tokenizer

    def encode_with_offsets(self, text: str):
        enc = self.tokenizer(
            text,
            return_offsets_mapping=True,
            add_special_tokens=False,
        )
        return enc["input_ids"], enc["offset_mapping"]

    def map_positions(self, text: str, span: StructuralSpan) -> tuple[list[int], list[int]]:
        """Return (token positions inside the span, full input_ids)."""
        input_ids, offsets = self.encode_with_offsets(text)
        positions: list[int] = []
        for i, (a, b) in enumerate(offsets):
            if a == b:  # zero-length / special token
                continue
            ta, tb = _trim(a, b, text)
            if ta >= tb:
                continue
            # trim-then-overlap: include if the trimmed char interval overlaps span
            if tb > span.start_char and ta < span.end_char:
                positions.append(i)
        return positions, input_ids

    def count(self, text: str, span: StructuralSpan) -> int:
        return len(self.map_positions(text, span)[0])

    def window_positions(self, text: str, anchor_char: int, k: int) -> tuple[list[int], list[int]]:
        """Return up to k token positions starting at the first real token that
        reaches `anchor_char`, plus the full input_ids. Used by the fixed-K
        detection window: tests K tokens from the span anchor regardless of the
        (unstable) constituent end."""
        input_ids, offsets = self.encode_with_offsets(text)
        i0 = None
        for i, (a, b) in enumerate(offsets):
            if a == b:
                continue
            ta, tb = _trim(a, b, text)
            if ta >= tb:
                continue
            if tb > anchor_char:               # first token that overlaps/at the anchor
                i0 = i
                break
        if i0 is None:
            return [], input_ids
        positions = [i for i in range(i0, min(i0 + k, len(input_ids)))]
        return positions, input_ids
