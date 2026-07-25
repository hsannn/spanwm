# ==============================================================
# spanwm_v7.py
# SpanWM v7 = v6 (multi-span, pooled exact binomial) + two SPLICE FIXES.
# Everything else (site scan, per-site verify, pooled detection) is inherited
# from v6 unchanged — detection is byte-for-byte the v6 detector.
#
# Fix 1 — anchor-boundary retokenization drift (detection power).
#   v6 generates the fill from left = text[:anchor], which almost always ends
#   with the space BEFORE the span. That dangling " " is its own token (220)
#   at generation, but after splicing, retokenization merges it into the
#   fill's first word (" dog"), so the detector sees a different token id AND
#   a different greenlist seed for the next token: the first 1-2 tokens of
#   EVERY window are lost. At K_s=6 that is up to 1/3 of all tested tokens
#   (v6 measured green ~67% vs v5 ~78%; z 4.72 vs 5.34 — the gap matches).
#   v7 strips the trailing space and lets the model emit the leading-space
#   token itself, so the spliced text retokenizes into exactly the generated
#   ids and the detector recomputes the same greenlist seeds as generation.
#   If the fill does not start with a space (rare), fall back to the v6
#   splice — the text up to the anchor must stay byte-identical for the role
#   PRF and the detector scan.
#
# Fix 2 — no space before punctuation (stealth / quality).
#   v6's sep_r inserted " " whenever the right context did not start with a
#   space — including before "," "." etc., stamping a ' ,' fingerprint into
#   133/200 watermarked outputs (vs 1/200 unwatermarked): a regex-detectable
#   tell and a quality artifact. v7 attaches punctuation directly.
# ==============================================================

from watermark.spanwm_v6.spanwm_v6 import SpanWMV6, SpanWMV6Config, SpanWMV6Utils
from watermark.kgw.kgw import KGWLogitsProcessor
from utils.transformers_config import TransformersConfig


class SpanWMV7Config(SpanWMV6Config):
    """v6 config, new algorithm name only."""

    @property
    def algorithm_name(self) -> str:
        return 'SpanWM_v7'


class SpanWMV7Utils(SpanWMV6Utils):
    """Unchanged from v6 (site scan / window / scoring)."""


class SpanWMV7(SpanWMV6):
    """v6 pipeline with drift-free, punctuation-aware splicing."""

    # chars that attach directly to the preceding word: never insert a space
    # before them when splicing the right context back
    _NO_SEP_BEFORE = set(",.;:!?)]}»’”'\"%…—–-")
    # chars a fill must not OPEN with after a restored space ('word , ...');
    # quotes/dashes excluded — they can legitimately open after a space
    _NO_OPEN = set(",.;:!?)]}»’”%…")

    def __init__(self, algorithm_config, transformers_config: TransformersConfig | None = None,
                 *args, **kwargs) -> None:
        if isinstance(algorithm_config, str):
            self.config = SpanWMV7Config(algorithm_config, transformers_config)
        elif isinstance(algorithm_config, SpanWMV7Config):
            self.config = algorithm_config
        else:
            raise TypeError("algorithm_config must be a path string or a SpanWMV7Config instance")
        self.utils = SpanWMV7Utils(self.config)
        self.logits_processor = KGWLogitsProcessor(self.config, self.utils.kgw)

    # -- embedding ---------------------------------------------------------
    def _regen_at(self, text: str, anchor_char: int, span_end_char: int):
        """Regenerate K_s watermarked tokens at anchor; splice the rest back.
        Returns (new_text, fill_len). See module docstring for the two fixes."""
        left = text[:anchor_char]
        right = text[span_end_char:]
        k = self.config.span_window_tokens
        strip_left = left.endswith(" ")
        ctx = left[:-1] if strip_left else left
        for _ in range(3):
            fill, _ = self._raw_generate(ctx, watermark=True, max_new_tokens=k, min_new_tokens=k)
            # resample a fill that opens with punctuation — it would splice as
            # 'word , ...' once the stripped space is restored
            if fill.lstrip(" ")[:1] not in self._NO_OPEN:
                break
        # prefix must reproduce text[:anchor] exactly; a space-opening fill
        # supplies the stripped space itself, otherwise restore it
        if strip_left and fill.lstrip(" ")[:1] in self._NO_OPEN:
            # the model insists on a punctuation opener (e.g. ',' after a
            # name): attach it directly — 'Bharti, which' reads naturally but
            # the pre-anchor byte changes, so this site is sacrificed (the
            # detector scan walks past it, same as any unverified site)
            fill = fill.lstrip(" ")
            prefix = ctx
        elif strip_left and fill.startswith(" "):
            prefix = ctx
        else:
            prefix = left
        # a trailing-space fill would splice as 'word  word' or 'word ,' when
        # the right context brings its own boundary — drop the dangling space
        if fill.endswith(" ") and right and (right[0].isspace() or right[0] in self._NO_SEP_BEFORE):
            fill = fill.rstrip(" ") or fill
        body = prefix + fill
        sep_r = "" if (not right or right[0].isspace() or right[0] in self._NO_SEP_BEFORE
                       or (fill and fill[-1].isspace())) else " "
        return body + sep_r + right, len(body) - anchor_char
