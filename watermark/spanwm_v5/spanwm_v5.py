# ==============================================================
# spanwm_v5.py
# SpanWM v5 = the v3 working recipe (base model + left-AR free generation +
# fixed K-token window + gamma=0.25 + exact binomial p) with ONE change:
#
#   PER-SPAN KEYED ROLE SELECTION
#   v3: role = PRF(master_key)                      -> same role for every text
#   v5: role = PRF(master_key, n-gram before span)  -> varies per position/text
#
# Selection rule (identical at embed and detect):
#   scan candidate spans in canonical order; for each span, derive its target
#   role from the n-gram of characters IMMEDIATELY BEFORE its anchor; the first
#   span whose actual role matches its target role (and passes the length
#   filter) is the watermark site.
#
# Why the n-gram comes from the LEFT of the span (critical invariant):
#   left-AR regeneration replaces text from the anchor onward and never touches
#   what precedes it, so the PRF input is byte-identical in the draft and in
#   the final text -> embed and detect derive the same target roles.
#
# Why a CHARACTER n-gram (not token ids):
#   BPE merges can cross the anchor boundary (the token covering the anchor
#   may merge differently once the regenerated text follows it), so boundary
#   token ids are not guaranteed stable. The last `role_ngram_chars` characters
#   of text[:anchor] are stable by construction.
#
# Everything else (generation, fixed-K detection window, verification loop,
# scoring) is inherited from the v3 SpanWM unchanged.
# ==============================================================

import hashlib

from watermark.spanwm.spanwm import SpanWM, SpanWMConfig, SpanWMUtils
from watermark.kgw.kgw import KGWLogitsProcessor
from utils.transformers_config import TransformersConfig
from watermark.spanwm.span_ops import StructuralSpan


class SpanWMV5Config(SpanWMConfig):
    """v3 config + per-span role-PRF parameters."""

    def initialize_parameters(self) -> None:
        super().initialize_parameters()
        # how many characters before the span anchor feed the role PRF
        self.role_ngram_chars = self.config_dict.get('role_ngram_chars', 16)

    @property
    def algorithm_name(self) -> str:
        return 'SpanWM_v5'


class SpanWMV5Utils(SpanWMUtils):
    """v3 utils with select_span replaced by per-span keyed role matching."""

    def role_for_anchor(self, text: str, anchor_char: int) -> str:
        """Target role at this position: PRF(master_key, preceding n-gram)."""
        n = self.config.role_ngram_chars
        ngram = text[max(0, anchor_char - n):anchor_char].encode("utf-8")
        digest = hashlib.sha256(
            self.config.master_key + b"role-per-span" + ngram).digest()
        idx = int.from_bytes(digest[:8], "big") % len(self.config.roles)
        return self.config.roles[idx]

    def select_span(self, text: str) -> StructuralSpan | None:
        """First span (canonical order) whose actual role matches the target
        role derived from its own preceding n-gram, and which passes the
        model-token length filter. Reproducible from the final text because
        the n-gram lies entirely in the untouched left context."""
        for sp in self.extractor.extract(text):        # canonically sorted
            if sp.role != self.role_for_anchor(text, sp.start_char):
                continue
            n = self.mapper.count(text, sp)
            if self.config.min_span_tokens <= n <= self.config.max_span_tokens:
                return sp
        return None


class SpanWMV5(SpanWM):
    """v3 pipeline with per-span keyed role selection."""

    def __init__(self, algorithm_config, transformers_config: TransformersConfig | None = None,
                 *args, **kwargs) -> None:
        if isinstance(algorithm_config, str):
            self.config = SpanWMV5Config(algorithm_config, transformers_config)
        elif isinstance(algorithm_config, SpanWMV5Config):
            self.config = algorithm_config
        else:
            raise TypeError("algorithm_config must be a path string or a SpanWMV5Config instance")
        self.utils = SpanWMV5Utils(self.config)
        self.logits_processor = KGWLogitsProcessor(self.config, self.utils.kgw)

    # generation (left-AR + fixed-K), verification (anchor distance),
    # detection (fixed-K window from the reconstructed anchor) are all
    # inherited from SpanWM; they call self.utils.select_span, which is the
    # v5 version above at both embed and detect time.
