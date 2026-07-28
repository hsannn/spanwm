# ==============================================================
# spanwm_v8.py
# SpanWM v8 = v7 (multi-span + splice fixes) with ONE change:
#
#   THE ANCHOR PRF SELECTS *TWO* ROLES, NOT ONE
#   v5-v7: target = PRF(master_key, n-gram before span) -> ONE role;
#          a span is a site only if sp.role == target        (1 of 3 -> ~1/3
#          of the parsed spans are role-eligible)
#   v8   : target = PRF(...) -> a k-subset of the role list (k =
#          `roles_per_anchor`, default 2); a span is a site if
#          sp.role in target                                 (2 of 3 -> ~2/3)
#
#   Why: with only one admissible role per position the left->right scan often
#   walks past long stretches of text before a span of the right role appears,
#   so a draft frequently yields fewer than max_spans sites (short N -> weak
#   pooled test). Admitting two roles roughly doubles the density of eligible
#   anchors, so max_spans sites are found earlier and in more drafts.
#
#   The key invariants are untouched:
#     * the PRF input is still the n-gram to the LEFT of the anchor, which
#       left-AR regeneration never rewrites -> embed and detect derive the same
#       target set from the same position (v5 invariant);
#     * the subset is drawn from a FIXED, ordered role list via
#       itertools.combinations, so it is a deterministic function of the key
#       and the n-gram only -- never of a mutable candidate list;
#     * embed and detect share the same scan (`scan_sites`), so the detector
#       still reconstructs the same anchors with no embed-time information.
#
#   Keying is unchanged from v5 (same domain-separation string), so
#   `roles_per_anchor = 1` reproduces v7's selection byte-for-byte:
#   combinations(roles, 1) is [(roles[0],), (roles[1],), ...], i.e. exactly the
#   `idx % len(roles)` rule.
#
#   Detection cost: a larger admissible set means an unwatermarked text also
#   has more role-eligible anchors, so the negative class reconstructs sites
#   more often too -- but its window tokens are still green with probability
#   gamma, so the pooled null is unchanged. What changes is only how many
#   tokens each class contributes.
#
# Splicing (drift-free left boundary, punctuation-aware right splice) and the
# pooled exact-binomial detector are inherited from v7/v6 unchanged.
# ==============================================================

import hashlib
from itertools import combinations

from watermark.spanwm_v7.spanwm_v7 import SpanWMV7, SpanWMV7Config, SpanWMV7Utils
from watermark.kgw.kgw import KGWLogitsProcessor
from utils.transformers_config import TransformersConfig
from watermark.spanwm.span_ops import StructuralSpan


class SpanWMV8Config(SpanWMV7Config):
    """v7 config + how many roles the anchor PRF admits."""

    def initialize_parameters(self) -> None:
        super().initialize_parameters()
        k = self.config_dict.get('roles_per_anchor', 2)
        # clamp: at least one role, at most the whole (fixed) role list
        self.roles_per_anchor = max(1, min(int(k), len(self.roles)))

    @property
    def algorithm_name(self) -> str:
        return 'SpanWM_v8'


class SpanWMV8Utils(SpanWMV7Utils):
    """v7 utils with the single-role PRF replaced by a k-subset PRF."""

    def roles_for_anchor(self, text: str, anchor_char: int) -> tuple[str, ...]:
        """Admissible roles at this position: PRF(master_key, preceding n-gram)
        indexes the fixed list of k-subsets of `roles`."""
        n = self.config.role_ngram_chars
        ngram = text[max(0, anchor_char - n):anchor_char].encode("utf-8")
        digest = hashlib.sha256(
            self.config.master_key + b"role-per-span" + ngram).digest()
        subsets = list(combinations(self.config.roles, self.config.roles_per_anchor))
        idx = int.from_bytes(digest[:8], "big") % len(subsets)
        return subsets[idx]

    def role_for_anchor(self, text: str, anchor_char: int) -> str:
        """Back-compat single-role view (first role of the admitted subset).
        Selection itself uses `roles_for_anchor`; this is for probes/logging."""
        return self.roles_for_anchor(text, anchor_char)[0]

    def select_span(self, text: str) -> StructuralSpan | None:
        """First span (canonical order) whose role is ADMITTED at its own
        anchor and which passes the model-token length filter."""
        for sp in self.extractor.extract(text):        # canonically sorted
            if sp.role not in self.roles_for_anchor(text, sp.start_char):
                continue
            n = self.mapper.count(text, sp)
            if self.config.min_span_tokens <= n <= self.config.max_span_tokens:
                return sp
        return None

    def select_span_from(self, text: str, from_char: int) -> StructuralSpan | None:
        """Same rule, bounded to anchors at/after from_char (multi-span scan)."""
        for sp in self.extractor.extract(text):        # canonically sorted
            if sp.start_char < from_char:
                continue
            if sp.role not in self.roles_for_anchor(text, sp.start_char):
                continue
            n = self.mapper.count(text, sp)
            if self.config.min_span_tokens <= n <= self.config.max_span_tokens:
                return sp
        return None


class SpanWMV8(SpanWMV7):
    """v7 pipeline with a two-role (k-subset) anchor PRF."""

    def __init__(self, algorithm_config, transformers_config: TransformersConfig | None = None,
                 *args, **kwargs) -> None:
        if isinstance(algorithm_config, str):
            self.config = SpanWMV8Config(algorithm_config, transformers_config)
        elif isinstance(algorithm_config, SpanWMV8Config):
            self.config = algorithm_config
        else:
            raise TypeError("algorithm_config must be a path string or a SpanWMV8Config instance")
        self.utils = SpanWMV8Utils(self.config)
        self.logits_processor = KGWLogitsProcessor(self.config, self.utils.kgw)
