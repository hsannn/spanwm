# ==============================================================
# spanwm_v9.py
# SpanWM v9 = v7 (multi-span + splice fixes) with the SPAN UNIT changed from
# "fixed K_s-token window at a dep-parse anchor" to "a CONSTITUENT from a
# constituency parse (benepar)". The v5 per-span PRF role rule and the v6
# left->right site scan / pooled exact binomial test are kept as-is; roles
# are now constituency labels (NP/VP/PP).
#
# What changes vs v7 (each piece small, everything else inherited):
#   1. extractor      = ConstituentExtractor (benepar via spaCy; spans carry
#                        char offsets from spaCy, never str.find()).
#   2. embed length   = _regen_at generates as many watermarked tokens as the
#                        ORIGINAL constituent had (variable per site), not a
#                        fixed span_window_tokens.
#   3. detect window  = the RECONSTRUCTED constituent's own token extent
#                        (map_positions), not K_s tokens from the anchor.
#   4. scan advance   = past the constituent's end_char (embed and detect use
#                        the same rule: window_end_char / scan_sites below).
#
# The bet being tested: benepar constituent boundaries survive regeneration
# well enough that the extent-based window (the v1 failure with the dep
# parser) becomes viable — buying naturally-scoped watermark regions instead
# of windows that overflow the constituent (the standing quality issue).
# Failure mode to watch: reconstructed constituent longer/shorter than the
# fill -> diluted or truncated N (embed info records fill vs recon extents).
# ==============================================================

from watermark.spanwm_v7.spanwm_v7 import SpanWMV7, SpanWMV7Config, SpanWMV7Utils
from watermark.kgw.kgw import KGWLogitsProcessor
from utils.transformers_config import TransformersConfig
from watermark.spanwm.span_ops import StructuralSpan
from watermark.spanwm_v9.constituent_ops import ConstituentExtractor


class SpanWMV9Config(SpanWMV7Config):
    """v7 config + constituency-parser parameters (roles = constituent labels)."""

    def initialize_parameters(self) -> None:
        super().initialize_parameters()
        self.benepar_model = self.config_dict.get('benepar_model', 'benepar_en3')

    @property
    def algorithm_name(self) -> str:
        return 'SpanWM_v9'


class SpanWMV9Utils(SpanWMV7Utils):
    """v7 utils with the extractor swapped for benepar constituents and the
    scan advancing past the constituent end instead of a fixed window."""

    def __init__(self, config: SpanWMV9Config) -> None:
        super().__init__(config)
        self.extractor = ConstituentExtractor(
            config.spacy_model, config.benepar_model, config.roles)

    def span_token_count(self, text: str, start_char: int, end_char: int) -> int:
        """Model-token length of a char range (trim-then-overlap, as scoring)."""
        probe = StructuralSpan(-1, "_RANGE", start_char, end_char,
                               text[start_char:end_char], -1, -1)
        return len(self.mapper.map_positions(text, probe)[0])

    def window_end_char(self, text: str, anchor_char: int) -> int:
        """v9: a site's extent is its constituent. Re-find the site from just
        left of the anchor (the embed loop verifies alignment within ±4 chars)
        and advance past its end. Parses are cached, so this is cheap."""
        sp = self.select_span_from(text, max(0, anchor_char - 4))
        if sp is not None:
            return sp.end_char
        return anchor_char + 1

    def scan_sites(self, text: str) -> list[StructuralSpan]:
        """Detector-side scan: same as v6 but each site advances past its own
        constituent end (no fixed-K window)."""
        sites, search_from = [], 0
        for _ in range(self.config.max_spans):
            sp = self.select_span_from(text, search_from)
            if sp is None:
                break
            sites.append(sp)
            search_from = sp.end_char
        return sites


class SpanWMV9(SpanWMV7):
    """v7 pipeline over constituency spans: constituent-length regeneration,
    constituent-extent detection."""

    def __init__(self, algorithm_config, transformers_config: TransformersConfig | None = None,
                 *args, **kwargs) -> None:
        if isinstance(algorithm_config, str):
            self.config = SpanWMV9Config(algorithm_config, transformers_config)
        elif isinstance(algorithm_config, SpanWMV9Config):
            self.config = algorithm_config
        else:
            raise TypeError("algorithm_config must be a path string or a SpanWMV9Config instance")
        self.utils = SpanWMV9Utils(self.config)
        self.logits_processor = KGWLogitsProcessor(self.config, self.utils.kgw)

    # -- embedding ---------------------------------------------------------
    def _regen_at(self, text: str, anchor_char: int, span_end_char: int):
        """v7's drift-free, punctuation-aware splice, but the number of
        watermarked tokens = the ORIGINAL constituent's token length (the
        span passed the min/max_span_tokens filter, so k is already bounded).
        Returns (new_text, fill_len)."""
        left = text[:anchor_char]
        right = text[span_end_char:]
        k = max(1, self.utils.span_token_count(text, anchor_char, span_end_char))
        strip_left = left.endswith(" ")
        ctx = left[:-1] if strip_left else left
        for _ in range(3):
            fill, _ = self._raw_generate(ctx, watermark=True, max_new_tokens=k, min_new_tokens=k)
            if fill.lstrip(" ")[:1] not in self._NO_OPEN:
                break
        if strip_left and fill.lstrip(" ")[:1] in self._NO_OPEN:
            fill = fill.lstrip(" ")
            prefix = ctx
        elif strip_left and fill.startswith(" "):
            prefix = ctx
        else:
            prefix = left
        if fill.endswith(" ") and right and (right[0].isspace() or right[0] in self._NO_SEP_BEFORE):
            fill = fill.rstrip(" ") or fill
        body = prefix + fill
        sep_r = "" if (not right or right[0].isspace() or right[0] in self._NO_SEP_BEFORE
                       or (fill and fill[-1].isspace())) else " "
        return body + sep_r + right, len(body) - anchor_char

    # -- detection ---------------------------------------------------------
    def detect_watermark(self, text: str, return_dict: bool = True, *args, **kwargs):
        """v6's pooled exact binomial test, but each site contributes the
        token extent of its RECONSTRUCTED constituent (variable N per site)."""
        sites = self.utils.scan_sites(text)
        if not sites:
            result = {"is_watermarked": False, "score": None, "p_value": None,
                      "num_tested_tokens": 0, "num_green_tokens": 0,
                      "num_sites": 0, "sites": [], "reconstructed": False}
            return result if return_dict else (False, None)

        positions = []
        input_ids = None
        for sp in sites:
            pos, input_ids = self.utils.mapper.map_positions(text, sp)
            positions.extend(pos)
        positions = sorted(set(positions))

        z, p, _, n, g = self.utils.score_span(input_ids, positions)
        is_watermarked = bool(p is not None and p < self.config.p_threshold)

        result = {
            "is_watermarked": is_watermarked,
            "score": z,
            "p_value": p,
            "num_tested_tokens": n,
            "num_green_tokens": g,
            "num_sites": len(sites),
            "sites": [{"role": sp.role, "anchor": sp.start_char, "text": sp.text}
                      for sp in sites],
            "reconstructed": True,
        }
        return result if return_dict else (is_watermarked, z)
