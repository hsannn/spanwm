# ==============================================================
# spanwm_v6.py
# SpanWM v6 = MULTI-SPAN watermarking.
#
#   v3/v5 put all the statistical power in ONE anchor with a large K=20
#   window, whose free generation rambles far past the original constituent
#   (the main quality cost). v6 spreads the same power over SEVERAL anchors,
#   each with a small constituent-scale window (span_window_tokens, e.g. 6):
#
#       total N = max_spans x span_window_tokens   (e.g. 4 x 6 = 24 ~ v3's 20)
#
#   and pools the green counts of all sites into ONE exact binomial test
#   (every window token shares the same per-token null: P(green) = gamma).
#
#   NOTE a small FIXED window per site is still required — the constituent's
#   *end* boundary is unstable under re-parsing (the v1 failure), so both
#   sides always test "K_s tokens from the reconstructed anchor". Multi-span
#   removes the need for a LARGE K, not for the fixed-window trick itself.
#
# Site discovery (identical scan at embed and detect):
#   search_from = 0
#   repeat up to max_spans times:
#     parse the CURRENT text; take the first span at/after search_from whose
#     role matches its per-span PRF target (v5 rule) and passes the length
#     filter -> that anchor is a site;
#     advance search_from past the site's K_s-token window.
#   Embed regenerates each site left->right (K_s watermarked tokens, splice),
#   re-scanning the updated text each round, so the detector — which scans the
#   final text the same way — visits the same anchors. Text to the left of a
#   site is never touched by later regenerations.
# ==============================================================

from math import sqrt

from scipy.stats import binom

from watermark.spanwm_v5.spanwm_v5 import SpanWMV5, SpanWMV5Config, SpanWMV5Utils
from watermark.kgw.kgw import KGWLogitsProcessor
from utils.transformers_config import TransformersConfig
from watermark.spanwm.span_ops import StructuralSpan


class SpanWMV6Config(SpanWMV5Config):
    """v5 config + multi-span parameters."""

    def initialize_parameters(self) -> None:
        super().initialize_parameters()
        self.span_window_tokens = self.config_dict.get('span_window_tokens', 6)
        self.max_spans = self.config_dict.get('max_spans', 4)

    @property
    def algorithm_name(self) -> str:
        return 'SpanWM_v6'


class SpanWMV6Utils(SpanWMV5Utils):
    """v5 utils + position-bounded selection and the shared site scan."""

    def select_span_from(self, text: str, from_char: int) -> StructuralSpan | None:
        """First PRF-matching eligible span whose anchor is at/after from_char."""
        for sp in self.extractor.extract(text):        # canonically sorted
            if sp.start_char < from_char:
                continue
            if sp.role != self.role_for_anchor(text, sp.start_char):
                continue
            n = self.mapper.count(text, sp)
            if self.config.min_span_tokens <= n <= self.config.max_span_tokens:
                return sp
        return None

    def window_end_char(self, text: str, anchor_char: int) -> int:
        """Char position just past the K_s-token window at anchor (used to
        advance the scan identically at embed and detect time)."""
        positions, input_ids = self.mapper.window_positions(
            text, anchor_char, self.config.span_window_tokens)
        if not positions:
            return anchor_char + 1
        _, offsets = self.mapper.encode_with_offsets(text)
        return offsets[positions[-1]][1]

    def scan_sites(self, text: str) -> list[StructuralSpan]:
        """The shared detector-side scan: up to max_spans anchors, left->right,
        each advancing past its own window."""
        sites, search_from = [], 0
        for _ in range(self.config.max_spans):
            sp = self.select_span_from(text, search_from)
            if sp is None:
                break
            sites.append(sp)
            search_from = self.window_end_char(text, sp.start_char)
        return sites


class SpanWMV6(SpanWMV5):
    """Multi-span variant: several small watermarked windows, pooled test."""

    def __init__(self, algorithm_config, transformers_config: TransformersConfig | None = None,
                 *args, **kwargs) -> None:
        if isinstance(algorithm_config, str):
            self.config = SpanWMV6Config(algorithm_config, transformers_config)
        elif isinstance(algorithm_config, SpanWMV6Config):
            self.config = algorithm_config
        else:
            raise TypeError("algorithm_config must be a path string or a SpanWMV6Config instance")
        self.utils = SpanWMV6Utils(self.config)
        self.logits_processor = KGWLogitsProcessor(self.config, self.utils.kgw)

    # -- embedding ---------------------------------------------------------
    def _regen_at(self, text: str, anchor_char: int, span_end_char: int):
        """Regenerate K_s watermarked tokens at anchor; splice the rest back.
        Returns (new_text, fill_len)."""
        left = text[:anchor_char]
        right = text[span_end_char:]
        k = self.config.span_window_tokens
        new_text, _ = self._raw_generate(left, watermark=True, max_new_tokens=k, min_new_tokens=k)
        sep_r = "" if (not right or right[0] == " " or new_text.endswith(" ")) else " "
        return left + new_text + sep_r + right, len(new_text)

    def generate_watermarked_text(self, prompt: str, *args, **kwargs) -> str:
        """Draft -> scan sites left->right; at each site regenerate a small
        watermarked window (verify + retry per site) -> final text."""
        draft = self._generate_draft(prompt)
        text = draft
        sites = []          # per-site dicts for last_embedding_info
        search_from = 0

        for _ in range(self.config.max_spans):
            sp = self.utils.select_span_from(text, search_from)
            if sp is None:
                break
            anchor = sp.start_char
            best = None      # (text_after, fill_len, dist)
            for attempt in range(1, self.config.max_regen_attempts + 1):
                cand, fill_len = self._regen_at(text, anchor, sp.end_char)
                recon = self.utils.select_span_from(cand, search_from)
                dist = abs(recon.start_char - anchor) if recon is not None else 10 ** 9
                if best is None or dist < best[2]:
                    best = (cand, fill_len, dist)
                if dist <= 4:
                    break
            text, fill_len, dist = best
            sites.append({
                "role": sp.role, "anchor": anchor,
                "draft_span": sp.text, "fill": text[anchor:anchor + fill_len],
                "attempts": attempt, "anchor_dist": dist, "verified": dist <= 4,
            })
            search_from = self.utils.window_end_char(text, anchor)

        # final check: does the detector-side scan of the FINAL text land on
        # the same anchors?
        final_anchors = [s.start_char for s in self.utils.scan_sites(text)]
        embedded_anchors = [s["anchor"] for s in sites]
        aligned = sum(1 for a, b in zip(embedded_anchors, final_anchors) if abs(a - b) <= 4)
        verified = len(sites) > 0 and aligned == len(sites) == len(final_anchors)

        self.last_embedding_info = {
            "skipped": len(sites) == 0, "verified": verified,
            "num_sites": len(sites), "num_aligned": aligned,
            "sites": sites, "draft": draft, "final_text": text,
            # kept for script compatibility
            "selected_span": None, "attempts": sum(s["attempts"] for s in sites),
            "anchor_dist": None,
        }
        return text

    # -- detection ---------------------------------------------------------
    def detect_watermark(self, text: str, return_dict: bool = True, *args, **kwargs):
        sites = self.utils.scan_sites(text)
        if not sites:
            result = {"is_watermarked": False, "score": None, "p_value": None,
                      "num_tested_tokens": 0, "num_green_tokens": 0,
                      "num_sites": 0, "sites": [], "reconstructed": False}
            return result if return_dict else (False, None)

        # pool the window positions of every site into one test
        positions = []
        input_ids = None
        for sp in sites:
            pos, input_ids = self.utils.mapper.window_positions(
                text, sp.start_char, self.config.span_window_tokens)
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
