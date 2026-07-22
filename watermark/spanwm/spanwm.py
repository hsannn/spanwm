# ==============================================================
# spanwm.py
# Description: Syntax-Guided Span Watermarking (SpanWM).
#
#   Embedding:  generate an unwatermarked draft -> parse -> key-select a
#               fixed role -> reconstruct that role's span -> regenerate
#               ONLY that span (left-context AR + splice) with a KGW-style
#               green-list watermark -> return the final text.
#   Detection:  parse the final text independently -> key-select the same
#               role -> reconstruct the span -> map to model tokens
#               (trim-then-overlap) -> compute a z-test over span tokens ONLY.
#
# The watermark core (green list, z-score) is reused from the KGW
# implementation; SpanWM adds the syntactic localization on top.
# ==============================================================

import torch
from math import sqrt
from scipy.stats import binom

from ..base import BaseWatermark, BaseConfig
from ..kgw.kgw import KGWUtils, KGWLogitsProcessor
from utils.transformers_config import TransformersConfig
from transformers import LogitsProcessorList
from visualize.data_for_visualization import DataForVisualization

from .span_ops import SpanExtractor, RoleSelector, TokenSpanMapper, StructuralSpan


class SpanWMConfig(BaseConfig):
    """Config class for SpanWM. Carries KGW watermark params + span policy."""

    def initialize_parameters(self) -> None:
        # --- KGW green-list watermark parameters (reused by KGWUtils) ------
        self.gamma = self.config_dict['gamma']
        self.delta = self.config_dict['delta']
        self.hash_key = self.config_dict['hash_key']
        self.z_threshold = self.config_dict['z_threshold']
        self.p_threshold = self.config_dict.get('p_threshold', 0.01)
        self.prefix_length = self.config_dict['prefix_length']
        self.f_scheme = self.config_dict['f_scheme']
        self.window_scheme = self.config_dict['window_scheme']

        # --- span policy --------------------------------------------------
        self.roles = self.config_dict['roles']
        self.spacy_model = self.config_dict['spacy_model']
        self.min_span_tokens = self.config_dict['min_span_tokens']
        self.max_span_tokens = self.config_dict['max_span_tokens']
        self.regen_max_new_tokens = self.config_dict['regen_max_new_tokens']
        # master key drives role selection (watermark key = hash_key)
        self.master_key = bytes.fromhex(self.config_dict['master_key'])

        # --- prompting (chat-template based generation) -------------------
        self.use_chat_template = self.config_dict.get('use_chat_template', True)
        self.system_prompt = self.config_dict['system_prompt']
        self.draft_prompt_template = self.config_dict['draft_prompt_template']
        self.regen_prompt_template = self.config_dict['regen_prompt_template']
        self.blank_marker = self.config_dict['blank_marker']

        # --- embed-time reconstruction verification -----------------------
        self.max_regen_attempts = self.config_dict.get('max_regen_attempts', 4)
        self.min_reconstruct_overlap = self.config_dict.get('min_reconstruct_overlap', 0.8)

        # --- fixed-K detection window (left-AR mode) ----------------------
        # Watermark exactly K tokens from the span anchor at embed time, and
        # test exactly K tokens from the reconstructed anchor at detect time.
        self.detect_window_tokens = self.config_dict.get('detect_window_tokens', 20)

    @property
    def algorithm_name(self) -> str:
        return 'SpanWM'


class SpanWMUtils:
    """Glue: green-list core (KGW) + span extraction / mapping / scoring."""

    def __init__(self, config: SpanWMConfig) -> None:
        self.config = config
        self.kgw = KGWUtils(config)                      # green list + f-scheme
        self.extractor = SpanExtractor(config.spacy_model)
        self.selector = RoleSelector(config.roles)
        self.mapper = TokenSpanMapper(config.generation_tokenizer)

    # -- span selection ----------------------------------------------------
    def select_span(self, text: str) -> StructuralSpan | None:
        """Reproducibly select one eligible span: keyed role, first-of-role."""
        role = self.selector.select_role(self.config.master_key)
        spans = self.extractor.extract(text)
        for sp in spans:                                 # spans are canonically sorted
            if sp.role != role:
                continue
            n = self.mapper.count(text, sp)
            if self.config.min_span_tokens <= n <= self.config.max_span_tokens:
                return sp
        return None

    @staticmethod
    def overlap_frac(span: "StructuralSpan", start: int, end: int) -> float:
        """Fraction of `span`'s characters that fall inside [start, end)."""
        if span is None:
            return 0.0
        inter = max(0, min(span.end_char, end) - max(span.start_char, start))
        span_len = span.end_char - span.start_char
        return inter / span_len if span_len > 0 else 0.0

    # -- span-only detection scoring --------------------------------------
    def score_span(self, input_ids: list[int], positions: list[int]):
        """Green-list test over the span token positions only.

        Each span token's green list is computed from its true preceding
        context in the full-text tokenization, so it matches generation time.

        Returns (z, p, flags, N, G) where
          z = normal-approx z-score (large-N),
          p = EXACT one-sided binomial p-value P(Binom(N, gamma) >= G),
              which is the correct significance for the small N of a span.
        """
        ids = torch.as_tensor(input_ids, dtype=torch.long, device=self.config.device)
        green_count = 0
        scored = 0
        flags = [-1] * len(input_ids)
        for idx in positions:
            if idx < self.config.prefix_length:
                continue                                 # no context to seed green list
            greenlist = self.kgw.get_greenlist_ids(ids[:idx])
            if ids[idx] in greenlist:
                green_count += 1
                flags[idx] = 1
            else:
                flags[idx] = 0
            scored += 1
        if scored == 0:
            return None, None, flags, 0, 0
        expected = self.config.gamma
        z = (green_count - expected * scored) / sqrt(scored * expected * (1 - expected))
        # exact one-sided binomial tail: P(X >= G) under X ~ Binom(N, gamma)
        p = float(binom.sf(green_count - 1, scored, expected))
        return z, p, flags, scored, green_count


class SpanWM(BaseWatermark):
    """Top-level class for Syntax-Guided Span Watermarking."""

    def __init__(self, algorithm_config: str | SpanWMConfig,
                 transformers_config: TransformersConfig | None = None, *args, **kwargs) -> None:
        if isinstance(algorithm_config, str):
            self.config = SpanWMConfig(algorithm_config, transformers_config)
        elif isinstance(algorithm_config, SpanWMConfig):
            self.config = algorithm_config
        else:
            raise TypeError("algorithm_config must be a path string or a SpanWMConfig instance")

        self.utils = SpanWMUtils(self.config)
        self.logits_processor = KGWLogitsProcessor(self.config, self.utils.kgw)

    # -- generation --------------------------------------------------------
    def _build_inputs(self, user_prompt: str):
        """Wrap a user instruction in the chat template (system + user)."""
        tok = self.config.generation_tokenizer
        if self.config.use_chat_template:
            messages = [
                {"role": "system", "content": self.config.system_prompt},
                {"role": "user", "content": user_prompt},
            ]
            text = tok.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True)
            enc = tok(text, return_tensors="pt", add_special_tokens=False)
        else:
            enc = tok(user_prompt, return_tensors="pt", add_special_tokens=True)
        return enc.to(self.config.device)

    def _chat_generate(self, user_prompt: str, watermark: bool,
                       max_new_tokens: int, min_new_tokens: int | None = None) -> str:
        """Generate the assistant response to an instruction; return only the
        newly generated text (stripped of preamble / quotes)."""
        tok = self.config.generation_tokenizer
        model = self.config.generation_model
        enc = self._build_inputs(user_prompt)
        prompt_len = enc["input_ids"].shape[1]

        gen_kwargs = dict(
            max_new_tokens=max_new_tokens,
            do_sample=bool(self.config.gen_kwargs.get("do_sample", True)),
            pad_token_id=tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id,
        )
        for k in ("top_p", "temperature", "top_k"):
            if k in self.config.gen_kwargs:
                gen_kwargs[k] = self.config.gen_kwargs[k]
        if min_new_tokens is not None:
            gen_kwargs["min_new_tokens"] = min_new_tokens
        if watermark:
            gen_kwargs["logits_processor"] = LogitsProcessorList([self.logits_processor])

        out = model.generate(**enc, **gen_kwargs)
        new_ids = out[0][prompt_len:]
        text = tok.decode(new_ids, skip_special_tokens=True)
        return self._clean(text)

    @staticmethod
    def _clean(text: str) -> str:
        """Strip stray surrounding quotes / whitespace from a model reply."""
        text = text.strip()
        if len(text) >= 2 and text[0] in "\"'" and text[-1] == text[0]:
            text = text[1:-1].strip()
        return text

    def _raw_generate(self, context: str, watermark: bool, max_new_tokens: int,
                      min_new_tokens: int | None = None):
        """Base-model completion of `context`. Returns (new_text, new_ids)."""
        tok = self.config.generation_tokenizer
        model = self.config.generation_model
        enc = tok(context, return_tensors="pt", add_special_tokens=True).to(self.config.device)
        plen = enc["input_ids"].shape[1]
        gen_kwargs = dict(
            max_new_tokens=max_new_tokens,
            do_sample=bool(self.config.gen_kwargs.get("do_sample", True)),
            pad_token_id=tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id,
        )
        for k in ("top_p", "temperature", "top_k"):
            if k in self.config.gen_kwargs:
                gen_kwargs[k] = self.config.gen_kwargs[k]
        if min_new_tokens is not None:
            gen_kwargs["min_new_tokens"] = min_new_tokens
        if watermark:
            gen_kwargs["logits_processor"] = LogitsProcessorList([self.logits_processor])
        out = model.generate(**enc, **gen_kwargs)
        new_ids = out[0][plen:]
        return tok.decode(new_ids, skip_special_tokens=True), new_ids

    def _generate_draft(self, prompt: str) -> str:
        max_new = self.config.gen_kwargs.get("max_new_tokens", 200)
        if self.config.use_chat_template:
            user = self.config.draft_prompt_template.format(prompt=prompt)
            return self._chat_generate(user, watermark=False, max_new_tokens=max_new)
        # base model: raw completion (prompt + continuation), like MarkLLM
        text, _ = self._raw_generate(prompt, watermark=False, max_new_tokens=max_new)
        return prompt + text

    def _regenerate_span(self, draft: str, span: StructuralSpan):
        """Regenerate the span region and return (final, wm_start, wm_end).

        Base / left-AR mode: feed only the left context, freely generate exactly
        K watermarked tokens (high entropy -> the green-list bias actually lands,
        and pure left-to-right keeps tokenization identical at detection), then
        splice the right context back. The watermark covers the K generated
        tokens starting at char = len(left)."""
        left = draft[:span.start_char]
        right = draft[span.end_char:]
        if self.config.use_chat_template:
            user = self.config.regen_prompt_template.format(
                left=left, right=right, blank=self.config.blank_marker)
            fill = self._chat_generate(
                user, watermark=True, max_new_tokens=self.config.regen_max_new_tokens)
            sep_l = "" if (not left or left[-1] == " " or fill.startswith(" ")) else " "
            sep_r = "" if (not right or right[0] == " " or fill.endswith(" ")) else " "
            prefix = left + sep_l
            return prefix + fill + sep_r + right, len(prefix), len(prefix) + len(fill)
        # base / left-AR: exactly K watermarked tokens continued from `left`
        k = self.config.detect_window_tokens
        new_text, _ = self._raw_generate(left, watermark=True, max_new_tokens=k, min_new_tokens=k)
        sep_r = "" if (not right or right[0] == " " or new_text.endswith(" ")) else " "
        final_text = left + new_text + sep_r + right
        return final_text, len(left), len(left) + len(new_text)

    def _verify_recon(self, recon, wm_start: int, wm_end: int):
        """Will the detector's reconstructed anchor land on the watermark?
        Returns (verified, score) — higher score = better alignment."""
        if recon is None:
            return False, -1e9
        if self.config.use_chat_template:                    # constituent-overlap mode
            frac = self.utils.overlap_frac(recon, wm_start, wm_end)
            return frac >= self.config.min_reconstruct_overlap, frac
        # fixed-K window mode: the anchor (recon.start) must align with wm_start
        dist = abs(recon.start_char - wm_start)
        return dist <= 4, -dist

    def generate_watermarked_text(self, prompt: str, *args, **kwargs) -> str:
        """Draft -> keyed-role span -> regenerate under watermark, then VERIFY
        that an independent re-parse reconstructs the same anchor. Retry until it
        does (or give up), so the detector tests the watermarked tokens."""
        draft = self._generate_draft(prompt)
        span = self.utils.select_span(draft)
        if span is None:
            self.last_embedding_info = {
                "selected_span": None, "skipped": True, "verified": False,
                "attempts": 0, "anchor_dist": None, "draft": draft}
            return draft

        best = None  # (final, recon, ws, we, score)
        for attempts in range(1, self.config.max_regen_attempts + 1):
            final_text, ws, we = self._regenerate_span(draft, span)
            recon = self.utils.select_span(final_text)          # what the detector will pick
            ok, score = self._verify_recon(recon, ws, we)
            if best is None or score > best[4]:
                best = (final_text, recon, ws, we, score)
            if ok:
                break

        final_text, recon, ws, we, score = best
        verified = self._verify_recon(recon, ws, we)[0]
        self.last_embedding_info = {
            "selected_span": span, "skipped": False, "verified": verified,
            "attempts": attempts,
            "anchor_dist": None if recon is None else abs(recon.start_char - ws),
            "wm_char_range": [ws, we], "draft": draft, "final_text": final_text,
            "reconstructed_span": None if recon is None else recon.text,
        }
        return final_text

    def generate_unwatermarked_text(self, prompt: str, *args, **kwargs) -> str:
        """Unwatermarked baseline: same generation path as the draft, no
        watermark anywhere (comparable negative class)."""
        return self._generate_draft(prompt)

    # -- detection ---------------------------------------------------------
    def detect_watermark(self, text: str, return_dict: bool = True, *args, **kwargs):
        span = self.utils.select_span(text)
        if span is None:
            result = {"is_watermarked": False, "score": None, "p_value": None,
                      "num_tested_tokens": 0, "num_green_tokens": 0,
                      "selected_role": self.utils.selector.select_role(self.config.master_key),
                      "selected_span": None, "reconstructed": False}
            return result if return_dict else (False, None)

        positions, input_ids = self._detection_positions(text, span)
        z, p, _, n, g = self.utils.score_span(input_ids, positions)
        # decide by the EXACT binomial p-value (correct for small span N)
        is_watermarked = bool(p is not None and p < self.config.p_threshold)

        result = {
            "is_watermarked": is_watermarked,
            "score": z,
            "p_value": p,
            "num_tested_tokens": n,
            "num_green_tokens": g,
            "selected_role": span.role,
            "selected_span": span.text,
            "span_char_range": (span.start_char, span.end_char),
            "reconstructed": True,
        }
        return result if return_dict else (is_watermarked, z)

    def _detection_positions(self, text: str, span):
        """Token positions to score: a fixed-K window from the anchor (base /
        left-AR mode) or the constituent itself (chat / infill mode)."""
        if self.config.use_chat_template:
            return self.utils.mapper.map_positions(text, span)
        return self.utils.mapper.window_positions(
            text, span.start_char, self.config.detect_window_tokens)

    # -- visualization -----------------------------------------------------
    def get_data_for_visualization(self, text: str, *args, **kwargs) -> DataForVisualization:
        span = self.utils.select_span(text)
        tok = self.config.generation_tokenizer
        input_ids, _ = self.utils.mapper.encode_with_offsets(text)
        if span is None:
            flags = [-1] * len(input_ids)
        else:
            positions, input_ids = self._detection_positions(text, span)
            _, _, flags, _, _ = self.utils.score_span(input_ids, positions)
        decoded = [tok.decode([tid]) for tid in input_ids]
        return DataForVisualization(decoded, flags)
