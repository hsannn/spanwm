# ==============================================================
# spanwm_v4.py
# SpanWM v4 = base model + FEW-SHOT infilling (+ gamma=0.25 + exact-p).
#
# This is an ablation of the infilling idea using a *base* model: base LMs do
# not follow zero-shot instructions, but they DO pattern-match few-shot
# examples, so we prompt with a few "Passage ... [BLANK] ... / Answer: ..."
# demonstrations and let the model produce the fill for our blank.
#
# Everything else (span extraction, role selection, KGW green list, exact
# binomial p, verification loop) is inherited from the v3 SpanWM. Only the
# regeneration (few-shot infilling instead of left-AR) and the detection unit
# (the reconstructed constituent instead of a fixed-K window) differ.
#
# NOTE: infilling constrains the fill between left+right context -> low entropy
# -> the green-list bias may not land (this is exactly why the earlier chat-
# infilling run failed). v4 exists to measure that with a base model + the
# gamma/exact-p improvements. Expect it to be weaker than v3.
# ==============================================================

from watermark.spanwm.spanwm import SpanWM, SpanWMConfig, SpanWMUtils
from watermark.kgw.kgw import KGWLogitsProcessor
from utils.transformers_config import TransformersConfig
from watermark.spanwm.span_ops import StructuralSpan


class SpanWMV4Config(SpanWMConfig):
    """v3 config + few-shot infilling prompt fields."""

    def initialize_parameters(self) -> None:
        super().initialize_parameters()
        self.fewshot_prefix = self.config_dict['fewshot_prefix']
        self.fewshot_template = self.config_dict['fewshot_template']

    @property
    def algorithm_name(self) -> str:
        return 'SpanWM_v4'


class SpanWMV4(SpanWM):
    """Base-model few-shot infilling variant of SpanWM."""

    def __init__(self, algorithm_config, transformers_config: TransformersConfig | None = None,
                 *args, **kwargs) -> None:
        if isinstance(algorithm_config, str):
            self.config = SpanWMV4Config(algorithm_config, transformers_config)
        elif isinstance(algorithm_config, SpanWMV4Config):
            self.config = algorithm_config
        else:
            raise TypeError("algorithm_config must be a path string or a SpanWMV4Config instance")
        self.utils = SpanWMUtils(self.config)
        self.logits_processor = KGWLogitsProcessor(self.config, self.utils.kgw)

    # draft generation is inherited (base raw completion, use_chat_template=false)

    def _fewshot_generate(self, prompt: str, watermark: bool) -> str:
        """Generate the fill with a base model via few-shot prompting; keep only
        the first answer line."""
        text, _ = self._raw_generate(
            prompt, watermark=watermark, max_new_tokens=self.config.regen_max_new_tokens)
        fill = text.split("\n")[0]            # stop at the first newline (next demo)
        return self._clean(fill)

    def _regenerate_span(self, draft: str, span: StructuralSpan):
        """Few-shot infilling of the blank; watermark the generated fill.
        Returns (final_text, fill_start_char, fill_end_char)."""
        left = draft[:span.start_char]
        right = draft[span.end_char:]
        prompt = self.config.fewshot_prefix + self.config.fewshot_template.format(
            left=left, right=right)
        fill = self._fewshot_generate(prompt, watermark=True)
        if not fill:
            fill = span.text                  # degenerate fallback: keep original text
        sep_l = "" if (not left or left[-1] == " " or fill.startswith(" ")) else " "
        sep_r = "" if (not right or right[0] == " " or fill.endswith(" ")) else " "
        prefix = left + sep_l
        final_text = prefix + fill + sep_r + right
        return final_text, len(prefix), len(prefix) + len(fill)

    def _verify_recon(self, recon, wm_start: int, wm_end: int):
        """v4 tests the reconstructed constituent, so verify by char-overlap of
        the reconstructed span with the watermarked fill region."""
        if recon is None:
            return False, -1e9
        frac = self.utils.overlap_frac(recon, wm_start, wm_end)
        return frac >= self.config.min_reconstruct_overlap, frac

    def _detection_positions(self, text: str, span):
        """v4 scores the reconstructed constituent (trim-then-overlap), not a
        fixed-K window."""
        return self.utils.mapper.map_positions(text, span)
