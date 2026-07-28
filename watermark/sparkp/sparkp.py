# ==============================================================
# sparkp.py
# Port of SpARK-P — "SpARK: An Embarrassingly Simple Sparse Watermarking in
# LLMs with Enhanced Text Quality" (Findings of EACL 2026), code from
# https://github.com/mail-research/SpARK-llm-watermarking
# (watermark/spark_p_watermark.py) — wrapped in this repo's MarkLLM-style
# interface.
#
# Method: watermark ONLY the token that starts a new word right after a word
# whose POS tag is in `pos_tags` (NLTK Penn tags, prefix match: "V" -> VB*,
# "NN" -> NN*/NNP*, "DT" -> determiners). The green list is drawn from a
# restricted table of word-initial English tokens, seeded with the KGW
# lefthash PRF (context width 1, hash_key 15485863). bl_type "hard" restricts
# generation at trigger steps to green tokens only (the paper's setting; delta
# unused); "soft" adds delta to green logits instead. Detection re-finds the
# trigger positions from the text alone and z-tests only those tokens.
#
# Deliberate deviations from upstream (everything else kept line-faithful):
#  1. The word-start marker is auto-detected from the tokenizer ("▁" for
#     SentencePiece/Llama-2, "Ġ" for BPE/Llama-3/GPT-2). Upstream hardcodes
#     "▁", which silently produces an EMPTY table (= no watermark) on Llama-3.
#  2. Per-step debug prints removed.
#  3. Config-file hyperparameters + BaseWatermark wrapper.
# ==============================================================

from math import sqrt

import torch
from transformers import LogitsProcessor, LogitsProcessorList

from ..base import BaseWatermark, BaseConfig
from utils.transformers_config import TransformersConfig
from .prf_schemes import prf_lookup, seeding_scheme_lookup

ENGLISH_ALPHABET = "abcdefghijklmnopqrstuvwxyz"


def _pos_tag_tools():
    """Import NLTK lazily with a helpful error if tagger data is missing."""
    from nltk import pos_tag, word_tokenize
    try:
        pos_tag(word_tokenize("The quick brown fox jumps."))
    except LookupError as e:
        raise LookupError(
            "NLTK data missing. Run on a node with internet:\n"
            "  python -m nltk.downloader punkt punkt_tab "
            "averaged_perceptron_tagger averaged_perceptron_tagger_eng"
        ) from e
    return pos_tag, word_tokenize


class SpARKPConfig(BaseConfig):
    """Config class for SpARK-P."""

    def initialize_parameters(self) -> None:
        self.gamma = self.config_dict['gamma']
        self.delta = self.config_dict['delta']
        self.bl_type = self.config_dict.get('bl_type', 'hard')
        self.pos_tags = list(self.config_dict.get('pos_tags', ['V']))
        self.seeding_scheme = self.config_dict.get('seeding_scheme', 'lefthash')
        self.z_threshold = self.config_dict.get('z_threshold', 4.0)

    @property
    def algorithm_name(self) -> str:
        return 'SpARKP'


class SpARKPLogitsProcessor(LogitsProcessor):
    """Upstream SpARK_PWatermark.__call__, marker-generalized."""

    def __init__(self, config: SpARKPConfig, utils: 'SpARKPUtils') -> None:
        self.config = config
        self.utils = utils
        self.prompt_slice = None

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        u = self.utils
        _, next_token = torch.sort(scores, dim=1, descending=True)
        next_token = next_token[:, 0]
        output_score = torch.zeros_like(scores)
        for b_idx in range(input_ids.shape[0]):
            tokens = input_ids[b_idx][self.prompt_slice:]
            text = u.tokenizer.decode(tokens)
            next_output = u.tokenizer.convert_ids_to_tokens(next_token[b_idx].item())
            if len(next_output) == 1 or next_output[0] != u.marker or len(tokens) == 0:
                output_score[b_idx] = scores[b_idx]
                continue
            curr_pos_tag = u.pos_tag(u.word_tokenize(text))
            if len(curr_pos_tag) == 0:
                output_score[b_idx] = scores[b_idx]
                continue
            _, current_tag = curr_pos_tag[-1]
            if not u.tag_allowed(current_tag):
                output_score[b_idx] = scores[b_idx]
                continue
            ids = u.get_greenlist_ids(tokens)
            mask_tokens = torch.zeros_like(scores[b_idx], dtype=torch.bool)
            mask_tokens[ids] = True
            if u.hard_encode:
                scores[b_idx] = scores[b_idx].masked_fill(~mask_tokens, -float("inf"))
            else:
                scores[b_idx] += mask_tokens.float() * self.config.delta
            output_score[b_idx] = scores[b_idx]
        return output_score


class SpARKPUtils:
    """Restricted word-initial table + lefthash green list + POS trigger."""

    def __init__(self, config: SpARKPConfig) -> None:
        self.config = config
        self.tokenizer = config.generation_tokenizer
        self.device = config.device
        self.hard_encode = config.bl_type == "hard"
        self.pos_tag, self.word_tokenize = _pos_tag_tools()

        # word-start marker: "▁" (SentencePiece) or "Ġ" (BPE)
        first_piece = self.tokenizer.tokenize(" hello")[0]
        self.marker = first_piece[0]
        if self.marker not in ("▁", "Ġ"):
            raise ValueError(f"unrecognized word-start marker {first_piece!r}")

        self.allowed_pos_tag = list(config.pos_tags)
        if "LRB" in self.allowed_pos_tag:
            self.allowed_pos_tag.append("-LRB-")
        if "RRB" in self.allowed_pos_tag:
            self.allowed_pos_tag.append("-RRB-")

        self.prf_type, self.context_width, self.self_salt, self.hash_key = \
            seeding_scheme_lookup(config.seeding_scheme)
        self.rng = torch.Generator(device=self.device)
        self.new_line = ["<0x0A>"]
        self.init_table()

    def init_table(self):
        table = []
        for i in range(len(self.tokenizer.get_vocab())):
            tok = self.tokenizer.convert_ids_to_tokens(i)
            if tok is None or len(tok) <= 1 or tok[0] != self.marker \
                    or tok[1].lower() not in ENGLISH_ALPHABET:
                continue
            table.append(i)
        self.table_size = len(table)
        if self.table_size == 0:
            raise ValueError("SpARK-P word-initial table is empty — wrong marker?")
        self.table = torch.tensor(table, device=self.device)

    def tag_allowed(self, current_tag: str) -> bool:
        for allowed_tag in self.allowed_pos_tag:
            if allowed_tag in current_tag[:len(allowed_tag)]:
                return True
        return False

    def _seed_rng(self, input_ids: torch.LongTensor) -> None:
        if input_ids.shape[-1] < self.context_width:
            raise ValueError(f"need >= {self.context_width} tokens to seed the RNG")
        prf_key = prf_lookup[self.prf_type](input_ids[-self.context_width:], salt_key=self.hash_key)
        self.rng.manual_seed(prf_key % (2 ** 64 - 1))

    def get_greenlist_ids(self, input_ids: torch.LongTensor) -> torch.LongTensor:
        self._seed_rng(input_ids)
        greenlist_size = int(self.table_size * self.config.gamma)
        vocab_permutation = torch.randperm(self.table_size, device=self.device, generator=self.rng)
        return self.table[vocab_permutation][:greenlist_size]

    def decode_bits(self, text: str) -> list[int]:
        """Upstream detector's decode(): reconstruct trigger positions from the
        text alone and return one green/red bit per tested position."""
        bits = []
        tokens = self.tokenizer.encode(text)
        next_outputs = self.tokenizer.convert_ids_to_tokens(tokens)
        for i in range(len(tokens)):
            next_output = next_outputs[i]
            prev_tokens = tokens[:i]
            if (next_output[0] == self.marker and next_output not in self.new_line) \
                    and len(prev_tokens) > 0 and len(next_output) > 1:
                inner_tokens = self.word_tokenize(self.tokenizer.decode(prev_tokens))
                tagged = self.pos_tag(inner_tokens)
                if len(tagged) == 0:
                    continue
                _, current_tag = tagged[-1]
                if not self.tag_allowed(current_tag):
                    continue
                ids = self.get_greenlist_ids(torch.tensor(prev_tokens, device=self.device))
                bits.append(1 if tokens[i] in ids else 0)
        return bits


class SpARKP(BaseWatermark):
    """Top-level class for the SpARK-P baseline."""

    def __init__(self, algorithm_config: str | SpARKPConfig,
                 transformers_config: TransformersConfig | None = None, *args, **kwargs) -> None:
        if isinstance(algorithm_config, str):
            self.config = SpARKPConfig(algorithm_config, transformers_config)
        elif isinstance(algorithm_config, SpARKPConfig):
            self.config = algorithm_config
        else:
            raise TypeError("algorithm_config must be a path string or a SpARKPConfig instance")
        self.utils = SpARKPUtils(self.config)
        self.logits_processor = SpARKPLogitsProcessor(self.config, self.utils)

    def generate_watermarked_text(self, prompt: str, *args, **kwargs) -> str:
        encoded_prompt = self.config.generation_tokenizer(
            prompt, return_tensors="pt", add_special_tokens=True).to(self.config.device)
        self.logits_processor.prompt_slice = encoded_prompt["input_ids"].shape[1]
        encoded = self.config.generation_model.generate(
            **encoded_prompt,
            logits_processor=LogitsProcessorList([self.logits_processor]),
            **self.config.gen_kwargs)
        return self.config.generation_tokenizer.batch_decode(
            encoded, skip_special_tokens=True)[0]

    def detect_watermark(self, text: str, return_dict: bool = True, *args, **kwargs):
        bits = self.utils.decode_bits(text)
        t, g = len(bits), sum(bits)
        gamma = self.config.gamma
        z = (g - gamma * t) / sqrt(t * gamma * (1 - gamma)) if t > 0 else 0.0
        is_watermarked = z > self.config.z_threshold
        if return_dict:
            return {"is_watermarked": is_watermarked, "score": z,
                    "num_tested_tokens": t, "num_green_tokens": g}
        return (is_watermarked, z)

    def get_data_for_visualization(self, text: str, *args, **kwargs):
        from visualize.data_for_visualization import DataForVisualization
        decoded_tokens = [self.config.generation_tokenizer.decode(tid)
                          for tid in self.config.generation_tokenizer.encode(text)]
        return DataForVisualization(decoded_tokens, [0] * len(decoded_tokens))
