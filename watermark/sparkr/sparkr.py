# ==============================================================
# sparkr.py
# Port of SpARK-R — "SpARK: An Embarrassingly Simple Sparse Watermarking in
# LLMs with Enhanced Text Quality" (Findings of EACL 2026),
# https://github.com/mail-research/SpARK-llm-watermarking
#
# Method (paper §3.3.2): a token is a watermark TRIGGER when the hash of its
# token ID is divisible by D (`modular`). Generation runs freely; whenever the
# sampled token is a trigger but NOT in a FIXED global green list (a keyed
# gamma-fraction of the word-initial-token table), that step is resampled with
# the raw logits restricted to the green list. Detection re-finds trigger
# tokens from the text alone and z-tests their green membership.
#
# ⚠ REIMPLEMENTATION NOTES (upstream code is not runnable as committed):
#  1. Upstream's SpARK_RWatermark.__call__ returns an (input_ids, scores)
#     tuple and truncates input_ids — outside the HuggingFace LogitsProcessor
#     contract (a processor may only return modified scores); the patched
#     generation loop the authors used is not in the repo. We therefore
#     implement the paper's algorithm as an explicit step-wise sampling loop:
#     sample -> if trigger and red -> resample the SAME step from green-masked
#     raw logits (equivalent to upstream's intended backtrack, without needing
#     to rewind). The sampling chain replicates HF's warper order
#     (temperature -> top_k -> top_p) so results are comparable to the other
#     baselines run through model.generate().
#  2. Upstream imports `selfhash_no_anchor` which is DEFINED NOWHERE in the
#     repo. Following the paper ("we compute its hash value using its token
#     ID and compute its remainder against a divisor D") we reconstruct it as
#     the codebase's avalanche integer hash of salt*token_id:
#         selfhash_no_anchor(t, salt) = hashint(salt * t)
#     (hashint = the seeded fixed-permutation hash in prf_schemes.py).
#  3. The word-start marker is auto-detected ("▁" SentencePiece / "Ġ" BPE);
#     upstream special-cases llama3 by model-name string.
# Faithful details kept: table filter (word-initial, len>1, English second
# char, hash%D==0), fixed green list drawn with a CUDA generator seeded 0 over
# a permutation of the table, generation-side trigger requires len>1 while
# detection-side does not (upstream asymmetry), z over trigger positions.
# ==============================================================

from math import sqrt

import torch

from ..base import BaseWatermark, BaseConfig
from utils.transformers_config import TransformersConfig
from ..sparkp.prf_schemes import hashint

ENGLISH_ALPHABET = "abcdefghijklmnopqrstuvwxyz"

# surface stopword list for table_filter="content" (survivability-targeted
# triggers: content-like tokens have measured paraphrase EXCESS survival
# +0.464 vs +0.298 for function-like — plant where the signal survives)
CONTENT_STOPWORDS = set(
    "this that these those have has had been being will would can could may "
    "might shall should must about into over after before between during "
    "under above again further then once here there when where why how all "
    "any both each more most other some such only own same than very just "
    "also with from they them their your what which while does did doing "
    "because through against among within without said says".split())


def selfhash_no_anchor(token_id: int, salt_key: int) -> int:
    """Reconstruction of upstream's missing PRF (see module docstring)."""
    return int(hashint(torch.as_tensor(salt_key * int(token_id))))


class SpARKRConfig(BaseConfig):
    """Config class for SpARK-R."""

    def initialize_parameters(self) -> None:
        self.gamma = self.config_dict['gamma']
        self.modular = int(self.config_dict['modular'])   # paper's divisor D
        self.hash_key = self.config_dict.get('hash_key', 15485863)
        self.greenlist_seed = self.config_dict.get('greenlist_seed', 0)
        self.z_threshold = self.config_dict.get('z_threshold', 4.0)
        # bl_type "hard" = paper's SpARK-R (trigger tokens resampled from the
        # green list).  [the "soft" variant -- a global +delta bias on the
        # fixed global green list at EVERY step, no resampling — trigger
        # positions then land green with boosted (not certain) probability.
        # "soft_fixed" = position-fixed soft: triggers are decided by the
        # UNBIASED natural sample (same application rate as hard, ~no
        # selection effect); a red trigger is resampled WITHIN the trigger
        # table with +delta preference for green (red stays possible).
        self.bl_type = self.config_dict.get('bl_type', 'hard')
        self.delta = float(self.config_dict.get('delta', 0.0))
        # "content" restricts the trigger table to content-like word-initial
        # tokens (len>=4, not in CONTENT_STOPWORDS) — survivability targeting.
        self.table_filter = self.config_dict.get('table_filter', 'none')
        # "hashint" derives greens by integer hashing (device-independent,
        # avoids the CUDA-randperm device dependence of the legacy scheme).
        self.green_scheme = self.config_dict.get('green_scheme', 'randperm')
        # generation-side entropy gate for soft_fixed (token-form quality
        # lever; detection never re-derives it)
        self.ent_tau = self.config_dict.get('ent_tau', None)

    @property
    def algorithm_name(self) -> str:
        return 'SpARKR'


class SpARKRUtils:
    """Trigger table + fixed global green list + detection bits."""

    def __init__(self, config: SpARKRConfig) -> None:
        self.config = config
        self.tokenizer = config.generation_tokenizer
        self.device = config.device

        first_piece = self.tokenizer.tokenize(" hello")[0]
        self.marker = first_piece[0]
        if self.marker not in ("▁", "Ġ"):
            raise ValueError(f"unrecognized word-start marker {first_piece!r}")

        self._hash_cache = {}
        table = []
        for i in range(len(self.tokenizer.get_vocab())):
            tok = self.tokenizer.convert_ids_to_tokens(i)
            if tok is None or len(tok) <= 1 or tok[0] != self.marker \
                    or tok[1].lower() not in ENGLISH_ALPHABET:
                continue
            if not self._passes_table_filter(tok):
                continue
            if self.prf(i) % config.modular == 0:
                table.append(i)
        self.table_size = len(table)
        if self.table_size == 0:
            raise ValueError("SpARK-R trigger table is empty")
        self.table = torch.tensor(table, device=self.device)

        if config.green_scheme == "hashint":
            # device-independent keyed greens over the table
            gkey = config.hash_key + 31
            green = [i for i in table
                     if int(hashint(torch.as_tensor(gkey * (i + 1)))) % 1000
                     < int(config.gamma * 1000)]
            self.greenlist_ids = torch.tensor(green, device=self.device)
        else:
            # legacy: fixed global green list, upstream cuda generator seed 0
            rng = torch.Generator(device=self.device)
            rng.manual_seed(config.greenlist_seed)
            perm = torch.randperm(self.table_size, device=self.device, generator=rng)
            greenlist_size = int(self.table_size * config.gamma)
            self.greenlist_ids = self.table[perm][:greenlist_size]
        self.greenlist_set = set(self.greenlist_ids.tolist())
        vocab_size = len(self.tokenizer)
        self.green_mask = torch.zeros(vocab_size, dtype=torch.bool, device=self.device)
        self.green_mask[self.greenlist_ids] = True
        self.table_mask = torch.zeros(vocab_size, dtype=torch.bool, device=self.device)
        self.table_mask[self.table] = True

    def _passes_table_filter(self, piece: str) -> bool:
        if self.config.table_filter != "content":
            return True
        w = piece[1:].lower()
        return len(w) >= 4 and w.isalpha() and w not in CONTENT_STOPWORDS

    def prf(self, token_id: int) -> int:
        v = self._hash_cache.get(token_id)
        if v is None:
            v = selfhash_no_anchor(token_id, self.config.hash_key)
            self._hash_cache[token_id] = v
        return v

    def is_generation_trigger(self, token_id: int) -> bool:
        """Upstream generation-side condition (needs len>1)."""
        tok = self.tokenizer.convert_ids_to_tokens(token_id)
        if tok is None or len(tok) <= 1 or tok[0] != self.marker:
            return False
        if not self._passes_table_filter(tok):
            return False
        return self.prf(token_id) % self.config.modular == 0

    def decode_bits(self, text: str) -> list[int]:
        """Upstream detection: trigger = hash%D==0 and word-initial (no len
        check), bit = membership in the fixed green list."""
        bits = []
        tokens = self.tokenizer.encode(text)
        pieces = self.tokenizer.convert_ids_to_tokens(tokens)
        for tid, piece in zip(tokens, pieces):
            if piece and piece[0] == self.marker and len(piece) > 1 \
                    and self._passes_table_filter(piece) \
                    and self.prf(tid) % self.config.modular == 0:
                bits.append(1 if tid in self.greenlist_set else 0)
        return bits


class SpARKR(BaseWatermark):
    """Top-level class for the SpARK-R baseline (custom sampling loop)."""

    def __init__(self, algorithm_config: str | SpARKRConfig,
                 transformers_config: TransformersConfig | None = None, *args, **kwargs) -> None:
        if isinstance(algorithm_config, str):
            self.config = SpARKRConfig(algorithm_config, transformers_config)
        elif isinstance(algorithm_config, SpARKRConfig):
            self.config = algorithm_config
        else:
            raise TypeError("algorithm_config must be a path string or a SpARKRConfig instance")
        self.utils = SpARKRUtils(self.config)
        self.logits_processor = None  # generation runs through our own loop

    def _warp(self, logits: torch.Tensor) -> torch.Tensor:
        """HF warper chain for do_sample: temperature -> top_k -> top_p."""
        gk = self.config.gen_kwargs
        temperature = float(gk.get("temperature", 1.0))
        top_k = int(gk.get("top_k", 50))
        top_p = float(gk.get("top_p", 1.0))
        logits = logits / temperature
        if top_k > 0 and top_k < logits.shape[-1]:
            kth = torch.topk(logits, top_k)[0][..., -1, None]
            logits = logits.masked_fill(logits < kth, -float("inf"))
        if top_p < 1.0:
            sorted_logits, sorted_idx = torch.sort(logits, descending=True)
            probs = torch.softmax(sorted_logits, dim=-1)
            cum = torch.cumsum(probs, dim=-1)
            remove = cum - probs > top_p
            mask = torch.zeros_like(logits, dtype=torch.bool)
            mask.scatter_(-1, sorted_idx, remove)
            logits = logits.masked_fill(mask, -float("inf"))
        return logits

    @torch.no_grad()
    def generate_watermarked_text(self, prompt: str, *args, **kwargs) -> str:
        tok = self.config.generation_tokenizer
        model = self.config.generation_model
        gk = self.config.gen_kwargs
        max_new = int(gk.get("max_new_tokens", 200))
        do_sample = bool(gk.get("do_sample", True))
        eos_id = tok.eos_token_id

        enc = tok(prompt, return_tensors="pt", add_special_tokens=True).to(self.config.device)
        ids = enc["input_ids"]
        past = None
        generated = []
        step_input = ids
        for _ in range(max_new):
            out = model(input_ids=step_input, past_key_values=past, use_cache=True)
            past = out.past_key_values
            logits = out.logits[:, -1, :].float()
            warped = self._warp(logits.clone()) if do_sample else logits
            if do_sample:
                probs = torch.softmax(warped, dim=-1)
                t = int(torch.multinomial(probs, 1).item())
            else:
                t = int(torch.argmax(warped, dim=-1).item())
            reroute = (self.utils.is_generation_trigger(t)
                       and t not in self.utils.greenlist_set)
            if reroute and self.config.bl_type == "soft_fixed" \
                    and self.config.ent_tau is not None:
                p0 = torch.softmax(logits, dim=-1)
                ent = float(-(p0 * torch.log(p0.clamp_min(1e-12))).sum())
                reroute = ent > float(self.config.ent_tau)
            if reroute:
                if self.config.bl_type == "soft_fixed":
                    # position-fixed soft: stay inside the trigger table,
                    # prefer (not force) green by +delta
                    biased = logits.masked_fill(
                        ~self.utils.table_mask[:logits.shape[-1]].unsqueeze(0), -float("inf"))
                    biased = biased + self.config.delta * \
                        self.utils.green_mask[:logits.shape[-1]].float().unsqueeze(0)
                    warped_g = self._warp(biased) if do_sample else biased
                else:
                    # paper (hard): regenerate this token out of the green list
                    warped_g = logits.masked_fill(
                        ~self.utils.green_mask[:logits.shape[-1]].unsqueeze(0), -float("inf"))
                    if do_sample:
                        warped_g = self._warp(warped_g)
                if do_sample:
                    probs = torch.softmax(warped_g, dim=-1)
                    t = int(torch.multinomial(probs, 1).item())
                else:
                    t = int(torch.argmax(warped_g, dim=-1).item())
            generated.append(t)
            if eos_id is not None and t == eos_id:
                break
            step_input = torch.tensor([[t]], device=self.config.device)
        full = torch.cat([ids[0], torch.tensor(generated, device=self.config.device)])
        return tok.decode(full, skip_special_tokens=True)

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
