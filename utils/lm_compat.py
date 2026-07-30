"""Model loading that works across both transformers majors this repo runs on.

Two environments are unavoidable here: `wmattack` (transformers 4.55) drives
Llama and Qwen, while gemma-4 is a `gemma4_unified` checkpoint that 4.55
refuses to load at all, so it needs `vllm` (transformers 5.14). The two majors
differ in ways that break silently rather than loudly, which is what this
module exists to absorb.
"""

import transformers


def dtype_kwargs(dtype="auto"):
    """`torch_dtype` (<=4.55) vs `dtype` (>=5.0) for from_pretrained.

    Passing the wrong one is not a no-op: 4.55 rejects `dtype` outright, and
    5.x ignores `torch_dtype`, quietly loading fp32 and doubling memory.
    """
    major = int(transformers.__version__.split(".")[0])
    return {"dtype": dtype} if major >= 5 else {"torch_dtype": dtype}


def vocab_size(model_or_config, tokenizer):
    """The vocab size the watermark must key on.

    The greenlist is randperm(vocab_size), so embedding and detection have to
    derive this identically or they score unrelated green lists. Multimodal
    checkpoints (gemma-4) leave `config.vocab_size` None and nest the real
    value under `text_config`; reading the attribute naively would fall through
    to len(tokenizer) on one side and None on the other.
    """
    cfg = getattr(model_or_config, "config", model_or_config)
    v = getattr(cfg, "vocab_size", None)
    if v is None:
        v = getattr(getattr(cfg, "text_config", None), "vocab_size", None)
    return int(v) if v else len(tokenizer)
