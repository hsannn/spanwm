"""Trace where the watermark signal dies: compare green-rate of the fill
tokens AT GENERATION TIME vs the reconstructed tokens AT DETECTION TIME.

If gen green-rate is high but detect green-rate is ~gamma -> decode/re-encode
tokenization drift (or greenlist context mismatch) is destroying the signal.
"""
import json, torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers import LogitsProcessorList
from watermark.auto_watermark import AutoWatermark
from utils.transformers_config import TransformersConfig

dev = "cuda" if torch.cuda.is_available() else "cpu"
tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-4B-Instruct-2507")
model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3-4B-Instruct-2507", dtype="auto", device_map=dev).eval()
tc = TransformersConfig(model=model, tokenizer=tok, vocab_size=model.config.vocab_size,
                        device=dev, max_new_tokens=160, do_sample=True, top_p=0.9, temperature=0.8)
wm = AutoWatermark.load("SpanWM", algorithm_config="config/SpanWM.json", transformers_config=tc)
U = wm.utils

def green_rate_of_ids(id_seq_prefix, target_ids):
    """green rate of target_ids given they follow id_seq_prefix (list ints)."""
    ids = torch.as_tensor(id_seq_prefix, dtype=torch.long, device=dev)
    g = 0
    for t in target_ids:
        gl = U.kgw.get_greenlist_ids(ids)
        hit = int(t in gl)
        g += hit
        ids = torch.cat([ids, torch.tensor([t], device=dev)])
    return g, len(target_ids)

lines = open("dataset/c4/processed_c4.json").readlines()
done = 0
for i in range(12):
    prompt = json.loads(lines[i])["prompt"]
    draft = wm._generate_draft(prompt)
    span = U.select_span(draft)
    if span is None:
        continue
    # --- reproduce the regen but capture the generated fill token ids ---
    left = draft[:span.start_char]; right = draft[span.end_char:]
    user = wm.config.regen_prompt_template.format(left=left, right=right, blank=wm.config.blank_marker)
    enc = wm._build_inputs(user)
    plen = enc["input_ids"].shape[1]
    out = model.generate(**enc, logits_processor=LogitsProcessorList([wm.logits_processor]),
                         max_new_tokens=wm.config.regen_max_new_tokens,
                         do_sample=True, top_p=0.9, temperature=0.8,
                         pad_token_id=tok.eos_token_id)
    gen_fill_ids = out[0][plen:].tolist()
    # strip trailing eos
    gen_fill_ids = [t for t in gen_fill_ids if t != tok.eos_token_id]
    fill_text = wm._clean(tok.decode(gen_fill_ids, skip_special_tokens=True))

    # GEN-TIME green rate: fill tokens given their generation context (chat prefix)
    gen_prefix = enc["input_ids"][0].tolist()
    g_gen, n_gen = green_rate_of_ids(gen_prefix, gen_fill_ids)

    # splice + DETECT-TIME green rate over the reconstructed span
    sep_l = "" if (not left or left[-1] == " " or fill_text.startswith(" ")) else " "
    sep_r = "" if (not right or right[0] == " " or fill_text.endswith(" ")) else " "
    final = left + sep_l + fill_text + sep_r + right
    det = wm.detect_watermark(final)

    # also: green rate of the fill tokens as they RE-TOKENIZE in the plain final text
    fs = len(left + sep_l); fe = fs + len(fill_text)
    from watermark.spanwm.span_ops import StructuralSpan
    fake = StructuralSpan(0, span.role, fs, fe, final[fs:fe], 0, 0)
    pos, ids_full = U.mapper.map_positions(final, fake)
    retok = [ids_full[p] for p in pos]

    print("="*90)
    print(f"[{i}] fill_text={fill_text!r}")
    print(f"    GEN fill ids (n={len(gen_fill_ids)}): green {g_gen}/{n_gen} = {g_gen/max(n_gen,1):.0%}")
    print(f"    fill re-tokenizes to n={len(retok)} tokens  (same ids as gen? {retok==gen_fill_ids})")
    print(f"    DETECT span={det['selected_span']!r}  N={det['num_tested_tokens']} G={det['num_green_tokens']} z={det['score']}")
    done += 1
    if done >= 6:
        break
