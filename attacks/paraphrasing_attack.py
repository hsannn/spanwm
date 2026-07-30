"""Paraphrasing attack on watermarked text.

Two paraphraser backends, one output convention:
  * `openai` — hosted API (gpt-5-mini, ...), parallelised with threads.
  * `hf`     — a local transformers model (openai/gpt-oss-20b,
               google/gemma-3-27b-it, Qwen/Qwen3-14B, ...), batched on GPU.
The backend is inferred from the model id ('/' -> hf) unless --backend says so.
"""

import os
import re
import sys
import json
import time
import argparse
from difflib import SequenceMatcher
from concurrent.futures import ThreadPoolExecutor
from dotenv import load_dotenv

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

DEFAULT_MODEL = 'gpt-5-mini'
DEFAULT_PROMPT_FILE = os.path.join(PROJECT_ROOT, 'attacks', 'paraphrase_prompt.txt')

# the original single-model run wrote *_GPTParaphrase.jsonl; keep that name so
# the already-detected outputs and their logs stay valid.
LEGACY_ATTACK_NAMES = {'gpt-5-mini': 'GPTParaphrase'}

NUDGE = ('That is the original passage, unchanged. Rewrite it now: new wording, '
         'new sentence structure, same content and same defects.')


def attack_name_for(model: str) -> str:
    """Filename/label tag for a paraphraser, unique per model.

    Works for a hub id (google/gemma-4-31B-it) and for a local checkpoint
    directory (/scratch/ssgyejin/models/gemma-4-31B-it/), which is what you get
    after copying weights in by hand.
    """
    base = os.path.basename(model.rstrip('/')) or model
    return LEGACY_ATTACK_NAMES.get(base, f'Paraphrase_{base}')


def resolve_backend(backend: str, model: str) -> str:
    """'auto' -> hf for a hub id (org/name) or a local path, openai otherwise."""
    if backend != 'auto':
        return backend
    return 'hf' if ('/' in model or os.path.isdir(model)) else 'openai'


def load_jsonl(path: str):
    """Read a jsonl file into a list of dicts."""
    records = []
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def write_jsonl(path: str, records):
    """Write a list of dicts as jsonl."""
    with open(path, 'w', encoding='utf-8') as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + '\n')


def token_change_ratio(original: str, attacked: str) -> float:
    """Fraction of the original whitespace tokens that the rewrite did not keep.

    Same alignment-based measure as attacks/text_attack.py, so the paraphrasing
    attack is comparable to the word-level ones.
    """
    orig_tokens = original.split()
    if not orig_tokens:
        return 0.0
    matcher = SequenceMatcher(a=orig_tokens, b=attacked.split(), autojunk=False)
    kept = sum(block.size for block in matcher.get_matching_blocks())
    return (len(orig_tokens) - kept) / len(orig_tokens)


def load_api_key(env_file: str):
    """Load OPENAI_API_KEY from a .env file (existing env vars win)."""

    if env_file and os.path.exists(env_file):
        load_dotenv(env_file, override=False)
    else:
        load_dotenv(override=False)  # fall back to a .env found upwards from cwd

    key = os.environ.get('OPENAI_API_KEY')
    if not key:
        raise RuntimeError(f'OPENAI_API_KEY not set (checked env + {env_file}).')
    return key


def paraphrase(client, model, system_prompt, text, max_completion_tokens,
               temperature, reasoning_effort, retries=5):
    """One paraphrase call, with retries and capability fallbacks."""
    kwargs = dict(
        model=model,
        messages=[
            {'role': 'system', 'content': system_prompt},
            {'role': 'user', 'content': text},
        ],
        max_completion_tokens=max_completion_tokens,
    )
    if temperature is not None:
        kwargs['temperature'] = temperature
    if reasoning_effort:
        kwargs['reasoning_effort'] = reasoning_effort

    last_err = None
    for attempt in range(retries):
        try:
            resp = client.chat.completions.create(**kwargs)
            content = (resp.choices[0].message.content or '').strip()
            if content and content != text.strip():
                return content
            if content:
                # verbatim echo (happens on garbled excerpts): nudge and retry.
                last_err = RuntimeError('model returned the passage unchanged')
                kwargs['messages'] = kwargs['messages'][:2] + [
                    {'role': 'assistant', 'content': content},
                    {'role': 'user', 'content': NUDGE},
                ]
                continue
            # empty completion: the token budget went to reasoning -> raise it.
            last_err = RuntimeError('empty completion')
            kwargs['max_completion_tokens'] *= 2
            continue
        except Exception as e:  # noqa: BLE001
            last_err = e
            msg = str(e)
            # gpt-5* reject a non-default temperature; drop it and retry.
            if 'temperature' in msg and 'temperature' in kwargs:
                kwargs.pop('temperature')
                continue
            # older models do not know reasoning_effort -> retry without it.
            if 'reasoning_effort' in msg and 'reasoning_effort' in kwargs:
                kwargs.pop('reasoning_effort')
                continue
            print(f'  api error (attempt {attempt + 1}): {e}')
            time.sleep(min(10, 2 ** attempt))
    raise RuntimeError(f'paraphrase failed after {retries} retries: {last_err}')


# --------------------------------------------------------------------------
# local transformers backend (gpt-oss-20b, gemma-3-27b-it, ...)
# --------------------------------------------------------------------------

# Three reasoning conventions have to be stripped, or the model's scratchpad
# gets stored as if it were the paraphrase.
#
# gpt-oss (harmony): <|channel|>analysis<|message|>...<|channel|>final<|message|>ANSWER
FINAL_MARKER = '<|channel|>final<|message|>'
SPECIAL_RE = re.compile(r'<\|[^|>]*\|>')
# qwen3 / deepseek-r1: <think>...</think>ANSWER
THINK_RE = re.compile(r'<think>.*?</think>', re.DOTALL)
# gemma-4: <|channel>thought\n...\n<channel|>ANSWER. Note the markers are not
# symmetric ('<|channel>' opens, '<channel|>' closes), so SPECIAL_RE misses them.
GEMMA_THOUGHT_OPEN = '<|channel>'
GEMMA_THOUGHT_CLOSE = '<channel|>'


def _strip_gemma_thought(text):
    """Drop gemma-4 thought channels, exactly as the model's own chat template does.

    Mirrors the `strip_thinking` macro in chat_template.jinja: split on the
    closing marker and, in any piece that opened a channel, keep only what came
    before it. An unclosed channel therefore collapses to '', which is what
    signals run_attack_hf to retry with a bigger budget.
    """
    kept = []
    for part in text.split(GEMMA_THOUGHT_CLOSE):
        kept.append(part.split(GEMMA_THOUGHT_OPEN)[0]
                    if GEMMA_THOUGHT_OPEN in part else part)
    return ''.join(kept)


# a quantised checkpoint loads only if its runtime is installed; say which one
# is missing instead of letting transformers raise deep inside the loader.
QUANT_RUNTIMES = {
    'compressed-tensors': 'compressed_tensors',
    'bitsandbytes': 'bitsandbytes',
    'gptq': 'gptqmodel',
    'awq': 'awq',
}


def _check_quantization_runtime(model_id):
    import importlib.util
    from transformers import AutoConfig
    try:
        quant = getattr(AutoConfig.from_pretrained(model_id),
                        'quantization_config', None)
    except Exception:  # noqa: BLE001 -- loading proper will report the real error
        return
    if not quant:
        return
    method = str(quant.get('quant_method') if isinstance(quant, dict)
                 else getattr(quant, 'quant_method', '')).lower()
    if method == 'mxfp4':
        # `kernels` installed is not enough: transformers pins a narrow range
        # (5.14 wants >=0.15.2,<0.16) and silently DEQUANTISES to bf16 when the
        # installed version is outside it -- gpt-oss-20b then needs ~42 GB, the
        # per-layer dequantise OOMs, and transformers falls back to producing
        # that expert on CPU without moving it back. The run dies much later
        # with "mat2 is on cpu" from _grouped_mm. Ask transformers itself.
        try:
            from transformers.utils import is_kernels_available
            from transformers.utils.import_utils import (KERNELS_MIN_VERSION,
                                                         KERNELS_MAX_VERSION)
        except ImportError:  # transformers too old to gate on kernels at all
            return
        if not is_kernels_available():
            raise RuntimeError(
                f'{model_id} is an mxfp4 checkpoint, but this transformers does '
                f'not accept the installed `kernels` (needs '
                f'>={KERNELS_MIN_VERSION},<{KERNELS_MAX_VERSION}).\n'
                f'  pip install "kernels>={KERNELS_MIN_VERSION},'
                f'<{KERNELS_MAX_VERSION}"\n'
                f'  Without it the model is dequantised to bf16 (~42 GB instead '
                f'of 13.8 GB) and half-lands on CPU.')
        return
    package = QUANT_RUNTIMES.get(method)
    if package and importlib.util.find_spec(package) is None:
        raise RuntimeError(
            f'{model_id} is a {method} checkpoint, but `{package}` is not '
            f'installed in this env. Install it '
            f'(pip install {method.replace("_", "-")}) or point --model at an '
            f'unquantised checkpoint.')


class HFParaphraser:
    """A local chat model used as the paraphraser, batched on GPU.

    Keep the model on ONE GPU. Peer-to-peer copies between the A6000s in this
    cluster are silently corrupt (measured: every ordered pair; each GPU's own
    memory is fine), so an accelerate-sharded model returns fluent garbage
    instead of a paraphrase. --gpus picks the single device.
    """

    def __init__(self, model_id, dtype='auto', device_map='auto',
                 reasoning_effort=None, enable_thinking=False, seed=0):
        import torch
        from transformers import AutoTokenizer, AutoModelForCausalLM

        self.torch = torch
        torch.manual_seed(seed)

        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = 'left'  # required for batched generation

        _check_quantization_runtime(model_id)
        try:
            try:
                self.model = AutoModelForCausalLM.from_pretrained(
                    model_id, dtype=dtype, device_map=device_map)
            except TypeError:  # transformers < 5 spells it torch_dtype
                self.model = AutoModelForCausalLM.from_pretrained(
                    model_id, torch_dtype=dtype, device_map=device_map)
        except torch.cuda.OutOfMemoryError as e:
            free, total = torch.cuda.mem_get_info()
            raise RuntimeError(
                f'{model_id} does not fit on the GPU you asked for '
                f'({free / 1e9:.1f} GB free of {total / 1e9:.1f} GB).\n'
                f'  Spreading it over more GPUs is not an option here: peer '
                f'copies between these A6000s are corrupt, so a sharded model '
                f'returns fluent garbage.\n'
                f'  Use a checkpoint that fits one card instead -- for '
                f'openai/gpt-oss-20b, `pip install kernels>=0.12.0` keeps it in '
                f'MXFP4 (13.8 GB) rather than dequantising to bf16 (~42 GB).'
            ) from e
        self.model.eval()
        self.device = getattr(self.model, 'device', None) or \
            next(self.model.parameters()).device

        placement = set((getattr(self.model, 'hf_device_map', None) or {'': 0}).values())
        self.placement = placement
        if len({d for d in placement if isinstance(d, int) or str(d).startswith('cuda')}) > 1:
            print(f'  !! model is SHARDED over {sorted(map(str, placement))} -- '
                  'peer copies on this cluster are corrupt, the output will be '
                  'garbage. Give one GPU with --gpus, or use a smaller model.')
        elif 'cpu' in {str(d) for d in placement} or 'disk' in {str(d) for d in placement}:
            print(f'  !! part of the model was offloaded ({sorted(map(str, placement))}) '
                  '-- it does not fit this GPU; generation will be very slow.')

        # probe the chat template once instead of guessing per family:
        # gemma-2 has no system role, only gpt-oss knows reasoning_effort,
        # only qwen3-style templates know enable_thinking.
        self.use_system = self._template_ok(system=True)
        self.template_kwargs = {}
        if reasoning_effort and self._template_reads('reasoning_effort',
                                                     reasoning_effort):
            self.template_kwargs['reasoning_effort'] = reasoning_effort
        # Qwen3 (and DeepSeek-R1 distills) default to enable_thinking=True: the
        # reply opens with a <think> block that routinely runs past
        # max_new_tokens, so _clean returns '' and the whole batch is retried
        # with double the budget -- slow, and the paraphrase never gets better
        # for it. Switch thinking off at the template unless asked for.
        if self._template_reads('enable_thinking', enable_thinking):
            self.template_kwargs['enable_thinking'] = enable_thinking
        self.reasoning_effort = self.template_kwargs.get('reasoning_effort')
        self.enable_thinking = self.template_kwargs.get('enable_thinking')

    def _render_probe(self, system, **flags):
        messages = [{'role': 'user', 'content': 'x'}]
        if system:
            messages.insert(0, {'role': 'system', 'content': 'x'})
        return self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True, **flags)

    def _template_ok(self, system, **flags):
        try:
            self._render_probe(system, **flags)
            return True
        except Exception:  # noqa: BLE001
            return False

    def _template_reads(self, key, value):
        """Does the chat template actually act on `key=value`?

        Jinja silently ignores template variables it never mentions, so a
        successful render proves nothing. Render with the flag and without it
        and compare: only a template that reads the flag produces a different
        prompt, and only then is it worth sending.
        """
        try:
            with_flag = self._render_probe(self.use_system, **{key: value})
            without = self._render_probe(self.use_system)
        except Exception:  # noqa: BLE001
            return False
        return with_flag != without

    def render(self, system_prompt, text, echo=None):
        """Chat-template a paraphrase request; `echo` re-asks after a verbatim reply."""
        if self.use_system:
            messages = [{'role': 'system', 'content': system_prompt},
                        {'role': 'user', 'content': text}]
        else:
            messages = [{'role': 'user', 'content': f'{system_prompt}\n\n{text}'}]
        if echo:
            messages += [{'role': 'assistant', 'content': echo},
                         {'role': 'user', 'content': NUDGE}]
        kwargs = {'tokenize': False, 'add_generation_prompt': True,
                  **self.template_kwargs}
        return self.tokenizer.apply_chat_template(messages, **kwargs)

    def _clean(self, text):
        """Reply text -> paraphrase ('' when the model never reached its answer).

        An unterminated reasoning block means the budget ran out before the
        answer; '' makes run_attack_hf double max_new_tokens and retry, which is
        the right response instead of storing the model's scratchpad as if it
        were the paraphrase.
        """
        if FINAL_MARKER in text:
            text = text.rsplit(FINAL_MARKER, 1)[-1]
        elif '<|channel|>' in text:
            return ''  # ran out of budget inside the analysis channel
        text = _strip_gemma_thought(text)
        text = THINK_RE.sub('', text)
        if '<think>' in text:
            return ''  # opened a think block it never closed
        text = SPECIAL_RE.sub('', text)
        for token in self.tokenizer.all_special_tokens:
            text = text.replace(token, '')
        return text.strip()

    def generate(self, prompts, max_new_tokens, temperature, top_p):
        # the template already emits BOS -> add_special_tokens=False (a second
        # BOS is exactly what garbles gemma, cf. baseline_ppl_vllm.py).
        enc = self.tokenizer(prompts, return_tensors='pt', padding=True,
                             add_special_tokens=False).to(self.device)
        with self.torch.inference_mode():
            out = self.model.generate(
                **enc,
                max_new_tokens=max_new_tokens,
                do_sample=temperature > 0,
                temperature=temperature if temperature > 0 else None,
                top_p=top_p if temperature > 0 else None,
                pad_token_id=self.tokenizer.pad_token_id)
        gen = out[:, enc['input_ids'].shape[1]:]
        return [self._clean(self.tokenizer.decode(g, skip_special_tokens=False))
                for g in gen]


def run_attack_hf(records, engine, args, system_prompt):
    """Paraphrase args.field of every record with a local model, in batches."""
    texts = [record.get(args.field) or '' for record in records]
    results = [text if not text.strip() else None for text in texts]
    echoes = {}  # index -> last verbatim reply, fed back as the nudge
    pending = [i for i, text in enumerate(texts) if text.strip()]
    n = len(pending)
    max_new = args.max_new_tokens or args.max_completion_tokens

    for attempt in range(args.retries):
        if not pending:
            break
        if attempt:
            print(f'  retry {attempt}: {len(pending)} records '
                  f'(max_new_tokens={max_new})', flush=True)
        saw_empty = False
        for start in range(0, len(pending), args.batch_size):
            chunk = pending[start:start + args.batch_size]
            prompts = [engine.render(system_prompt, texts[i], echoes.get(i))
                       for i in chunk]
            outputs = engine.generate(prompts, max_new, args.temperature, args.top_p)
            for i, output in zip(chunk, outputs):
                if output and output != texts[i].strip():
                    results[i] = output
                    echoes.pop(i, None)
                elif output:
                    echoes[i] = output  # verbatim echo -> nudge on the next pass
                else:
                    saw_empty = True
                    echoes.pop(i, None)
            n_done = sum(1 for i, r in enumerate(results)
                         if r is not None and texts[i].strip())
            print(f'  pass {attempt}: {start + len(chunk)}/{len(pending)} '
                  f'sent, {n_done}/{n} paraphrased', flush=True)
        pending = [i for i in pending if results[i] is None]
        if saw_empty:
            # the budget went to reasoning / a long rewrite -> raise it
            max_new *= 2

    attacked_records = []
    for i, record in enumerate(records):
        failed = results[i] is None
        new_record = dict(record)
        new_record['attacked_text'] = texts[i] if failed else results[i]
        new_record['attack'] = args.attack_name
        new_record['attack_model'] = args.model
        if failed:
            print(f'  [idx {record.get("index", i)}] FAILED after {args.retries} passes')
            new_record['attack_failed'] = True
        attacked_records.append(new_record)

    return _summarise(attacked_records, args)


def _summarise(attacked_records, args):
    ratios = [token_change_ratio(r.get(args.field) or '', r['attacked_text'])
              for r in attacked_records
              if (r.get(args.field) or '').strip() and not r.get('attack_failed')]
    mean_ratio = sum(ratios) / len(ratios) if ratios else 0.0
    n_failed = sum(1 for r in attacked_records if r.get('attack_failed'))
    return attacked_records, mean_ratio, n_failed


def run_attack(records, client, args, system_prompt):
    """Paraphrase args.field of every record, in parallel."""
    n = len(records)
    done = [0]

    def work(item):
        i, record = item
        text = record.get(args.field) or ''
        if not text.strip():
            attacked_text, failed = text, False
        else:
            try:
                attacked_text = paraphrase(
                    client, args.model, system_prompt, text,
                    args.max_completion_tokens, args.temperature,
                    args.reasoning_effort, args.retries)
                failed = False
            except Exception as e:  # noqa: BLE001
                # keep the record; mark it so detection can exclude it
                print(f'  [idx {record.get("index", i)}] FAILED: {e}')
                attacked_text, failed = text, True

        new_record = dict(record)
        new_record['attacked_text'] = attacked_text
        new_record['attack'] = args.attack_name
        new_record['attack_model'] = args.model
        if failed:
            new_record['attack_failed'] = True

        done[0] += 1
        if done[0] % 10 == 0 or done[0] == n:
            print(f'  {done[0]}/{n}', flush=True)
        return new_record

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        attacked_records = list(pool.map(work, enumerate(records)))

    return _summarise(attacked_records, args)


def main():
    parser = argparse.ArgumentParser(description='Paraphrasing attack on watermarked text.')
    parser.add_argument('--input', type=str,
                        default=os.path.join(PROJECT_ROOT, 'outputs', 'spanwm_v7_c4_n200.jsonl'),
                        help='input jsonl produced by spanwm_embed*.py')
    parser.add_argument('--output_dir', type=str,
                        default=os.path.join(PROJECT_ROOT, 'outputs', 'attacked'),
                        help='directory for the attacked jsonl file')
    parser.add_argument('--output', type=str, default=None,
                        help='explicit output path (default: <stem>_GPTParaphrase.jsonl)')
    parser.add_argument('--field', type=str, default='watermarked_text',
                        help='field of the input jsonl to paraphrase')
    parser.add_argument('--model', type=str, default=DEFAULT_MODEL,
                        help='paraphraser: an OpenAI model name (gpt-5-mini) or '
                             'a HF repo id (openai/gpt-oss-20b, '
                             'google/gemma-3-27b-it, Qwen/Qwen3-14B)')
    parser.add_argument('--backend', type=str, default='auto',
                        choices=['auto', 'openai', 'hf'],
                        help='auto = hf when --model looks like a HF repo id')
    parser.add_argument('--attack_name', type=str, default=None,
                        help='tag written to the "attack" field and the output '
                             'filename (default: derived from --model)')
    parser.add_argument('--prompt_file', type=str, default=DEFAULT_PROMPT_FILE,
                        help='text file with the paraphrasing instruction (system message)')
    parser.add_argument('--env_file', type=str,
                        default=os.path.join(PROJECT_ROOT, '.env'),
                        help='.env file holding OPENAI_API_KEY')
    parser.add_argument('--temperature', type=float, default=None,
                        help='sampling temperature (omitted by default; gpt-5* reject it)')
    parser.add_argument('--reasoning_effort', type=str, default='low',
                        choices=['minimal', 'low', 'medium', 'high', 'none'],
                        help='reasoning effort for gpt-5* models ("none" to omit)')
    parser.add_argument('--max_completion_tokens', type=int, default=2000)
    parser.add_argument('--retries', type=int, default=5)
    parser.add_argument('--workers', type=int, default=8,
                        help='parallel API calls (openai backend)')
    parser.add_argument('--limit', type=int, default=None,
                        help='only attack the first N records (smoke test)')
    # --- hf backend only ---
    parser.add_argument('--batch_size', type=int, default=8,
                        help='records generated per forward pass (hf backend)')
    parser.add_argument('--max_new_tokens', type=int, default=None,
                        help='generation budget (hf backend; default: '
                             '--max_completion_tokens)')
    parser.add_argument('--top_p', type=float, default=0.95,
                        help='nucleus sampling (hf backend)')
    parser.add_argument('--enable_thinking', action='store_true',
                        help='let a qwen3-style template emit its <think> block '
                             '(hf backend). Off by default: paraphrasing needs '
                             'no reasoning and the block eats the generation '
                             'budget. Raise --max_new_tokens if you turn it on.')
    parser.add_argument('--dtype', type=str, default='auto',
                        help='torch dtype for the local model')
    parser.add_argument('--gpus', type=str, default=None,
                        help='GPU index the local model runs on, e.g. "2" '
                             '(sets CUDA_VISIBLE_DEVICES). Pass ONE index: '
                             'peer-to-peer copies between this cluster\'s '
                             'A6000s are corrupt, so a sharded model generates '
                             'garbage.')
    parser.add_argument('--device_map', type=str, default='auto',
                        help='accelerate device map for the local model')
    parser.add_argument('--seed', type=int, default=0,
                        help='sampling seed (hf backend)')
    args = parser.parse_args()

    if args.reasoning_effort == 'none':
        args.reasoning_effort = None
    args.backend = resolve_backend(args.backend, args.model)
    args.attack_name = args.attack_name or attack_name_for(args.model)

    client = None
    if args.backend == 'openai':
        load_api_key(args.env_file)
        from openai import OpenAI
        client = OpenAI()
    else:
        # local models need a temperature; the API path omits it (gpt-5* reject it)
        if args.temperature is None:
            args.temperature = 0.7
        if args.gpus:
            # must happen before torch is imported (HFParaphraser imports it)
            os.environ['CUDA_VISIBLE_DEVICES'] = args.gpus
            n_gpus = len([g for g in args.gpus.split(',') if g.strip()])
            if n_gpus > 1:
                print(f'  !! --gpus {args.gpus}: more than one GPU. Peer copies '
                      'on this cluster are corrupt -- a sharded model will '
                      'generate garbage.')
            elif args.device_map == 'auto':
                # "--gpus N" means the WHOLE model goes on GPU N. device_map
                # "auto" instead spills whatever does not fit to CPU, and the
                # run then dies inside the first fused kernel that meets a
                # half-offloaded weight ("mat2 is on cpu" from gpt-oss's
                # _grouped_mm). Pin every module to the one visible device so a
                # model that is too big fails as a plain, readable OOM.
                args.device_map = {'': 0}

    with open(args.prompt_file, 'r', encoding='utf-8') as f:
        system_prompt = f.read().strip()

    records = load_jsonl(args.input)
    if args.limit:
        records = records[:args.limit]
    os.makedirs(args.output_dir, exist_ok=True)

    stem = os.path.splitext(os.path.basename(args.input))[0]
    out_path = args.output or os.path.join(
        args.output_dir, f'{stem}_{args.attack_name}.jsonl')

    print(f'input      : {args.input} ({len(records)} records)')
    print(f'field      : {args.field}')
    print(f'backend    : {args.backend}')
    print(f'model      : {args.model}')
    if args.backend == 'hf':
        print(f'gpus       : {args.gpus or os.environ.get("CUDA_VISIBLE_DEVICES", "all visible")}')
    print(f'attack     : {args.attack_name}')
    print(f'prompt     : {args.prompt_file}')
    print(f'output     : {out_path}\n')

    if args.backend == 'hf':
        engine = HFParaphraser(args.model, dtype=args.dtype,
                               device_map=args.device_map,
                               reasoning_effort=args.reasoning_effort,
                               enable_thinking=args.enable_thinking,
                               seed=args.seed)
        print(f'loaded     : system_role={engine.use_system} '
              f'reasoning_effort={engine.reasoning_effort} '
              f'enable_thinking={engine.enable_thinking}\n', flush=True)
        attacked_records, mean_ratio, n_failed = run_attack_hf(
            records, engine, args, system_prompt)
    else:
        attacked_records, mean_ratio, n_failed = run_attack(
            records, client, args, system_prompt)
    write_jsonl(out_path, attacked_records)

    print(f'\n{args.attack_name:<28} measured token change={mean_ratio:.3f}  '
          f'failed={n_failed}/{len(records)}  -> {os.path.basename(out_path)}')


if __name__ == '__main__':
    main()
