import os
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

ATTACK_NAME = 'GPTParaphrase'
DEFAULT_MODEL = 'gpt-5-mini'
DEFAULT_PROMPT_FILE = os.path.join(PROJECT_ROOT, 'attacks', 'paraphrase_prompt.txt')


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
                    {'role': 'user', 'content': 'That is the original passage, '
                     'unchanged. Rewrite it now: new wording, new sentence '
                     'structure, same content and same defects.'},
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
        new_record['attack'] = ATTACK_NAME
        new_record['attack_model'] = args.model
        if failed:
            new_record['attack_failed'] = True

        done[0] += 1
        if done[0] % 10 == 0 or done[0] == n:
            print(f'  {done[0]}/{n}', flush=True)
        return new_record

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        attacked_records = list(pool.map(work, enumerate(records)))

    ratios = [token_change_ratio(r.get(args.field) or '', r['attacked_text'])
              for r in attacked_records
              if (r.get(args.field) or '').strip() and not r.get('attack_failed')]
    mean_ratio = sum(ratios) / len(ratios) if ratios else 0.0
    n_failed = sum(1 for r in attacked_records if r.get('attack_failed'))
    return attacked_records, mean_ratio, n_failed


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
                        help='OpenAI model used as the paraphraser')
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
                        help='parallel API calls')
    parser.add_argument('--limit', type=int, default=None,
                        help='only attack the first N records (smoke test)')
    args = parser.parse_args()

    if args.reasoning_effort == 'none':
        args.reasoning_effort = None

    load_api_key(args.env_file)
    from openai import OpenAI
    client = OpenAI()

    with open(args.prompt_file, 'r', encoding='utf-8') as f:
        system_prompt = f.read().strip()

    records = load_jsonl(args.input)
    if args.limit:
        records = records[:args.limit]
    os.makedirs(args.output_dir, exist_ok=True)

    stem = os.path.splitext(os.path.basename(args.input))[0]
    out_path = args.output or os.path.join(
        args.output_dir, f'{stem}_{ATTACK_NAME}.jsonl')

    print(f'input      : {args.input} ({len(records)} records)')
    print(f'field      : {args.field}')
    print(f'model      : {args.model}')
    print(f'prompt     : {args.prompt_file}')
    print(f'output     : {out_path}\n')

    attacked_records, mean_ratio, n_failed = run_attack(
        records, client, args, system_prompt)
    write_jsonl(out_path, attacked_records)

    print(f'\n{ATTACK_NAME:<20} measured token change={mean_ratio:.3f}  '
          f'failed={n_failed}/{len(records)}  -> {os.path.basename(out_path)}')


if __name__ == '__main__':
    main()
