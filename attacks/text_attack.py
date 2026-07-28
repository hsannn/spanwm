import sys
import json
import random
import os
import argparse
from difflib import SequenceMatcher

# make `evaluation.tools.text_editor` importable when run from anywhere
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from evaluation.tools.text_editor import WordDeletion, SynonymSubstitution

ATTACKS = {
    'WordDeletion': WordDeletion,
    'SynonymSubstitution': SynonymSubstitution,
}

DEFAULT_RATIOS = [0.1, 0.2, 0.3, 0.4, 0.5]


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
    """Fraction of the original whitespace tokens that were removed / replaced.

    Uses an alignment instead of a positionwise diff: a synonym may itself be a
    multi-word expression ("chase after"), so the token counts do not match.
    """
    orig_tokens = original.split()
    if not orig_tokens:
        return 0.0
    att_tokens = attacked.split()

    # WordDeletion: the attacked text is a subsequence of the original, so a
    # greedy two-pointer scan gives the exact number of surviving tokens.
    i = 0
    for token in orig_tokens:
        if i < len(att_tokens) and att_tokens[i] == token:
            i += 1
    if i == len(att_tokens):
        return (len(orig_tokens) - len(att_tokens)) / len(orig_tokens)

    # SynonymSubstitution: tokens are replaced (possibly by multi-word
    # expressions), so fall back to an alignment -- approximate within ~0.02.
    matcher = SequenceMatcher(a=orig_tokens, b=att_tokens, autojunk=False)
    kept = sum(block.size for block in matcher.get_matching_blocks())
    return (len(orig_tokens) - kept) / len(orig_tokens)


def run_attack(records, attack_name: str, field: str, seed: int,
               ratio: float = None, ratio_range: tuple = None):
    """Apply one attack configuration to every record.

    Either `ratio` (the same ratio for every record) or `ratio_range`
    (a ratio drawn uniformly at random per record) must be given.
    """
    # ratio is re-set per record, so a single editor instance is enough
    editor = ATTACKS[attack_name](ratio=ratio if ratio is not None else ratio_range[0])

    # one seed per configuration -> the whole file is reproducible.
    # the per-record ratios are drawn from a separate stream so that every
    # attack sees the same ratio assignment (the editors consume the global
    # stream at different rates).
    random.seed(seed)
    ratio_rng = random.Random(seed)

    attacked_records = []
    targets = []
    measured = []
    for record in records:
        record_ratio = ratio if ratio is not None else ratio_rng.uniform(*ratio_range)
        editor.ratio = record_ratio

        text = record.get(field) or ''
        attacked_text = editor.edit(text) if text else text

        new_record = dict(record)
        new_record['attacked_text'] = attacked_text
        new_record['attack'] = attack_name
        new_record['attack_ratio'] = record_ratio
        attacked_records.append(new_record)

        targets.append(record_ratio)
        if text:
            measured.append(token_change_ratio(text, attacked_text))

    mean = lambda values: sum(values) / len(values) if values else 0.0
    return attacked_records, mean(targets), mean(measured)


def main():
    parser = argparse.ArgumentParser(description='Word-level attacks on watermarked text.')
    parser.add_argument('--input', type=str,
                        default=os.path.join(PROJECT_ROOT, 'outputs', 'spanwm_v7_c4_n200.jsonl'),
                        help='input jsonl produced by spanwm_embed*.py')
    parser.add_argument('--output_dir', type=str,
                        default=os.path.join(PROJECT_ROOT, 'outputs', 'attacked'),
                        help='directory for the attacked jsonl files')
    parser.add_argument('--attacks', nargs='+', default=list(ATTACKS.keys()),
                        choices=list(ATTACKS.keys()), help='attacks to apply (each one separately)')
    parser.add_argument('--ratios', nargs='+', type=float, default=DEFAULT_RATIOS,
                        help='fixed token ratios to attack, e.g. 0.1 0.2 0.3 0.4 0.5')
    parser.add_argument('--mode', choices=['fixed', 'random', 'both'], default='both',
                        help='fixed = one file per --ratios value; '
                             'random = one file per attack, ratio drawn uniformly '
                             'per sample from --ratio_range')
    parser.add_argument('--ratio_range', nargs=2, type=float, default=[0.1, 0.5],
                        metavar=('LOW', 'HIGH'),
                        help='ratio range for --mode random')
    parser.add_argument('--field', type=str, default='watermarked_text',
                        help='field of the input jsonl to attack')
    parser.add_argument('--seed', type=int, default=42, help='random seed')
    args = parser.parse_args()

    records = load_jsonl(args.input)
    os.makedirs(args.output_dir, exist_ok=True)
    stem = os.path.splitext(os.path.basename(args.input))[0]

    print(f'input      : {args.input} ({len(records)} records)')
    print(f'field      : {args.field}')
    print(f'output dir : {args.output_dir}')
    print(f'seed       : {args.seed}\n')

    low, high = args.ratio_range

    for attack_name in args.attacks:
        if args.mode in ('fixed', 'both'):
            for ratio in args.ratios:
                attacked_records, mean_target, mean_measured = run_attack(
                    records, attack_name, args.field, args.seed, ratio=ratio)

                out_name = f'{stem}_{attack_name}_r{int(round(ratio * 100))}.jsonl'
                write_jsonl(os.path.join(args.output_dir, out_name), attacked_records)

                print(f'{attack_name:<20} ratio={ratio:.2f}       '
                      f'measured={mean_measured:.3f}  -> {out_name}')

        if args.mode in ('random', 'both'):
            attacked_records, mean_target, mean_measured = run_attack(
                records, attack_name, args.field, args.seed, ratio_range=(low, high))

            out_name = (f'{stem}_{attack_name}_'
                        f'rand{int(round(low * 100))}-{int(round(high * 100))}.jsonl')
            write_jsonl(os.path.join(args.output_dir, out_name), attacked_records)

            print(f'{attack_name:<20} ratio=U({low:.2f},{high:.2f})  '
                  f'measured={mean_measured:.3f}  (mean target {mean_target:.3f}) '
                  f'-> {out_name}')


if __name__ == '__main__':
    main()
