"""spaCy + textacy 파싱 결과를 눈으로 보기 위한 probe.

핵심 관전 포인트: textacy가 뽑은 S-V-O triple의 원본 spaCy Token을 추적해
token.idx ~ token.idx+len(token) 로 '원문 character offset'을 역산할 수 있는지.

Usage:
    python parser_testing/spacy_textacy_probe.py                    # 내장 예문
    python parser_testing/spacy_textacy_probe.py "Your sentence."   # 직접 입력
    python parser_testing/spacy_textacy_probe.py --file some.txt
    python parser_testing/spacy_textacy_probe.py --model en_core_web_trf
"""

import argparse
import sys

import spacy
from textacy import extract

DEFAULT_TEXTS = [
    "The quick brown fox jumps over the lazy dog.",
    "Researchers at Anthropic did not release the model weights last year, "
    "but they published a detailed technical report.",
    "Although the committee rejected the initial proposal, the mayor, who had "
    "campaigned on transit reform, promised that the city would fund the "
    "subway extension by 2030.",
]

# ANSI
DIM, BOLD, RESET = "\033[2m", "\033[1m", "\033[0m"
CYAN, YELLOW, GREEN, RED = "\033[36m", "\033[33m", "\033[32m", "\033[31m"


def rule(char="=", width=88):
    print(char * width)


def span_of(tokens):
    """Token 리스트에서 (start_char, end_char) 역산."""
    start = min(t.idx for t in tokens)
    end = max(t.idx + len(t) for t in tokens)
    return start, end


def is_contiguous(tokens, doc):
    """토큰들이 원문에서 연속인지 (사이에 다른 토큰이 끼지 않았는지)."""
    idxs = sorted(t.i for t in tokens)
    return idxs == list(range(idxs[0], idxs[-1] + 1))


def show_tokens(doc):
    print(f"{BOLD}[1] TOKENS{RESET} {DIM}(i / text / idx:end / POS / dep / head){RESET}")
    print(f"  {'i':>3}  {'text':<16} {'char':<12} {'POS':<6} {'dep':<12} head")
    for t in doc:
        print(
            f"  {t.i:>3}  {t.text:<16} "
            f"{f'{t.idx}:{t.idx + len(t)}':<12} "
            f"{t.pos_:<6} {t.dep_:<12} {t.head.text}"
        )
    print()


def show_noun_chunks(doc):
    print(f"{BOLD}[2] NOUN CHUNKS{RESET}")
    chunks = list(doc.noun_chunks)
    if not chunks:
        print(f"  {DIM}(none){RESET}")
    for c in chunks:
        print(f"  [{c.start_char:>3}:{c.end_char:<3}] {c.text!r:<38} root={c.root.text} ({c.root.dep_})")
    print()


def show_entities(doc):
    print(f"{BOLD}[3] ENTITIES{RESET}")
    if not doc.ents:
        print(f"  {DIM}(none){RESET}")
    for e in doc.ents:
        print(f"  [{e.start_char:>3}:{e.end_char:<3}] {e.text!r:<38} {e.label_}")
    print()


def show_triples(doc, text):
    print(f"{BOLD}[4] textacy S-V-O TRIPLES → char offset 역산{RESET}")
    triples = list(extract.subject_verb_object_triples(doc))
    if not triples:
        print(f"  {DIM}(no triples extracted){RESET}\n")
        return

    for n, tri in enumerate(triples, 1):
        print(f"  {CYAN}triple #{n}{RESET}")
        parts = [("SUBJ", tri.subject), ("VERB", tri.verb), ("OBJ ", tri.object)]

        for label, toks in parts:
            toks = list(toks)
            if not toks:
                print(f"    {label}  {DIM}(empty){RESET}")
                continue

            start, end = span_of(toks)
            sliced = text[start:end]
            contig = is_contiguous(toks, doc)

            # 검증: 역산한 오프셋으로 원문을 잘라 다시 토큰 텍스트와 맞는지
            joined = " ".join(t.text for t in toks)
            exact = sliced == joined or sliced.replace("  ", " ") == joined
            mark = f"{GREEN}✓{RESET}" if (contig and exact) else f"{YELLOW}~{RESET}"

            print(
                f"    {label}  {mark} [{start:>3}:{end:<3}] "
                f"slice={sliced!r:<34} tokens={[t.text for t in toks]}"
            )
            if not contig:
                print(
                    f"          {RED}⚠ non-contiguous{RESET} token i={[t.i for t in toks]} "
                    f"— hull이 원문의 다른 토큰을 삼킴"
                )

        # triple 전체를 하나의 span으로 볼 때
        all_toks = [t for _, toks in parts for t in toks]
        s, e = span_of(all_toks)
        print(f"    {DIM}HULL  [{s}:{e}] {text[s:e]!r}{RESET}")
        print()


def show_subtree_spans(doc):
    """참고: 각 동사의 subtree — clause 단위 span 후보."""
    print(f"{BOLD}[5] VERB SUBTREES{RESET} {DIM}(clause 단위 span 후보){RESET}")
    found = False
    for t in doc:
        if t.pos_ in ("VERB", "AUX"):
            sub = list(t.subtree)
            start, end = span_of(sub)
            found = True
            print(f"  {t.text:<12} [{start:>3}:{end:<3}] {doc.text[start:end]!r}")
    if not found:
        print(f"  {DIM}(none){RESET}")
    print()


def analyze(nlp, text):
    rule()
    print(f"{BOLD}TEXT{RESET} ({len(text)} chars)")
    print(f"  {text!r}")
    rule("-")
    doc = nlp(text)
    show_tokens(doc)
    show_noun_chunks(doc)
    show_entities(doc)
    show_triples(doc, text)
    show_subtree_spans(doc)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("text", nargs="*", help="분석할 문장 (없으면 내장 예문)")
    ap.add_argument("--file", help="한 줄에 한 문장씩 담긴 텍스트 파일")
    ap.add_argument("--model", default="en_core_web_sm")
    args = ap.parse_args()

    if args.file:
        with open(args.file, encoding="utf-8") as f:
            texts = [ln.strip() for ln in f if ln.strip()]
    elif args.text:
        texts = [" ".join(args.text)]
    else:
        texts = DEFAULT_TEXTS

    try:
        nlp = spacy.load(args.model)
    except OSError:
        sys.exit(f"모델 없음: python -m spacy download {args.model}")

    print(f"{DIM}spacy={spacy.__version__}  model={args.model}{RESET}")
    for t in texts:
        analyze(nlp, t)


if __name__ == "__main__":
    main()
