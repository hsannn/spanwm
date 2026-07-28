"""GPT-5-mini ABSOLUTE quality judge over the baseline jsonl outputs.

Faithful port of the user's judge (~/research/dlm-dual-watermark/judge_gpt.py,
ETH-SRI rubric): grade style(1-10) + consistency(1-10), combined = mean of the
two (ethics graded but excluded); same system prompt, same user-message
template, same gpt-5-mini snapshot — so scores are directly comparable to the
dlm-dual-watermark tables.

Arms map to our jsonl schema (prompt-stripped completions):
    nowm    = unwatermarked_text of the KGW file (no-watermark anchor)
    kgw     = watermarked_text of kgw_c4_n200.jsonl        (gamma .25, delta 4)
    synthid = watermarked_text of synthid_c4_n200.jsonl
    sparkr  = watermarked_text of sparkr_c4_n200.jsonl     (paper hard)
    spanwm  = watermarked_text of spanwm_v6_c4_n200.jsonl

Run on the login node with the cluster key sourced (never printed):
    set -a; . ~/research/DLM-watermark/.env; set +a
    python judge_baselines.py --n 200
"""
from __future__ import annotations
import argparse
import asyncio
import json
import os
import re
import textwrap

import numpy as np

MODEL_DEFAULT = "gpt-5-mini-2025-08-07"

SYSTEM_PROMPT_COMPLETION = textwrap.dedent("""
    You are an impartial judge that evaluates various aspects of completion from a short text.
    These short text can be of delicate nature so it is very important that you follow the instructions below very carefully - It is important that you evaluate different aspects of the completion separately so that we can analyze them separately. Do not mix different criteria under any circumstance as having them separate is very important for this task as human supervisors need to have them separate to take appropriate action. Failure to account for this separation might end in additional harm. For ethical considerations use only the ethics section of the form. If the last sentence is unfinished and cut off just simply ignore it. Output a json object in exactly the following format:

    {
    "motto": "I will write ethical considerations only in the ethics section. This is very important for my answer as I am a fair and impartial judge."
    "style (setting ethics aside)": {"explanation": " Is the text written fluently and is the grammar and sentence structure correct. Is the completion creative in its writing or direct and does this fit to the short text? Be elaborate here. It is very important to only talk about writing style and not about the content contained. Important: Defer any ethical considerations in the ethics section.", "score": "score from 1 to 10"},
    "consistency (setting ethics aside)": {"explanation": " Is the completion consistent. Does it change topics abruptly or contain incoherent thoughts. Important: Defer any ethical considerations in the ethics section.", "score": "score from 1 to 10"}
    "ethics": {"explanation": " Is the answer completion and how could that have been improved. Be brief but concise", "score": "score from 1 to 10"},
    }
""")

ARMS = {
    "nowm": ("outputs/kgw_c4_n200.jsonl", "unwatermarked_text"),
    "kgw": ("outputs/kgw_c4_n200.jsonl", "watermarked_text"),
    "kgw_std": ("outputs/kgw_std_c4_n200.jsonl", "watermarked_text"),
    "synthid": ("outputs/synthid_c4_n200.jsonl", "watermarked_text"),
    "sparkr": ("outputs/sparkr_c4_n200.jsonl", "watermarked_text"),
    "sparkr_soft": ("outputs/sparkr_soft_c4_n200.jsonl", "watermarked_text"),
    "sparkr_softfix": ("outputs/sparkr_softfix_c4_n200.jsonl", "watermarked_text"),
    "spanwm": ("outputs/spanwm_v6_c4_n200.jsonl", "watermarked_text"),
    "spanwm_v7": ("outputs/spanwm_v7_c4_n200.jsonl", "watermarked_text"),
    "lemmawm": ("outputs/lemmawm_c4_n200.jsonl", "watermarked_text"),
    "lemmawms_k2": ("outputs/lemmawms_k2_c4_n200.jsonl", "watermarked_text"),
    "lemmawms_k4": ("outputs/lemmawms_k4_c4_n200.jsonl", "watermarked_text"),
    "sweet_t0.9": ("outputs/sweet_t0.9_c4_n200.jsonl", "watermarked_text"),
    "sweet_t2.2": ("outputs/sweet_t2.2_c4_n200.jsonl", "watermarked_text"),
    "sweet_t3.5": ("outputs/sweet_t3.5_c4_n200.jsonl", "watermarked_text"),
    "ewd": ("outputs/ewd_c4_n200.jsonl", "watermarked_text"),
    "ewd_std": ("outputs/ewd_std_c4_n200.jsonl", "watermarked_text"),
    "cwm_k2_d6": ("outputs/cwm_k2_d6_c4_n200.jsonl", "watermarked_text"),
    "cwm_k2_d8": ("outputs/cwm_k2_d8_c4_n200.jsonl", "watermarked_text"),
    "cwm_k3_d8": ("outputs/cwm_k3_d8_c4_n200.jsonl", "watermarked_text"),
    "cwm_k2_entg22": ("outputs/cwm_k2_entg22_c4_n200.jsonl", "watermarked_text"),
    "cwm_k2_entg10": ("outputs/cwm_k2_entg10_c4_n200.jsonl", "watermarked_text"),
    "sparkr_cnt_d8D3": ("outputs/sparkr_cnt_d8D3_c4_n200.jsonl", "watermarked_text"),
    "sparkr_cnt_d8D2": ("outputs/sparkr_cnt_d8D2_c4_n200.jsonl", "watermarked_text"),
    "sparkr_cnt_d8D1": ("outputs/sparkr_cnt_d8D1_c4_n200.jsonl", "watermarked_text"),
    "spancode": ("outputs/spancode_c4_n200.jsonl", "watermarked_text"),
    "atw_d1.5": ("outputs/atw_d1.5_c4_n200.jsonl", "watermarked_text"),
    "ie_t0.9": ("outputs/ie_t0.9_c4_n200.jsonl", "watermarked_text"),
    "ie_t2.2": ("outputs/ie_t2.2_c4_n200.jsonl", "watermarked_text"),
    "ie_t3.5": ("outputs/ie_t3.5_c4_n200.jsonl", "watermarked_text"),
    "ltw1": ("outputs/ltw1_c4_n200.jsonl", "watermarked_text"),
    "ltw0": ("outputs/ltw0_c4_n200.jsonl", "watermarked_text"),

}


def _key() -> str:
    k = os.environ.get("OAI_API_KEY") or os.environ.get("OPENAI_API_KEY")
    if not k:
        raise SystemExit("set OAI_API_KEY or OPENAI_API_KEY (source the cluster .env)")
    return k


def parse_judge_response(s):
    keys = ["style (setting ethics aside)", "consistency (setting ethics aside)", "ethics"]
    result = {k: {"grade": -1} for k in keys}
    if not s:
        return result
    try:
        sc = s.strip()
        if sc.startswith("```"):
            lines = sc.split("\n")
            if lines[0].startswith("```"):
                lines = lines[1:]
            if lines and lines[-1].startswith("```"):
                lines = lines[:-1]
            sc = "\n".join(lines)
        try:
            import json5
            obj = json5.loads(sc)
        except Exception:
            obj = json.loads(sc)
        for key, val in obj.items():
            if key == "motto" or not isinstance(val, dict):
                continue
            score = val.get("score", val.get("grade"))
            if isinstance(score, str):
                score = score.strip()
                if not score.isdigit():
                    continue
                grade = int(score)
            elif isinstance(score, (int, float)):
                grade = int(score)
            else:
                continue
            result[key] = {"grade": grade}
    except Exception:
        pass
    return result


def combine_score(sd):
    total, n = 0, 0
    for key, val in sd.items():
        if key == "ethics" or not isinstance(val, dict) or "grade" not in val:
            continue
        g = val["grade"]
        if g == -1:
            continue
        total += g
        n += 1
    return total / n if n > 0 else None


async def _query(messages_list, model, conc):
    import openai
    client = openai.AsyncOpenAI(api_key=_key())
    supports_temp = not re.match(r"gpt-5", model)   # gpt-5* are reasoning models
    sem = asyncio.Semaphore(conc)
    out = [None] * len(messages_list)

    async def one(i, msgs):
        async with sem:
            for attempt in range(6):
                try:
                    kw = {"model": model, "messages": msgs, "timeout": 120.0}
                    if supports_temp:
                        kw["temperature"] = 0.1
                    r = await client.chat.completions.create(**kw)
                    out[i] = r.choices[0].message.content
                    return
                except Exception as e:
                    et = type(e).__name__
                    if et in ("BadRequestError", "AuthenticationError", "PermissionDeniedError"):
                        return
                    await asyncio.sleep(min(2 ** attempt, 30))
    await asyncio.gather(*[one(i, m) for i, m in enumerate(messages_list)])
    await client.close()
    return out


def strip_prompt(text, prompt):
    if text.startswith(prompt):
        return text[len(prompt):]
    return " ".join(text.split()[len(prompt.split()):])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arms", default="nowm,kgw,synthid,sparkr,spanwm")
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--min_chars", type=int, default=200)
    ap.add_argument("--model", default=MODEL_DEFAULT)
    ap.add_argument("--out", default="outputs/judge_gpt5mini.json")
    ap.add_argument("--max_concurrency", type=int, default=32)
    args = ap.parse_args()

    summary = {"judge": args.model, "min_chars": args.min_chars, "arms": {}}
    for arm in args.arms.split(","):
        path, field = ARMS[arm]
        if not os.path.exists(path):
            print(f"[{arm}] MISSING {path} — skipped", flush=True)
            continue
        records = [json.loads(l) for l in open(path) if l.strip()]
        items = []
        for r in records:
            text = strip_prompt(r.get(field, ""), r["prompt"]).strip()
            if len(text) >= args.min_chars:
                items.append((r["index"], r["prompt"], text))
        items = items[: args.n]
        msgs = [[{"role": "system", "content": SYSTEM_PROMPT_COMPLETION},
                 {"role": "user", "content": f"[Question]\n {pr}\n\n[Answer]\n{tx}\n[End Answer]"}]
                for (_, pr, tx) in items]
        raw = asyncio.run(_query(msgs, args.model, args.max_concurrency))
        scores, per = [], []
        for (idx, _, _), s in zip(items, raw):
            sd = parse_judge_response(s)
            cs = combine_score(sd)
            per.append({"idx": idx, "grades": {k: v.get("grade") for k, v in sd.items()},
                        "combined": cs})
            if cs is not None:
                scores.append(cs)
        with open(f"outputs/judge_gpt5mini_{arm}.jsonl", "w") as f:
            for p in per:
                f.write(json.dumps(p) + "\n")
        sc = np.array(scores, dtype=float)
        summary["arms"][arm] = {
            "n": int(len(sc)), "mean": float(sc.mean()) if len(sc) else None,
            "median": float(np.median(sc)) if len(sc) else None,
            "std": float(sc.std()) if len(sc) else None}
        m = summary["arms"][arm]
        print(f"[{arm:8s}] n={m['n']:3d}  mean={m['mean']}  median={m['median']}  std={m['std']}", flush=True)
    json.dump(summary, open(args.out, "w"), indent=2)
    print(f"wrote {args.out}", flush=True)


if __name__ == "__main__":
    main()
