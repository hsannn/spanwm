"""Paraphrase-robustness results in the project's standard column format.

    AUROC | TPR@10% | TPR@5% | TPR@1% | TPR@0.1% | z-score | p-value

Metrics are recomputed from the stored per-sample scores (the
`*.attacked.scores.json` sidecars) with the same code path as
tools/harvest_scaleup.py, so a robustness row cannot disagree with a clean row
by construction. SpanWM runs write no sidecar, so those fall back to parsing
the detection log -- flagged with `log` in the source column.

Sample size is printed because it is not constant: some attacked files we were
given stop short of 200, which leaves the low-FPR columns resting on a handful
of negatives.

    python tools/harvest_paraphrase.py [scores_dir] [log ...]
"""

import glob
import json
import math
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from tools.harvest_scaleup import roc  # noqa: E402  (same ROC as the main table)


def from_scores(path):
    d = json.load(open(path))

    def sc(rs):
        return [-math.log10(max(r["p"], 1e-300)) if r and r.get("p") is not None
                else -1.0 for r in rs]

    y = [1] * len(d["pos"]) + [0] * len(d["neg"])
    a, t = roc(y, sc(d["pos"]) + sc(d["neg"]))
    zs = [r["z"] for r in d["pos"] if r]
    ps = [r["p"] for r in d["pos"] if r and r.get("p") is not None]
    return {"n": len(d["pos"]), "auroc": a, "t10": t.get(0.10), "t5": t.get(0.05),
            "t1": t.get(0.01), "t01": t.get(0.001),
            "z": sum(zs) / len(zs) if zs else float("nan"),
            "p": sum(ps) / len(ps) if ps else float("nan"), "src": "scores"}


LOGPAT = {"auroc": r"^AUROC\s+:\s+([\d.]+)", "t10": r"TPR@FPR= 10\.0%\s+:\s+([\d.]+)",
          "t5": r"TPR@FPR=  5\.0%\s+:\s+([\d.]+)", "t1": r"TPR@FPR=  1\.0%\s+:\s+([\d.]+)",
          "t01": r"TPR@FPR=  0\.1%\s+:\s+([\d.]+)", "n": r"samples\s+:\s+(\d+)",
          "z": r"mean z \(pos\)\s+:\s+([+\-\d.]+)", "p": r"mean p \(pos\):\s+(\S+)"}


def from_log(path):
    out, cur = [], None
    for line in open(path):
        m = re.match(r"==== (.+?) ====", line)
        if m:
            cur = {"cell": m.group(1).strip(), "src": "log"}
            out.append(cur)
            continue
        if cur is None:
            continue
        for k, pat in LOGPAT.items():
            hit = re.search(pat, line)
            if hit and k not in cur:
                cur[k] = float(hit.group(1))
    return [r for r in out if "auroc" in r]


def label(path):
    b = os.path.basename(path).replace(".attacked.scores.json", "")
    # model keys arrive in both dotted and underscored spellings
    # (llama3.2-3b and llama3_2_3b name the same run)
    m = re.match(r"(sparkr_softfix|sweet_tau|ie_tau|spanwm_v\d)_(.+?)_"
                 r"(c4|cnn_dailymail|wmt16_de_en|wmt16)_n\d+_(.+)", b, re.I)
    if not m:
        return b, "?"
    sch, mdl, ds, atk = m.groups()
    sch = {"sparkr_softfix": "SpARK-R soft-fix", "sweet_tau": "SWEET",
           "ie_tau": "IE"}.get(sch, sch)
    mdl = re.sub(r"^(llama3)[._-](\d)[._-](\d+b)$", r"\1.\2-\3", mdl.lower())
    mdl = re.sub(r"^(qwen3)[._-](\d+b)", r"\1-\2", mdl)
    ds = "wmt16_de_en" if ds.lower().startswith("wmt16") else ds
    atk = {"Paraphrase_gpt_oss": "Paraphrase_gpt-oss-20b",
           "Paraphrase_gpt_oss_20b": "Paraphrase_gpt-oss-20b"}.get(atk, atk)
    return f"{sch} | {mdl} | {ds}", atk


def main() -> None:
    args = sys.argv[1:] or ["outputs/paraphrase"]
    rows = []
    for a in args:
        if os.path.isdir(a):
            for p in sorted(glob.glob(os.path.join(a, "**", "*.attacked.scores.json"),
                                      recursive=True)):
                cell, atk = label(p)
                rows.append({**from_scores(p), "cell": cell, "atk": atk})
        else:
            for r in from_log(a):
                r.setdefault("atk", "?")
                rows.append(r)

    print("| cell | attack | n | AUROC ↑ | 10% ↑ | 5% ↑ | 1% ↑ | 0.1% ↑ | z-score ↑ | p-value ↓ |")
    print("|---|---|---|---|---|---|---|---|---|---|")
    order = {"GPTParaphrase": 0, "Paraphrase_gemma-4-12B-it": 1, "Paraphrase_gpt-oss-20b": 2,
             "Paraphrase_gpt_oss": 2, "Paraphrase_gpt_oss_20b": 2}
    for r in sorted(rows, key=lambda x: (x.get("cell", ""), order.get(x.get("atk"), 9))):
        f = lambda k, d=3: ("—" if r.get(k) is None or (isinstance(r.get(k), float)
                            and math.isnan(r[k])) else f"{r[k]:.{d}f}")
        print(f"| {r.get('cell','?')} | {r.get('atk','?')} | {r.get('n','—')} | "
              f"{f('auroc',4)} | {f('t10')} | {f('t5')} | {f('t1')} | **{f('t01')}** | "
              f"{r['z']:+.2f} | {r['p']:.2e} |")


if __name__ == "__main__":
    main()
