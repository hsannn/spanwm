"""Harvest finished scale-up cells into the reporting table.

Reads the per-cell score files pulled from the cluster and emits one row per
(scheme, model, dataset) in the requested column order:

    AUROC | TPR@10% | TPR@5% | TPR@1% | TPR@0.1% | z-score | p-value

Metrics are recomputed from the stored per-sample scores rather than trusted
from the logs, so a row can never disagree with the data behind it. Ranking is
by -log10(exact binomial p), matching every other table in this project;
positives are watermarked texts, negatives the paired unwatermarked ones.

    python tools/harvest_scaleup.py <dir with outputs/*.scores.json>
"""

import glob
import json
import math
import os
import re
import sys


def roc(y, s):
    """AUROC and TPR at target FPRs without sklearn (trapezoid over the ROC)."""
    order = sorted(zip(s, y), key=lambda t: -t[0])
    P, N = sum(y), len(y) - sum(y)
    if P == 0 or N == 0:
        return float("nan"), {}
    tp = fp = 0
    pts = [(0.0, 0.0)]
    prev = None
    for sc, lab in order:
        if prev is not None and sc != prev:
            pts.append((fp / N, tp / P))
        tp += lab
        fp += 1 - lab
        prev = sc
    pts.append((fp / N, tp / P))
    auroc = sum((pts[i + 1][0] - pts[i][0]) * (pts[i + 1][1] + pts[i][1]) / 2
                for i in range(len(pts) - 1))

    def at(f):
        lo = (0.0, 0.0)
        for x, yv in pts:
            if x <= f:
                lo = (x, yv)
            else:
                if x == lo[0]:
                    return lo[1]
                return lo[1] + (yv - lo[1]) * (f - lo[0]) / (x - lo[0])
        return lo[1]

    return auroc, {f: at(f) for f in (0.10, 0.05, 0.01, 0.001)}


def cell_metrics(scores_path):
    d = json.load(open(scores_path))
    def sc(rs):
        return [-math.log10(max(r["p"], 1e-300)) if r and r.get("p") is not None
                else -1.0 for r in rs]
    y = [1] * len(d["pos"]) + [0] * len(d["neg"])
    a, t = roc(y, sc(d["pos"]) + sc(d["neg"]))
    zs = [r["z"] for r in d["pos"] if r]
    ps = [r["p"] for r in d["pos"] if r and r.get("p") is not None]
    return {"auroc": a, "t10": t.get(0.10), "t5": t.get(0.05),
            "t1": t.get(0.01), "t01": t.get(0.001),
            "z": sum(zs) / len(zs) if zs else float("nan"),
            "p": sum(ps) / len(ps) if ps else float("nan")}


SCHEME = {"sparkr": "SpARK-R soft-fix", "sweet": "SWEET", "ie": "IE"}


def parse_name(base):
    m = re.match(r"(sparkr_softfix|sweet_tau|ie_tau)_"
                 r"(llama3\.2-3b|llama3\.1-8b|qwen3-4b|qwen3-8b|gemma-4-12b)_"
                 r"(c4|cnn_dailymail|cnn|wmt16_de_en|wmt16)_n\d+", base)
    if not m:
        return None
    scheme, model, ds = m.groups()
    key = scheme.split("_")[0]
    return SCHEME.get(key, key), model, ds, False


def main():
    root = sys.argv[1] if len(sys.argv) > 1 else "."
    rows = []
    for p in sorted(glob.glob(os.path.join(root, "**", "*.unwatermarked.scores.json"),
                              recursive=True)):
        if os.sep + "_legacy" + os.sep in p:
            continue
        base = os.path.basename(p).replace(".unwatermarked.scores.json", "")
        parsed = parse_name(base)
        if not parsed:
            continue
        scheme, model, ds, legacy = parsed
        m = cell_metrics(p)
        meta_p = p.replace(".unwatermarked.scores.json", ".meta.json")
        tau = params = None
        if os.path.exists(meta_p):
            md = json.load(open(meta_p))
            tau = md.get("config", {}).get("entropy_threshold")
            params = md.get("n_params_b")
        rows.append((scheme, model, ds, tau, params, m, legacy))

    if not rows:
        print("no finished cells yet")
        return

    order = {"SpARK-R soft-fix": 0, "SWEET": 1, "IE": 2}
    ds_order = {"c4": 0, "cnn_dailymail": 1, "cnn": 2, "wmt16_de_en": 3, "wmt16": 4}
    md_order = {"llama3.2-3b": 0, "llama3.1-8b": 1, "qwen3-4b": 2,
                "qwen3-8b": 3, "gemma-4-12b": 4}
    rows.sort(key=lambda r: (md_order.get(r[1], 9), ds_order.get(r[2], 9),
                             order.get(r[0], 9)))

    print("| scheme | model | params | dataset | τ | AUROC | 10% | 5% | 1% | 0.1% | z-score | p-value |")
    print("|---|---|---|---|---|---|---|---|---|---|---|---|")
    for scheme, model, ds, tau, params, m, legacy in rows:
        mark = " ⚠stale-τ" if legacy else ""
        print(f"| {scheme}{mark} | {model} | {params or '-'}B | {ds} | "
              f"{tau if tau is not None else '—'} | "
              f"{m['auroc']:.4f} | {m['t10']:.3f} | {m['t5']:.3f} | "
              f"{m['t1']:.3f} | **{m['t01']:.3f}** | {m['z']:+.2f} | {m['p']:.2e} |")
    print(f"\n{len(rows)} of 36 cells complete."
          "  ⚠stale-τ = run before per-cell entropy calibration; superseded.")


if __name__ == "__main__":
    main()
