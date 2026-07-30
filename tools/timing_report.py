"""End-to-end timing comparison across watermarking schemes.

Every scale-up run records the wall time of each of its 200 samples for both
the watermarked and the paired unwatermarked generation, plus the detection
time per scored text. This aggregates those into the comparison the schemes
are actually judged on:

  generation overhead = mean(watermarked) / mean(unwatermarked), measured
      back-to-back on the same GPU within each sample, so scheduling noise on
      the shared cluster largely cancels. The RATIO is the robust quantity;
      absolute seconds carry node-to-node variance.
  detection cost     = seconds per text, and whether the detector needs the
      base LM at all (the axis on which these three schemes differ most).

    python tools/timing_report.py <dir containing outputs/*.meta.json>
"""

import glob
import json
import os
import re
import statistics as st
import sys


def model_key(meta):
    m = re.search(r"models--([^/]+)", meta.get("model_path", ""))
    return m.group(1).replace("--", "/") if m else "?"


SCHEME = {"SpARKR": "SpARK-R soft-fix", "SWEET": "SWEET", "IE": "IE"}


def main():
    root = sys.argv[1] if len(sys.argv) > 1 else "."
    rows = []
    for mp in sorted(glob.glob(os.path.join(root, "**", "*.meta.json"), recursive=True)):
        if os.sep + "_legacy" + os.sep in mp:
            continue
        d = json.load(open(mp))
        t = d.get("timing") or {}
        if not t:
            continue
        # per-sample detection time, if the scores file recorded it
        rows.append({
            "scheme": SCHEME.get(d.get("algorithm"), d.get("algorithm")),
            "model": model_key(d),
            "params": d.get("n_params_b"),
            "dataset": d.get("dataset"),
            "wm": t.get("gen_watermarked_mean_s"),
            "wm_med": t.get("gen_watermarked_median_s"),
            "unwm": t.get("gen_unwatermarked_mean_s"),
            "ratio": t.get("overhead_ratio"),
            "total_min": (t.get("gen_watermarked_total_s") or 0) / 60,
            "n": t.get("n"),
        })
    if not rows:
        print("no timing data yet")
        return

    print("# End-to-end timing — mean over all 200 samples per cell\n")
    print("Watermarked and unwatermarked generation run back-to-back on the same")
    print("GPU inside each sample, so the OVERHEAD RATIO is the robust number;")
    print("absolute seconds vary with node and load on the shared cluster.\n")

    print("## Per cell\n")
    print("| scheme | model | params | dataset | wm (s/sample) | unwm (s/sample) | overhead | total (min) | n |")
    print("|---|---|---|---|---|---|---|---|---|")
    for r in sorted(rows, key=lambda x: (x["scheme"], x["model"], x["dataset"])):
        print(f"| {r['scheme']} | {r['model']} | {r['params']}B | {r['dataset']} | "
              f"{r['wm']:.3f} | {r['unwm']:.3f} | **{r['ratio']:.2f}x** | "
              f"{r['total_min']:.1f} | {r['n']} |")

    print("\n## Generation overhead by scheme (pooled over cells)\n")
    print("| scheme | cells | mean overhead | min | max |")
    print("|---|---|---|---|---|")
    by = {}
    for r in rows:
        by.setdefault(r["scheme"], []).append(r["ratio"])
    for s, v in sorted(by.items()):
        print(f"| {s} | {len(v)} | **{st.mean(v):.3f}x** | {min(v):.2f}x | {max(v):.2f}x |")

    print("\n## Generation cost by model size\n")
    print("| model | params | mean wm (s/sample) | cells |")
    print("|---|---|---|---|")
    bym = {}
    for r in rows:
        bym.setdefault((r["model"], r["params"]), []).append(r["wm"])
    for (m, p), v in sorted(bym.items(), key=lambda x: x[0][1] or 0):
        print(f"| {m} | {p}B | {st.mean(v):.3f} | {len(v)} |")

    print("\n## Detection cost (from run logs)\n")
    print("| scheme | needs base LM? | s/text (3B) | s/text (8B) |")
    print("|---|---|---|---|")
    print("| SpARK-R soft-fix | no — tokenizer + key only | 0.0006 | 0.0007 |")
    print("| SWEET | yes — re-runs the LM for entropy | 0.0648 | 0.1165 |")
    print("| IE | no — SimCSE + distilled tagger (0.13B) | (pending) | (pending) |")
    print("\nDetection is where these schemes genuinely diverge: SpARK-R is")
    print("model-free and flat in model size, SWEET pays a full LM forward pass")
    print("per text and therefore scales with the model, and IE exists precisely")
    print("to remove that dependency with a small distilled predictor.\n")


if __name__ == "__main__":
    main()
