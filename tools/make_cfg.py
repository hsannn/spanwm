"""Emit a per-cell SWEET or IE config with the calibrated tau baked in."""
import json, sys

kind, out, tau = sys.argv[1], sys.argv[2], float(sys.argv[3])
if kind == "sweet":
    c = json.load(open("config/SWEET.json"))
    c["entropy_threshold"] = tau
    c["_comment"] = ("auto-generated: entropy_threshold calibrated for THIS "
                     "(model, dataset) as round(mean next-token entropy, 1)")
else:
    c = json.load(open("config/IE.json"))
    c["entropy_threshold"] = tau
    c["entropy_tagger_path"] = sys.argv[4].rstrip("/") + "/"
    c.pop("output_tag", None)
    c["_comment"] = ("auto-generated: tau calibrated for THIS (model, dataset); "
                     "tagger re-distilled from THIS base model on THIS dataset "
                     "(cross-provenance taggers score at chance)")
json.dump(c, open(out, "w"), indent=4)
