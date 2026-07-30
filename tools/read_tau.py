"""Print the calibrated tau for a (model_key, dataset), or nothing if absent."""
import json, sys
try:
    stats = json.load(open("outputs/entropy_stats.json"))
    print(stats[f"{sys.argv[1]}_{sys.argv[2]}"]["tau"])
except Exception:
    print("")
