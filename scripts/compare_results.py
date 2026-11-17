#!/usr/bin/env python3
"""Compare scientific numerical leaves from a cached-analysis replay."""
import argparse
import json
import math
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
IGNORE = {"reporting_correction", "reproduction_of_stored_results", "meta", "provenance", "timing", "timings", "seconds", "elapsed_seconds", "runtime_seconds", "elapsed_s", "runtime_s", "wall_seconds", "wall_s", "created_at", "timestamp", "generated_at", "input_manifest", "hashes", "reproduction", "repro", "timing_s", "timing_s_per_20", "estimated_seconds"}
def numbers(x, path=""):
    out = {}
    if isinstance(x, dict):
        for k, v in x.items():
            if k not in IGNORE and not any(t in k.lower() for t in ["elapsed", "runtime", "timing", "timestamp"]):
                out.update(numbers(v, path + "/" + k.replace("~", "~0").replace("/", "~1")))
    elif isinstance(x, list):
        for i, v in enumerate(x): out.update(numbers(v, path + "/" + str(i)))
    elif isinstance(x, (int, float)) and not isinstance(x, bool): out[path] = x
    return out

def main():
    ap = argparse.ArgumentParser(description=__doc__); ap.add_argument("run", type=Path)
    ap.add_argument("--atol", type=float, default=1e-8); args = ap.parse_args()
    summary = json.loads((args.run / "summary.json").read_text()); records = []
    for stage in summary["stages"]:
        if stage["status"] != "successful": continue
        for name in stage.get("results", []):
            ref = numbers(json.loads((ROOT / name).read_text()))
            got = numbers(json.loads((args.run / "work" / name).read_text()))
            changed = []
            for k in sorted(ref.keys() | got.keys()):
                a, b = ref.get(k), got.get(k)
                if k not in ref or k not in got or not (a == b or (math.isnan(a) and math.isnan(b)) or math.isclose(a, b, abs_tol=args.atol, rel_tol=0)):
                    changed.append({"pointer": k, "reference": a, "recomputed": b})
            records.append({"file": name, "numeric_fields": len(ref), "differences": changed})
    result = {"absolute_tolerance": args.atol, "files": records, "difference_count": sum(len(x["differences"]) for x in records)}
    (args.run / "comparison.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"files": len(records), "numeric_fields": sum(x["numeric_fields"] for x in records), "differences": result["difference_count"]}, indent=2))
    for record in records:
        if record["differences"]:
            print(json.dumps({"file": record["file"], "differences": record["differences"]}, indent=2))
    return int(result["difference_count"] > 0)
if __name__ == "__main__": raise SystemExit(main())
