#!/usr/bin/env python3
"""Verify distributed file hashes; local outputs and caches are not inputs."""
import hashlib
import json
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
def verify(root=ROOT):
    manifest = json.loads((root / "snapshot.json").read_text())
    failures = []
    for item in manifest["files"]:
        p = root / item["path"]
        if not p.is_file():
            failures.append({"path": item["path"], "reason": "missing"})
        elif hashlib.sha256(p.read_bytes()).hexdigest() != item["sha256"]:
            failures.append({"path": item["path"], "reason": "hash mismatch"})
    return {"files_checked": len(manifest["files"]), "failures": failures}
if __name__ == "__main__":
    result = verify()
    print(json.dumps(result, indent=2))
    raise SystemExit(bool(result["failures"]))
