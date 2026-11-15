"""Shared helpers for the E0 measurement-replay scripts (analysis step E0).

Nothing here edits a pipeline module on disk. Existing scripts are imported and
their module-level names are monkeypatched for the duration of one call, then
restored. Outputs go only under results/analysis/.
"""
from __future__ import annotations

import contextlib
import hashlib
import io
import json
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PIPE = ROOT / "pipeline"
ANNOT = ROOT / "annotation"
PROC = ROOT / "data/processed"
RES = ROOT / "results"
OUT = RES / "analysis"
RAW = OUT / "e0_raw"

for p in (str(PIPE), str(ANNOT)):
    if p not in sys.path:
        sys.path.insert(0, p)
sys.dont_write_bytecode = True

import metrics  # noqa: E402

_ORIG_GOLD_IN_TEXT = metrics.gold_in_text

# The generators render every passage as  text[:DOC_CHARS]  with DOC_CHARS=600
# (local_generate.py, generate.py, gen_gguf_ranked.py, local_generate_gguf.py,
# gen_openai.py). The visible-text label applies the repository's own
# gold_in_text (SQuAD normalisation, substring test) to exactly that prefix.
GEN_DOC_CHARS = 600


def window_fn(spec):
    """Return a gold_in_text replacement that looks only at a text window.

    spec: None   -> full stored text (the original behaviour)
          int n  -> raw text[:n]
          "700nl"-> text.replace("\\n", " ")[:700]  (build_annotation_pack.py)
          "700ws"-> " ".join(text.split())[:700]    (build_masked_pack.py)
    """
    def f(gold, text):
        text = text or ""
        if spec is None:
            t = text
        elif spec == "700nl":
            t = text.replace("\n", " ")[:700]
        elif spec == "700ws":
            t = " ".join(text.split())[:700]
        else:
            t = text[: int(spec)]
        return _ORIG_GOLD_IN_TEXT(gold, t)
    f.window = spec
    return f


@contextlib.contextmanager
def patched(module, **attrs):
    old = {k: getattr(module, k) for k in attrs}
    try:
        for k, v in attrs.items():
            setattr(module, k, v)
        yield
    finally:
        for k, v in old.items():
            setattr(module, k, v)


def run_main(module, argv, log_path=None, **patches):
    """Run module.main() with sys.argv=argv and module attributes patched."""
    saved_argv = sys.argv
    buf = io.StringIO()
    t0 = time.time()
    try:
        sys.argv = [module.__name__ + ".py"] + [str(a) for a in argv]
        with patched(module, **patches), contextlib.redirect_stdout(buf):
            rc = module.main()
    finally:
        sys.argv = saved_argv
    if log_path:
        Path(log_path).parent.mkdir(parents=True, exist_ok=True)
        Path(log_path).write_text(buf.getvalue())
    return rc, round(time.time() - t0, 1)


def deep_diff(stored, rerun, path="", out=None, tol=0.0):
    """List of (path, stored, rerun) where the two JSON trees differ."""
    if out is None:
        out = []
    if isinstance(stored, dict) and isinstance(rerun, dict):
        for k in sorted(set(stored) | set(rerun), key=str):
            if k not in stored or k not in rerun:
                out.append((f"{path}/{k}", stored.get(k, "<missing>"),
                            rerun.get(k, "<missing>")))
            else:
                deep_diff(stored[k], rerun[k], f"{path}/{k}", out, tol)
    elif isinstance(stored, list) and isinstance(rerun, list):
        if len(stored) != len(rerun):
            out.append((path, f"len {len(stored)}", f"len {len(rerun)}"))
        else:
            for i, (a, b) in enumerate(zip(stored, rerun)):
                deep_diff(a, b, f"{path}[{i}]", out, tol)
    elif isinstance(stored, (int, float)) and isinstance(rerun, (int, float)) \
            and not isinstance(stored, bool) and not isinstance(rerun, bool):
        if abs(stored - rerun) > tol:
            out.append((path, stored, rerun))
    elif stored != rerun:
        out.append((path, stored, rerun))
    return out


def repro_record(stored, rerun, what):
    d = deep_diff(stored, rerun)
    return {"compared": what, "reproduced_exactly": not d,
            "n_mismatches": len(d),
            "first_mismatches": [{"path": p, "stored": s, "rerun": r}
                                 for p, s, r in d[:15]]}


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def git_head():
    try:
        return subprocess.check_output(["git", "-C", str(ROOT), "rev-parse", "HEAD"],
                                       text=True).strip()
    except Exception:
        return None


def meta(extra_inputs=()):
    files = [PROC / "questions.jsonl", PROC / "corpus.jsonl"] + list(extra_inputs)
    return {
        "generated": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "git_head": git_head(),
        "git_worktree_note": "working tree has uncommitted edits to other files; "
                             "pipeline modules used here were imported unmodified "
                             "and patched in memory only",
        "python": sys.version.split()[0],
        "input_sha256": {str(Path(f).relative_to(ROOT)): sha256(f)
                         for f in files if Path(f).is_file()},
    }


def load_json(p):
    return json.loads(Path(p).read_text())


def dump(obj, p):
    Path(p).parent.mkdir(parents=True, exist_ok=True)
    Path(p).write_text(json.dumps(obj, indent=2) + "\n", encoding="utf-8")
