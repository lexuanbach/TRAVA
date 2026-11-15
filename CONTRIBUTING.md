# Contributing

For a code or analysis change, describe the affected analysis and the expected behavioral difference. Include the command and input snapshot used to check it.

```bash
python -m pytest tests -q
python scripts/verify_snapshot.py
```

`verify_snapshot.py` detects intentional edits as well as accidental changes. Keep the original snapshot manifest with the baseline and record a new version for any new release; do not silently replace frozen scientific results.

Run analyses through `scripts/reproduce.py` in a new output directory. Keep generated runs, downloaded weights and credentials out of commits. Do not edit cached answers or human labels to make a result agree. If a result changes, explain the changed population, estimator or inputs and retain the previous snapshot for comparison.

Report issues with the relevant command, Python/package versions, stage ID and a small non-sensitive example. Contact details are in the README.
