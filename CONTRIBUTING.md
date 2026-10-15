# Contributing

Read [AGENTS.md](AGENTS.md), [README.md](README.md), and the
[Kaggle protocol](docs/kaggle.md) before changing research code.

- Put importable code in `src/room_for_doubt/`, separated by responsibility.
- Keep reusable model definitions outside executable training modules.
- Add lightweight offline checks for fragile dataset, metric, and artifact logic.
- Use Hydra for new experiments and save resolved configuration with results.
- Keep arrays and weights local; keep generated reports and tables in local `results/` or `outputs/`.
- Preserve provenance; mark unknown metadata as missing rather than inferring it
  from ambiguous filenames.

## Before publishing changes

```bash
python -m pytest -q -m 'not integration'
python scripts/check_file_sizes.py --history
```

Enable the included pre-push size check in this clone with:

```bash
git config core.hooksPath .githooks
```

The hook and CI reject files at or above 100,000,000 bytes. `.gitignore` excludes
known artifact formats; the size check also covers permitted JSON and images.
`--history` checks already committed blobs reachable from local refs.

No publication license has been selected. The owner should choose one before
presenting the repository as licensed open-source software.
