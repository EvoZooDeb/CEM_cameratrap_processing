# Repository Guidelines

## Project Structure & Module Organization
- Source: `felis.py` (entry) and `felis_cli/` (CLI + core logic).
  - `felis_cli/app.py`: CLI bootstrap (via `cliff`).
  - `felis_cli/commands/`: Subcommands (`predict`, `exif`, `aggregate`, `validate`, `run`).
  - `felis_cli/core.py`, `config.py`: Pipeline steps and layered config.
- Config: `./.felis.yml` (local), or env vars `FELIS_*`.
- Data: `raw/` input samples, `results/` outputs, `models/` weights.
- Docs/CI: `README.md`, `DEPLOYMENT.md`, `.github/workflows/`.

## Build, Test, and Development Commands
- Install (dev): `pip install -e .` — editable install with console script `felis`.
- Run end‑to‑end: `felis run --config .felis.yml --validate --no-show`.
- Run steps: `felis predict|exif|aggregate|validate --config .felis.yml`.
- Docker (optional): build with `docker build -t felis:local .`; run by mounting config, raw, results, and model as in README examples.

## Coding Style & Naming Conventions
- Language: Python 3.10+. Indent 4 spaces; keep lines < 100 chars.
- Types: prefer type hints and return types; add concise docstrings for public funcs/classes.
- Names: `snake_case` for modules/functions/vars, `CapWords` for classes, constants `UPPER_SNAKE`.
- CLI: add new commands under `felis_cli/commands/` using clear, verb‑first names.

## Testing Guidelines
- Current: no formal test suite. Validate changes by running a small sample:
  - `felis run --config .felis.yml --no-show` and verify CSVs in `results/`.
- Additions welcome: lightweight unit tests for `config` merging and `core` helpers (use `pytest`).

## Commit & Pull Request Guidelines
- Commits: imperative, present tense and concise (e.g., "Add aggregate grouping by burst").
- PRs: include purpose, key changes, example command used for local validation, and before/after notes or small screenshots for `validate` when relevant. Link issues if applicable.
- Scope: keep PRs focused; update README or DEPLOYMENT snippets if flags/paths change.

## Security & Configuration Tips
- Do not commit model weights or raw data; use local paths or Docker volumes.
- Prefer env vars `FELIS_*` or `.felis.yml` for non‑secret config; keep secrets out of VCS.
