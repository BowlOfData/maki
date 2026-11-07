# Contributing

Thanks for helping. Setup:

```bash
pip install -e ".[dev,distributed]"
pytest              # the default run skips tests that need the network
flake8 . --select=E9,F63,F7,F82
black .
```

- Python 3.10+ must keep working; CI runs 3.10–3.13.
- Add a test with every fix or feature. Core must keep only its three runtime dependencies (`requests`, `httpx`,
  `python-dotenv`, plus `urllib3`); anything heavier belongs in an extra.
- All outbound HTTP goes through `maki/connector.py`.
- Note user-visible changes under `[Unreleased]` in `CHANGELOG.md`.
- Security problems: see [SECURITY.md](SECURITY.md), not a public issue.

## Stability

Anything exported from `maki` (`maki/__init__.py`) and documented in `docs/user/` is public API. Until 1.0 a minor
release may change behaviour, always with a changelog entry; deprecated names keep working for at least one minor
release and emit a `DeprecationWarning`.
