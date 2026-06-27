# Contributing

## Quick setup

```bash
git clone https://github.com/pearthink123/judgment.git
cd judgment
pip install -e ".[dev,dashboard]"
python -m pytest tests/ -q
```

## What to work on

Look for issues tagged `good first issue`. Areas that always need love:

- **Tests**: property-based, edge cases, integration with real LLM traces
- **Docs**: examples, tutorials, benchmark reports
- **Adapters**: support for new agent frameworks
- **Performance**: profiling, JIT, early-stopping heuristics

## Code conventions

- Type hints on public APIs, optional on internals
- `np.ndarray` for all numerical arrays — no raw lists in math code
- `Optional[T]` not `T | None` (Python 3.9 compat)
- Tests: one file per module, named `test_<module>.py`

## Before submitting

```bash
python -m pytest tests/ -q        # all 181 tests must pass
python -m pytest tests/ -q --tb=short  # for details on failure
```

## Architecture

```
observation → [Layer 1: CUSUM+Hawkes] → [Layer 2: HMM] → [Layer 3: POMDP] → action
```

See `docs/architecture-redesign.md` for the full design rationale.

## Adding a new adapter

1. Create `integration/<framework>.py`
2. Implement at least one integration pattern (node, callback, wrapper, or context manager)
3. Add tests in `tests/test_integration_<framework>.py`
4. Add a one-liner example to README

See `integration/custom.py` for the simplest adapter template.

## Questions

Open a GitHub issue or discussion. PRs welcome.
