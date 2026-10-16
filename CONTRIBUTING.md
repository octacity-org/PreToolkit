# Contributing

Thanks for helping build Preprocessing Toolkit! This guide covers the conventions every
contribution follows.

## Language

All code, docstrings, comments, error messages, commit messages and pull request
descriptions are written in **English**. Discussion in issues and team chats can be in any
language the participants share.

## Development setup

```bash
git clone https://github.com/octacity-org/PreToolkit.git
cd PreToolkit
pip install -e ".[dev]"
```

scikit-learn is a **development-only** dependency: it is used in tests to prove
compatibility, never imported by the package itself. Runtime dependencies stay limited
to `numpy` and `pandas`.

Before opening a pull request, run:

```bash
ruff format .
ruff check .
pytest
```

CI runs the same commands on Python 3.10 and 3.13 with the latest dependency versions.

## Adding a block

1. Open an issue describing the block and its parameters.
2. Subclass `BasePreprocessor` in its own module (`src/preprocessing_toolkit/<block>.py`)
   and implement `_fit(self, X, y=None)` and `_transform(self, X)`.
3. Follow the block rules:
   - `__init__` stores each argument unchanged in an attribute with the same name and
     does nothing else: no validation, no computation, no `*args` / `**kwargs`.
     `get_params`, `set_params` and `sklearn.base.clone` depend on this.
   - Everything learned in `_fit` goes into attributes ending in `_`
     (for example `mean_`, `categories_`).
   - `_transform` returns a new DataFrame and never modifies its input.
   - If the block creates or renames columns, override `get_feature_names_out`.
4. Add tests in `tests/test_<block>.py`, including leakage (statistics come from the
   training data only) and a run inside `sklearn` `cross_val_score`.
5. Open a pull request; the maintainers review it for interface consistency.
