# Contributing

Thanks for helping build PreToolkit! This guide covers the conventions every
contribution follows.

## Language

All code, docstrings, comments, error messages, commit messages and pull request
descriptions are written in **English**. Discussion in issues and team chats can be in any
language the participants share.

## Development setup

The project uses [uv](https://docs.astral.sh/uv/). `uv sync` creates `.venv` with the
package and the `dev` dependency group, at the exact versions in `uv.lock`:

```bash
git clone https://github.com/octacity-org/PreToolkit.git
cd PreToolkit
uv sync
```

scikit-learn is a **development-only** dependency: it is used in tests to prove
compatibility, never imported by the package itself. Runtime dependencies stay limited
to `numpy` and `pandas`. When you add or change a dependency in `pyproject.toml`, run
`uv lock` and commit the updated `uv.lock`.

Before opening a pull request, run:

```bash
uv run ruff format .
uv run ruff check .
uv run pytest
```

CI runs the same commands on Python 3.10 and 3.13 with the locked dependency versions,
and on Python 3.10 with the oldest supported scikit-learn (1.3).

## Claiming an issue

1. Comment on the issue that you'd like to work on it.
2. Wait until a maintainer assigns the issue to you before you start, so two people
   don't work on the same thing. Maintainers only assign people who asked on the issue.
3. Open a pull request early (a draft is fine) and link it with `Closes #<number>`.
4. If an assigned issue has no activity for 7 days, a maintainer may unassign it so
   someone else can pick it up.

## Adding a block

1. Claim an existing issue for the block (see above), or open one describing the block
   and its parameters.
2. Subclass `BasePreprocessor` in its own module (`src/pretoolkit/<block>.py`)
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
5. Add the block to `BLOCKS` in `tests/test_leakage.py`, so the shared leakage checks
   run on it too. A test fails if an exported block is missing from that list.
6. Open a pull request; the maintainers review it for interface consistency.

## Releasing

Releases are built and published by [OmniShip](https://github.com/octacity-org/omniship).
`workflow.py` defines the pipeline; `omniship.yaml`, `omniship.lock` and
`.github/workflows/{check,build,ship}.yml` are generated from it and must not be edited
by hand. CI fails if they are out of date.

1. In a pull request, set the new `version` in `pyproject.toml`, add a section to
   `CHANGELOG.md`, and regenerate the OmniShip files (the release tag is derived from
   the version):

   ```bash
   uvx --from omniship==0.6.0 omniship generate
   ```

2. After the pull request is merged, tag the merge commit on `main` and push the tag:

   ```bash
   git checkout main && git pull
   git tag v0.1.0
   git push origin v0.1.0
   ```

3. The tag starts the Ship workflow: Check, then Build, then a maintainer approves the
   `release` environment, the wheel is published to PyPI with Trusted Publishing (no
   token is stored), and a GitHub Release with generated notes is created.

A published version can never be replaced on PyPI. If a release is broken, fix it and
publish a new patch version.
