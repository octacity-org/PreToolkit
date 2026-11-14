# Changelog

All notable changes to this project are documented in this file. The format is based
on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the project uses
[semantic versioning](https://semver.org/). Before 1.0, minor versions may change the
public API.

## [0.1.0] - 2026-11-14

The first release: composable preprocessing blocks that learn only from training data,
a pipeline to chain them, and a shared test suite that checks every block for data
leakage.

### Added

- `BasePreprocessor`, the shared block contract: `fit` / `transform` / `fit_transform`
  on pandas DataFrames, fitted-state and column checks, `get_feature_names_out`, and
  scikit-learn compatible `get_params` / `set_params` without depending on
  scikit-learn at runtime.
- `MissingValueImputer`: fills missing values with the training `mean`, `median` or
  `most_frequent` value; non-numeric and boolean columns use the most frequent value.
  ([#2](https://github.com/octacity-org/PreToolkit/issues/2), by @Venius24)
- `CategoricalEncoder`: `onehot` and `ordinal` encoding of text and category columns,
  keeping the declared order of ordered categoricals, with
  `handle_unknown="error"` or `"ignore"`.
  ([#3](https://github.com/octacity-org/PreToolkit/issues/3), by @mehmeterencakmak)
- `FeatureScaler`: `standard` and `minmax` scaling of numeric columns.
  ([#4](https://github.com/octacity-org/PreToolkit/issues/4), by @montionugera)
- `Pipeline`: chains blocks, with named steps, nested `get_params` / `set_params`
  (`<step>__<param>`), step replacement, and feature names passed through every step.
  Works inside scikit-learn pipelines, `clone`, `cross_val_score` and `GridSearchCV`.
  ([#5](https://github.com/octacity-org/PreToolkit/issues/5), by @akifkulll)
- A shared data leakage test suite that runs every block and the pipeline through the
  same checks: training-only statistics, transform leaves the learned state unchanged,
  row order does not matter, refit replaces state, and cross-validation folds see only
  their training rows. ([#6](https://github.com/octacity-org/PreToolkit/issues/6), by
  @akifkulll)
- A runnable README quick start and an `examples/quickstart.ipynb` walkthrough.
  ([#7](https://github.com/octacity-org/PreToolkit/issues/7), by @mertcankaratass)
- Releases built and published with [OmniShip](https://github.com/octacity-org/omniship)
  and PyPI Trusted Publishing.

### Known limitations

- `FeatureScaler` scales every numeric column, including one-hot columns produced by
  `CategoricalEncoder`. Applying a block to selected columns only is planned for 0.2.
- `FeatureScaler` treats only an exactly zero spread as constant; for a nearly constant
  float column (spread around 1e-16) it amplifies rounding noise instead of returning 0.

[0.1.0]: https://github.com/octacity-org/PreToolkit/releases/tag/v0.1.0
