# Preprocessing Toolkit

![CI](https://github.com/octacity-org/PreToolkit/actions/workflows/ci.yml/badge.svg)

Composable, reproducible, and maintainable data preprocessing for machine learning.

Preprocessing Toolkit provides data preparation steps — imputation, encoding, scaling, and more — as self-contained building blocks that you chain into a pipeline. Instead of rewriting the same cleaning code in every project, you compose blocks that keep track of how they transform your data and guarantee the same input is always transformed the same way.

> Part of the [Octacity](https://github.com/octacity-org) community · built by the **.LUCID** team.

> **Status: v0.1 in development.** The core building blocks (`MissingValueImputer`,
> `CategoricalEncoder`, `FeatureScaler`) and `Pipeline` are in place. See
> [examples/quickstart.ipynb](examples/quickstart.ipynb) for a walkthrough notebook.

---

## Why

In almost every ML project, a large share of the work is preparing the data. Yet that code is usually rewritten from scratch, slightly differently each time, and it's easy to fall into traps — most commonly **data leakage**, where information from the test set silently influences training.

scikit-learn gives you the primitives, but leaves reproducibility, data-type awareness, and feature-name tracking up to you. Preprocessing Toolkit is a higher-level layer that makes those three things central:

- **Reproducibility first** — every block records its fitted state and the exact transformation applied, so the same dataset yields identical output across runs and people.
- **Data-type awareness** — blocks know which columns are numeric vs. categoric and route transformations automatically; no manual column slicing.
- **Metadata tracking** — the toolkit tracks how input feature names map to output feature names across the whole chain, even after one-hot encoding or binning.

This is **not** a replacement for scikit-learn — it is designed to work alongside it:
blocks run inside scikit-learn pipelines, `cross_val_score` and `GridSearchCV`,
without depending on it.

---

## Installation

The package is not on PyPI yet. Install it from source:

```bash
git clone https://github.com/octacity-org/PreToolkit.git
cd PreToolkit
pip install -e .
```

For development (adds `pytest`, `ruff`, `scikit-learn` and notebook tooling):

```bash
pip install -e ".[dev]"
```

Requires Python 3.10+. Runtime dependencies: `numpy`, `pandas`. scikit-learn is not
required at runtime, but the blocks work inside scikit-learn pipelines, `clone`,
`cross_val_score` and `GridSearchCV`.

---

## Quick start

The example below creates a small inline dataset with numeric, categorical,
and missing values, and processes it through a complete pipeline:

```python
import numpy as np
import pandas as pd
from preprocessing_toolkit import (
    CategoricalEncoder,
    FeatureScaler,
    MissingValueImputer,
    Pipeline,
)

# Training and test data with numeric, categorical, and missing values
X_train = pd.DataFrame(
    {
        "age": [25.0, np.nan, 35.0, 40.0],
        "city": ["Paris", "London", "Paris", np.nan],
        "salary": [50000.0, 60000.0, np.nan, 80000.0],
    }
)

X_test = pd.DataFrame(
    {
        "age": [30.0, 45.0],
        "city": ["London", "Paris"],
        "salary": [55000.0, 75000.0],
    }
)

pipe = Pipeline(
    [
        MissingValueImputer(strategy="median"),
        CategoricalEncoder(method="onehot"),
        FeatureScaler(method="standard"),
    ]
)

# Fit on training data only, then transform both — no data leakage
X_train_clean = pipe.fit_transform(X_train)
X_test_clean = pipe.transform(X_test)

print(pipe.get_feature_names_out())
# ['age', 'city_London', 'city_Paris', 'salary']

print(X_train_clean.round(2))
#     age  city_London  city_Paris  salary
# 0 -1.61        -0.58        0.58   -1.15
# 1  0.23         1.73       -1.73   -0.23
# 2  0.23        -0.58        0.58   -0.23
# 3  1.15        -0.58        0.58    1.61
```

The key detail: `fit` runs on the training data only. When you call `transform` on the test set, the statistics learned from training are applied — the test set never leaks into the fitting process.

For an end-to-end interactive tutorial including scikit-learn cross-validation and a data leakage demonstration, see [`examples/quickstart.ipynb`](examples/quickstart.ipynb).

---

## How it works

A single abstract base class defines the contract; every block inherits from it and shares the same interface.

```
BasePreprocessor (abstract)
  |-- fit()  .  transform()  .  fit_transform()   -> shared logic
  \-- _fit() .  _transform()                       -> each block implements

Concrete blocks:
  |-- MissingValueImputer   (mean / median / most_frequent)
  |-- CategoricalEncoder    (one-hot / ordinal)
  \-- FeatureScaler         (standard / minmax)

Pipeline (also a BasePreprocessor):
  \-- chains blocks, applied in sequence
```

Because every block shares `fit` / `transform`, the pipeline treats them uniformly, and new blocks can be added without touching existing code. Each block only implements its own `_fit` and `_transform`; the shared logic (fitted-state checks, leakage guard) lives in the base class.

---

## Blocks

`MissingValueImputer` learns fill values from training data and reuses them for
new data. Numeric columns use `mean` (the default), `median` or `most_frequent`;
non-numeric columns always use their most frequent value.

```python
from preprocessing_toolkit import MissingValueImputer

imputer = MissingValueImputer(strategy="median")
X_train_filled = imputer.fit_transform(X_train)
X_test_filled = imputer.transform(X_test)
print(imputer.statistics_)
```

| Block | What it does | Options |
|-------|--------------|---------|
| `MissingValueImputer` | Fills missing values | `mean`, `median`, `most_frequent` |
| `CategoricalEncoder` | Converts categories to numbers | `onehot`, `ordinal`; `handle_unknown="error"` or `"ignore"` |
| `FeatureScaler` | Normalizes value ranges | `standard`, `minmax` |
| `Pipeline` | Chains blocks in sequence | — |

---

## Roadmap

**v0.1 (current focus)** — the three blocks above, a pipeline to chain them, leakage protection, unit tests, and documentation.

Beyond v0.1, each feature is an open contribution opportunity:

- **v0.2** — ordinal encoding with custom order, robust scaler, outlier handling
- **v0.3** — feature binning, interaction terms, full-chain feature-name mapping
- **v0.4+** — target / frequency / binary encoding for high-cardinality columns, pipeline save & load

---

## Contributing

Contributions are welcome — the modular design means each block is largely independent, so you can own one without blocking anyone else.

To add a new block:

1. Open an issue describing the block and its interface.
2. Subclass `BasePreprocessor` and implement `_fit` and `_transform`.
3. Add unit tests and a short usage example.
4. Open a PR; the maintaining team reviews for interface consistency.

See `CONTRIBUTING.md` for details.

---

## License

MIT — see [LICENSE](LICENSE).
