"""Shared data leakage checks, run the same way for every block and the pipeline.

Data leakage means information from the test set, or from a validation fold,
influencing what a block learns. Every check here works only through the public
block contract (``fit``, ``transform`` and the learned attributes ending in
``_``), so a new block is covered by adding it to ``BLOCKS``.
"""

import math

import numpy as np
import pandas as pd
import pytest
from sklearn.base import clone
from sklearn.dummy import DummyClassifier
from sklearn.model_selection import KFold, cross_validate
from sklearn.pipeline import make_pipeline

import preprocessing_toolkit
from preprocessing_toolkit import (
    BasePreprocessor,
    CategoricalEncoder,
    FeatureScaler,
    MissingValueImputer,
    Pipeline,
)

BLOCKS = [
    MissingValueImputer(strategy="mean"),
    MissingValueImputer(strategy="median"),
    MissingValueImputer(strategy="most_frequent"),
    CategoricalEncoder(method="onehot", handle_unknown="ignore"),
    CategoricalEncoder(method="ordinal", handle_unknown="ignore"),
    FeatureScaler(method="standard"),
    FeatureScaler(method="minmax"),
    # The README quick start chain.
    Pipeline(
        [
            ("imputer", MissingValueImputer(strategy="median")),
            ("encoder", CategoricalEncoder(handle_unknown="ignore")),
            FeatureScaler(),
        ]
    ),
]

SIZES = pd.CategoricalDtype(["S", "M", "L"], ordered=True)


def _block_id(block):
    if isinstance(block, Pipeline):
        return "Pipeline(" + ", ".join(block.named_steps) + ")"
    params = ", ".join(f"{k}={v!r}" for k, v in block.get_params().items())
    return f"{type(block).__name__}({params})"


@pytest.fixture(params=BLOCKS, ids=_block_id)
def block(request):
    # A fresh, unfitted copy for every test, so no state leaks between tests.
    return clone(request.param)


@pytest.fixture
def train():
    """Mixed training data: a numeric column with gaps, an integer column,
    a text column and an ordered category column (categories have no gaps,
    because the encoder asks for imputation first)."""
    return pd.DataFrame(
        {
            "age": [25.0, np.nan, 31.0, 47.0, np.nan, 52.0, 38.0, 29.0],
            "income": [30, 42, 55, 61, 38, 75, 47, 33],
            "city": ["Izmir", "Ankara", "Izmir", "Istanbul"] * 2,
            "size": pd.Categorical(
                ["S", "M", "M", "L", "S", "M", "L", "M"], dtype=SIZES
            ),
        }
    )


@pytest.fixture
def test():
    """Test data unlike the training data: far larger values, gaps, and a
    city never seen in training."""
    return pd.DataFrame(
        {
            "age": [np.nan, 900.0, 1000.0],
            "income": [10_000, -50_000, 99_999],
            "city": ["Bursa", "Izmir", "Bursa"],
            "size": pd.Categorical(["L", "L", "S"], dtype=SIZES),
        },
        index=[100, 101, 102],
    )


@pytest.fixture
def extreme():
    """More test rows with extreme values, appended to the test set to see
    whether the output for the original test rows moves."""
    return pd.DataFrame(
        {
            "age": [1e9, np.nan],
            "income": [10**9, -(10**9)],
            "city": ["Trabzon", "Trabzon"],
            "size": pd.Categorical(["L", "L"], dtype=SIZES),
        },
        index=[200, 201],
    )


# --- helpers ----------------------------------------------------------------


def learned_state(block):
    """Everything a block learned in fit: its attributes ending in ``_``.

    For a pipeline, the learned state of every step is included as well.
    """
    state = {
        name: value
        for name, value in vars(block).items()
        if name.endswith("_") and not name.startswith("_")
    }
    if isinstance(block, Pipeline):
        for name, step in block.named_steps.items():
            state[f"step:{name}"] = learned_state(step)
    return state


def assert_same(actual, expected, path="state"):
    """Compare learned states, allowing float rounding (sums in another order)."""
    if isinstance(expected, dict):
        assert isinstance(actual, dict), path
        assert actual.keys() == expected.keys(), f"{path}: keys differ"
        for key in expected:
            assert_same(actual[key], expected[key], f"{path}[{key!r}]")
    elif isinstance(expected, (list, tuple)):
        assert type(actual) is type(expected), path
        assert len(actual) == len(expected), f"{path}: lengths differ"
        for i, (a, e) in enumerate(zip(actual, expected, strict=True)):
            assert_same(a, e, f"{path}[{i}]")
    elif isinstance(expected, pd.Series):
        pd.testing.assert_series_equal(actual, expected, check_exact=False, rtol=1e-9)
    elif isinstance(expected, (float, np.floating)):
        assert isinstance(actual, (float, np.floating)), path
        both_nan = math.isnan(actual) and math.isnan(expected)
        assert both_nan or math.isclose(actual, expected, rel_tol=1e-9), (
            f"{path}: {actual!r} != {expected!r}"
        )
    else:
        assert actual == expected, f"{path}: {actual!r} != {expected!r}"


def assert_same_output(actual, expected):
    pd.testing.assert_frame_equal(actual, expected, check_exact=False, rtol=1e-9)


def state_differs(a, b):
    try:
        assert_same(a, b)
    except AssertionError:
        return True
    return False


# --- train-only statistics --------------------------------------------------


def test_test_data_would_change_the_learned_state(block, train, test):
    """Guard for the checks below: the test set is different enough that a
    block fitted on it, even partly, would learn something else."""
    fitted_on_train = clone(block).fit(train)
    fitted_with_test = clone(block).fit(pd.concat([train, test]))
    assert state_differs(
        learned_state(fitted_with_test), learned_state(fitted_on_train)
    )


def test_output_depends_only_on_training_statistics(block, train, test, extreme):
    """Transforming a test row gives the same result whatever other rows are
    transformed with it, so no statistic is taken from the test set."""
    block.fit(train)
    alone = block.transform(test)
    with_extreme_rows = block.transform(pd.concat([test, extreme]))
    assert_same_output(with_extreme_rows.loc[test.index], alone)


def test_output_matches_a_block_that_never_saw_the_test_set(block, train, test):
    """Fitting, transforming other data, then transforming the test set gives
    exactly what a block fitted on the training set alone gives."""
    used = block.fit(train)
    used.transform(pd.concat([test, test]))
    fresh = clone(block).fit(train)
    assert_same_output(used.transform(test), fresh.transform(test))


# --- test data changes nothing ----------------------------------------------


@pytest.mark.parametrize("which", ["test", "extreme", "test+extreme"])
def test_transform_leaves_learned_state_unchanged(block, train, test, extreme, which):
    data = {
        "test": test,
        "extreme": extreme,
        "test+extreme": pd.concat([test, extreme]),
    }
    block.fit(train)
    before = learned_state(clone(block).fit(train))
    block.transform(data[which])
    assert_same(learned_state(block), before)


# --- deterministic ----------------------------------------------------------


def test_row_order_does_not_change_what_is_learned(block, train, test):
    shuffled = train.sample(frac=1, random_state=0)
    assert not shuffled.index.equals(train.index)  # the order really changed
    in_order = clone(block).fit(train)
    reordered = clone(block).fit(shuffled)
    assert_same(learned_state(reordered), learned_state(in_order))
    assert_same_output(reordered.transform(test), in_order.transform(test))


# --- refit replaces state ---------------------------------------------------


def test_refit_keeps_nothing_from_the_first_fit(block, train):
    other = pd.DataFrame(
        {
            "age": [5.0, 6.0, np.nan, 8.0],
            "income": [1, 2, 3, 4],
            "city": ["Van", "Van", "Mus", "Van"],
            "colour": ["red", "blue", "red", "red"],  # a column `train` lacks
        }
    )
    refitted = block.fit(train).fit(other)
    fresh = clone(block).fit(other)
    assert_same(learned_state(refitted), learned_state(fresh))
    new_rows = other.iloc[:2].set_index(pd.Index([7, 8]))
    assert_same_output(refitted.transform(new_rows), fresh.transform(new_rows))


# --- cross-validation -------------------------------------------------------


@pytest.fixture
def xy():
    """24 rows like ``train``, with a binary target."""
    rng = np.random.default_rng(0)
    age = rng.normal(40, 10, 24).round(1)
    age[[2, 9, 17]] = np.nan
    X = pd.DataFrame(
        {
            "age": age,
            "income": rng.integers(20, 90, 24),
            "city": ["Izmir", "Ankara", "Istanbul"] * 8,
            "size": pd.Categorical(rng.choice(["S", "M", "L"], 24), dtype=SIZES),
        }
    )
    y = np.tile([0, 1], 12)
    return X, y


def test_cross_validation_fits_each_block_on_its_training_fold_only(block, xy):
    """Inside scikit-learn's cross-validation (the machinery behind
    ``cross_val_score``), each fold's block has exactly the state of a block
    fitted on that fold's training rows."""
    X, y = xy
    folds = list(KFold(n_splits=3, shuffle=True, random_state=0).split(X))
    model = make_pipeline(block, DummyClassifier())
    result = cross_validate(
        model, X, y, cv=folds, return_estimator=True, error_score="raise"
    )
    assert not np.isnan(result["test_score"]).any()
    for (train_rows, _), fitted_model in zip(folds, result["estimator"], strict=True):
        expected = clone(block).fit(X.iloc[train_rows])
        assert_same(learned_state(fitted_model[0]), learned_state(expected))


# --- new blocks are covered -------------------------------------------------


def test_every_exported_block_is_in_the_leakage_suite():
    exported = {
        obj
        for name in preprocessing_toolkit.__all__
        if isinstance(obj := getattr(preprocessing_toolkit, name), type)
        and issubclass(obj, BasePreprocessor)
        and obj is not BasePreprocessor
    }
    covered = {type(block) for block in BLOCKS}
    missing = sorted(cls.__name__ for cls in exported - covered)
    assert not missing, (
        f"Add these blocks to BLOCKS in tests/test_leakage.py: {missing}"
    )
