import numpy as np
import pandas as pd
import pytest
from sklearn.base import clone
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.pipeline import make_pipeline

from preprocessing_toolkit import BasePreprocessor, FeatureScaler, NotFittedError


@pytest.fixture
def train():
    return pd.DataFrame({"a": [1.0, 2.0, 3.0], "b": [10, 20, 40]})


@pytest.fixture
def xy():
    """~30 rows of numeric data with a balanced binary target."""
    rng = np.random.default_rng(0)
    X = pd.DataFrame(rng.normal(size=(30, 3)), columns=["x1", "x2", "x3"])
    y = np.tile([0, 1], 15)
    return X, y


# --- construction -----------------------------------------------------------


def test_is_a_base_preprocessor():
    assert isinstance(FeatureScaler(), BasePreprocessor)


def test_default_method_is_standard():
    assert FeatureScaler().get_params() == {"method": "standard"}


def test_init_stores_unknown_method_without_validating():
    assert FeatureScaler(method="bogus").method == "bogus"


def test_unknown_method_raises_in_fit_and_leaves_block_unfitted(train):
    block = FeatureScaler(method="bogus")
    with pytest.raises(ValueError, match="bogus"):
        block.fit(train)
    assert not block.__sklearn_is_fitted__()
    with pytest.raises(NotFittedError):
        block.transform(train)


# --- standard ---------------------------------------------------------------


def test_standard_learns_mean_and_population_std(train):
    block = FeatureScaler().fit(train)
    assert block.center_ == pytest.approx({"a": 2.0, "b": 70 / 3})
    assert block.scale_["a"] == pytest.approx(np.std([1.0, 2.0, 3.0]))
    assert block.scale_["b"] == pytest.approx(np.std([10, 20, 40]))


def test_standard_output_has_zero_mean_and_unit_std(train):
    out = FeatureScaler().fit_transform(train)
    for col in ["a", "b"]:
        assert out[col].mean() == pytest.approx(0.0)
        assert out[col].std(ddof=0) == pytest.approx(1.0)


# --- minmax -----------------------------------------------------------------


def test_minmax_learns_min_and_range(train):
    block = FeatureScaler(method="minmax").fit(train)
    assert block.center_ == {"a": 1.0, "b": 10.0}
    assert block.scale_ == {"a": 2.0, "b": 30.0}


def test_minmax_maps_training_data_to_unit_interval(train):
    out = FeatureScaler(method="minmax").fit_transform(train)
    assert out["a"].tolist() == [0.0, 0.5, 1.0]
    assert out["b"].tolist() == pytest.approx([0.0, 1 / 3, 1.0])


# --- leakage ----------------------------------------------------------------


@pytest.mark.parametrize("method", ["standard", "minmax"])
def test_transform_uses_statistics_from_fit_only(train, method):
    block = FeatureScaler(method=method).fit(train)
    test = pd.DataFrame({"a": [5.0], "b": [100]})
    out = block.transform(test)
    for col in ["a", "b"]:
        expected = (test.loc[0, col] - block.center_[col]) / block.scale_[col]
        assert out.loc[0, col] == pytest.approx(expected)


# --- column handling --------------------------------------------------------


@pytest.mark.parametrize("method", ["standard", "minmax"])
def test_non_numeric_columns_pass_through(method):
    X = pd.DataFrame(
        {
            "num": [1.0, 2.0, 3.0],
            "cat": ["x", "y", "z"],
            "flag": [True, False, True],
        }
    )
    block = FeatureScaler(method=method).fit(X)
    assert list(block.center_) == ["num"]
    assert list(block.scale_) == ["num"]
    out = block.transform(X)
    pd.testing.assert_series_equal(out["cat"], X["cat"])
    pd.testing.assert_series_equal(out["flag"], X["flag"])
    assert list(out.columns) == ["num", "cat", "flag"]


@pytest.mark.parametrize("method", ["standard", "minmax"])
def test_constant_column_gets_scale_one_and_becomes_zero(method):
    X = pd.DataFrame({"c": [4.0, 4.0, 4.0]})
    block = FeatureScaler(method=method).fit(X)
    assert block.scale_ == {"c": 1.0}
    out = block.transform(X)
    assert out["c"].tolist() == [0.0, 0.0, 0.0]


@pytest.mark.parametrize("method", ["standard", "minmax"])
def test_missing_values_ignored_in_fit_and_kept_in_output(method):
    X = pd.DataFrame({"a": [1.0, np.nan, 3.0]})
    block = FeatureScaler(method=method).fit(X)
    assert block.center_["a"] == pytest.approx(2.0 if method == "standard" else 1.0)
    assert block.scale_["a"] == pytest.approx(1.0 if method == "standard" else 2.0)
    out = block.transform(X)
    assert np.isnan(out.loc[1, "a"])
    assert not np.isnan(out.loc[[0, 2], "a"]).any()


@pytest.mark.parametrize("method", ["standard", "minmax"])
def test_transform_returns_new_frame_and_does_not_modify_input(train, method):
    original = train.copy()
    block = FeatureScaler(method=method).fit(train)
    out = block.transform(train)
    assert out is not train
    pd.testing.assert_frame_equal(train, original)


def test_learned_statistics_are_plain_floats(train):
    block = FeatureScaler().fit(train)
    for value in [*block.center_.values(), *block.scale_.values()]:
        assert type(value) is float


# --- refit / clone ----------------------------------------------------------


def test_refit_with_new_method_replaces_state(train):
    block = FeatureScaler().fit(train)
    block.set_params(method="minmax").fit(train)
    assert block.center_ == {"a": 1.0, "b": 10.0}


def test_clone_of_fitted_block_is_unfitted_copy(train):
    block = FeatureScaler(method="minmax").fit(train)
    copy = clone(block)
    assert copy.get_params() == {"method": "minmax"}
    assert not copy.__sklearn_is_fitted__()
    assert not hasattr(copy, "center_")


# --- scikit-learn compatibility ---------------------------------------------


@pytest.mark.parametrize("method", ["standard", "minmax"])
def test_cross_val_score_runs_in_sklearn_pipeline(xy, method):
    X, y = xy
    pipe = make_pipeline(FeatureScaler(method=method), LogisticRegression())
    scores = cross_val_score(pipe, X, y, cv=3, error_score="raise")
    assert len(scores) == 3
    assert not np.isnan(scores).any()


def test_grid_search_tunes_method(xy):
    X, y = xy
    pipe = make_pipeline(FeatureScaler(), LogisticRegression())
    search = GridSearchCV(
        pipe,
        {"featurescaler__method": ["standard", "minmax"]},
        cv=3,
        error_score="raise",
    )
    search.fit(X, y)
    assert search.best_params_["featurescaler__method"] in ("standard", "minmax")
