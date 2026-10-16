import numpy as np
import pandas as pd
import pytest
from sklearn.base import clone
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.utils.validation import check_is_fitted

from preprocessing_toolkit import BasePreprocessor, NotFittedError


class Dummy(BasePreprocessor):
    """Minimal block: subtracts the column means learned in fit."""

    def __init__(self, a=1):
        self.a = a

    def _fit(self, X, y=None):
        self.means_ = X.mean()

    def _transform(self, X):
        return X - self.means_


class Picky(Dummy):
    """Fails inside _fit when the data contains missing values."""

    def _fit(self, X, y=None):
        if X.isna().any().any():
            raise ValueError("missing values")
        super()._fit(X, y)


class NoInit(BasePreprocessor):
    def _fit(self, X, y=None):
        pass

    def _transform(self, X):
        return X


class VarArgs(NoInit):
    def __init__(self, *args):
        self.args = args


@pytest.fixture
def train():
    return pd.DataFrame({"a": [1.0, 2.0, 3.0], "b": [10.0, 20.0, 30.0]})


@pytest.fixture
def xy():
    """~30 rows of numeric data with a balanced binary target."""
    rng = np.random.default_rng(0)
    X = pd.DataFrame(rng.normal(size=(30, 3)), columns=["x1", "x2", "x3"])
    y = np.tile([0, 1], 15)
    return X, y


# --- fit / transform contract ----------------------------------------------


def test_base_cannot_be_instantiated():
    with pytest.raises(TypeError):
        BasePreprocessor()


def test_fit_returns_self_and_learns_columns(train):
    block = Dummy()
    assert block.fit(train) is block
    assert block.feature_names_in_ == ["a", "b"]


def test_transform_uses_statistics_from_fit_only(train):
    test = pd.DataFrame({"a": [100.0], "b": [100.0]})
    out = Dummy().fit(train).transform(test)
    # train means are 2 and 20; the test set's own mean must not be used
    assert out.loc[0, "a"] == 98.0
    assert out.loc[0, "b"] == 80.0


def test_fit_transform_equals_fit_then_transform(train):
    pd.testing.assert_frame_equal(
        Dummy().fit_transform(train), Dummy().fit(train).transform(train)
    )


def test_transform_does_not_modify_input(train):
    original = train.copy()
    Dummy().fit_transform(train)
    pd.testing.assert_frame_equal(train, original)


def test_transform_before_fit_raises(train):
    with pytest.raises(NotFittedError, match="not fitted"):
        Dummy().transform(train)


# --- input validation -------------------------------------------------------


@pytest.mark.parametrize("bad", [[[1, 2]], {"a": [1]}, pd.Series([1, 2])])
def test_non_dataframe_in_fit_raises(bad):
    with pytest.raises(TypeError, match="DataFrame"):
        Dummy().fit(bad)


def test_non_dataframe_in_transform_raises(train):
    block = Dummy().fit(train)
    with pytest.raises(TypeError, match="DataFrame"):
        block.transform(train.to_numpy())


def test_duplicate_column_names_raise():
    X = pd.DataFrame([[1.0, 2.0]], columns=["a", "a"])
    with pytest.raises(ValueError, match="duplicate.*'a'"):
        Dummy().fit(X)


def test_missing_column_raises(train):
    block = Dummy().fit(train)
    with pytest.raises(ValueError, match=r"missing: \['b'\]"):
        block.transform(train[["a"]])


def test_extra_column_raises(train):
    block = Dummy().fit(train)
    with pytest.raises(ValueError, match=r"unexpected: \['c'\]"):
        block.transform(train.assign(c=0.0))


def test_missing_and_extra_reported_in_one_message(train):
    block = Dummy().fit(train)
    with pytest.raises(ValueError, match=r"missing: \['b'\].*unexpected: \['c'\]"):
        block.transform(train[["a"]].assign(c=0.0))


def test_transform_reorders_columns_to_fit_order(train):
    block = Dummy().fit(train)
    assert list(block.transform(train[["b", "a"]]).columns) == ["a", "b"]


# --- refit ------------------------------------------------------------------


def test_refit_replaces_state(train):
    block = Dummy().fit(train)
    block.fit(pd.DataFrame({"c": [5.0, 7.0]}))
    assert block.feature_names_in_ == ["c"]
    assert block.transform(pd.DataFrame({"c": [6.0]})).loc[0, "c"] == 0.0


def test_failed_refit_in_fit_logic_leaves_block_unfitted(train):
    block = Picky().fit(train)
    with pytest.raises(ValueError, match="missing values"):
        block.fit(pd.DataFrame({"a": [1.0, None]}))
    assert not block.__sklearn_is_fitted__()
    with pytest.raises(NotFittedError):
        block.transform(train)


def test_failed_refit_on_bad_input_leaves_block_unfitted(train):
    block = Dummy().fit(train)
    with pytest.raises(TypeError):
        block.fit(train.to_numpy())
    assert not block.__sklearn_is_fitted__()


# --- get_feature_names_out --------------------------------------------------


def test_get_feature_names_out_before_fit_raises():
    with pytest.raises(NotFittedError):
        Dummy().get_feature_names_out()


def test_get_feature_names_out_returns_fit_columns(train):
    block = Dummy().fit(train)
    names = block.get_feature_names_out()
    assert names == ["a", "b"]
    names.append("x")  # returned list is a copy, not internal state
    assert block.feature_names_in_ == ["a", "b"]


def test_get_feature_names_out_accepts_matching_input_features(train):
    block = Dummy().fit(train)
    assert block.get_feature_names_out(np.array(["a", "b"])) == ["a", "b"]


def test_get_feature_names_out_rejects_mismatched_input_features(train):
    block = Dummy().fit(train)
    with pytest.raises(ValueError, match="input_features"):
        block.get_feature_names_out(["b", "a"])


# --- get_params / set_params ------------------------------------------------


def test_get_params_returns_init_arguments():
    assert Dummy(a=5).get_params() == {"a": 5}
    assert Dummy(a=5).get_params(deep=True) == {"a": 5}


def test_get_params_without_init_is_empty():
    assert NoInit().get_params() == {}


def test_get_params_rejects_varargs_init():
    with pytest.raises(TypeError, match=r"\*args"):
        VarArgs().get_params()


def test_set_params_round_trip():
    block = Dummy(a=1)
    assert block.set_params(a=7) is block
    assert block.get_params() == {"a": 7}


def test_set_params_unknown_parameter_raises_and_changes_nothing():
    block = Dummy(a=1)
    with pytest.raises(ValueError, match="'b'"):
        block.set_params(a=2, b=3)
    assert block.a == 1


# --- scikit-learn compatibility ---------------------------------------------


def test_sklearn_is_fitted(train):
    block = Dummy()
    assert not block.__sklearn_is_fitted__()
    block.fit(train)
    assert block.__sklearn_is_fitted__()
    check_is_fitted(block)  # does not raise


def test_sklearn_tags_describe_an_unsupervised_transformer():
    try:
        from sklearn.utils import get_tags
    except ImportError:
        pytest.skip("scikit-learn < 1.6 has no estimator tags API")
    tags = get_tags(Dummy())
    assert tags.estimator_type is None
    assert tags.requires_fit
    assert not tags.target_tags.required
    assert tags.transformer_tags is not None


def test_clone_of_fitted_block_is_unfitted_copy(train):
    block = Dummy(a=3).fit(train)
    copy = clone(block)
    assert copy is not block
    assert copy.get_params() == {"a": 3}
    assert not copy.__sklearn_is_fitted__()
    assert not hasattr(copy, "means_")


def test_cross_val_score_runs_in_sklearn_pipeline(xy):
    X, y = xy
    pipe = make_pipeline(Dummy(), LogisticRegression())
    scores = cross_val_score(pipe, X, y, cv=3, error_score="raise")
    assert len(scores) == 3
    assert not np.isnan(scores).any()


def test_grid_search_sets_block_parameters(xy):
    X, y = xy
    pipe = make_pipeline(Dummy(), LogisticRegression())
    search = GridSearchCV(pipe, {"dummy__a": [1, 2]}, cv=3, error_score="raise")
    search.fit(X, y)
    assert search.best_params_["dummy__a"] in (1, 2)
