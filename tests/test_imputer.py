import numpy as np
import pandas as pd
import pytest
from sklearn.base import clone
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.pipeline import make_pipeline

from preprocessing_toolkit import MissingValueImputer, NotFittedError


def test_fit_uses_training_statistics_for_numeric_and_text_columns():
    train = pd.DataFrame({"value": [1.0, 3.0, np.nan], "label": ["b", "a", None]})
    test = pd.DataFrame({"value": [np.nan, 100.0], "label": [None, "z"]})
    original = test.copy(deep=True)

    imputer = MissingValueImputer(strategy="median").fit(train)
    result = imputer.transform(test)

    assert imputer.statistics_ == {"value": 2.0, "label": "a"}
    assert result.to_dict("list") == {"value": [2.0, 100.0], "label": ["a", "z"]}
    pd.testing.assert_frame_equal(test, original)
    assert result is not test
    assert imputer.get_feature_names_out() == ["value", "label"]


@pytest.mark.parametrize(
    ("strategy", "expected"),
    [("mean", 3.0), ("median", 1.0), ("most_frequent", 1.0)],
)
def test_numeric_strategies(strategy, expected):
    train = pd.DataFrame({"value": [1.0, 1.0, 7.0, np.nan]})
    result = MissingValueImputer(strategy=strategy).fit_transform(train)
    assert result.loc[3, "value"] == expected
    assert pd.api.types.is_numeric_dtype(result["value"])


def test_mode_ties_choose_smallest_value_independent_of_row_order():
    train = pd.DataFrame({"value": ["z", "a", "z", "a", None]})
    for values in (train, train.iloc[::-1]):
        imputer = MissingValueImputer().fit(values)
        assert imputer.statistics_["value"] == "a"


@pytest.mark.parametrize("strategy", ["mean", "median", "most_frequent"])
def test_nullable_boolean_uses_most_frequent_for_every_strategy(strategy):
    train = pd.DataFrame(
        {"flag": pd.Series([True, True, False, pd.NA], dtype="boolean")}
    )
    imputer = MissingValueImputer(strategy=strategy).fit(train)
    result = imputer.transform(train)
    assert imputer.statistics_["flag"] == True  # noqa: E712
    assert result.loc[3, "flag"] == True  # noqa: E712
    assert result["flag"].dtype == "boolean"


def test_mode_ties_with_mixed_types_do_not_raise():
    train = pd.DataFrame({"value": pd.Series(["a", 1, None], dtype=object)})
    for values in (train, train.iloc[::-1]):
        imputer = MissingValueImputer().fit(values)
        assert imputer.statistics_["value"] == 1


def test_invalid_strategy_is_rejected_in_fit_not_init():
    imputer = MissingValueImputer(strategy="unknown")
    with pytest.raises(ValueError, match="unknown"):
        imputer.fit(pd.DataFrame({"value": [1.0]}))
    with pytest.raises(NotFittedError):
        imputer.transform(pd.DataFrame({"value": [np.nan]}))


def test_all_missing_column_names_itself_in_error():
    with pytest.raises(ValueError, match="empty_column"):
        MissingValueImputer().fit(pd.DataFrame({"empty_column": [None, None]}))


def test_nullable_integer_with_fractional_mean_stays_numeric():
    train = pd.DataFrame({"value": pd.Series([1, 2, pd.NA], dtype="Int64")})
    result = MissingValueImputer().fit_transform(train)
    assert result.loc[2, "value"] == 1.5
    assert pd.api.types.is_numeric_dtype(result["value"])


def test_sklearn_pipeline_cross_validation_and_grid_search():
    rng = np.random.default_rng(7)
    X = pd.DataFrame(rng.normal(size=(30, 2)), columns=["a", "b"])
    X.loc[::5, "a"] = np.nan
    y = np.tile([0, 1], 15)
    pipeline = make_pipeline(MissingValueImputer(), LogisticRegression())

    assert len(cross_val_score(pipeline, X, y, cv=3)) == 3
    search = GridSearchCV(
        pipeline,
        {"missingvalueimputer__strategy": ["mean", "median"]},
        cv=3,
        error_score="raise",
    ).fit(X, y)
    assert search.best_estimator_.named_steps["missingvalueimputer"].statistics_
    assert not hasattr(clone(MissingValueImputer().fit(X)), "statistics_")
