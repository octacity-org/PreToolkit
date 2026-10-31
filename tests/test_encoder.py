import numpy as np
import pandas as pd
import pytest
from sklearn.base import clone
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.pipeline import make_pipeline

from preprocessing_toolkit import CategoricalEncoder, NotFittedError


@pytest.fixture
def train():
    return pd.DataFrame(
        {
            "age": [30, 25, 40],
            "city": ["Izmir", "Ankara", "Izmir"],
            "size": ["S", "L", "M"],
        }
    )


@pytest.fixture
def xy():
    """~30 rows with a numeric and a categorical feature and a binary target."""
    rng = np.random.default_rng(0)
    X = pd.DataFrame(
        {
            "x": rng.normal(size=30),
            "city": rng.choice(["Ankara", "Izmir", "Istanbul"], size=30),
        }
    )
    y = np.tile([0, 1], 15)
    return X, y


# --- one-hot ----------------------------------------------------------------


def test_onehot_replaces_column_in_place_with_sorted_categories(train):
    encoder = CategoricalEncoder().fit(train)
    out = encoder.transform(train)

    assert encoder.categories_ == {"city": ["Ankara", "Izmir"], "size": ["L", "M", "S"]}
    assert list(out.columns) == [
        "age",
        "city_Ankara",
        "city_Izmir",
        "size_L",
        "size_M",
        "size_S",
    ]
    assert out["city_Ankara"].tolist() == [0, 1, 0]
    assert out["city_Izmir"].tolist() == [1, 0, 1]
    assert out["size_M"].tolist() == [0, 0, 1]


def test_onehot_columns_are_integers(train):
    out = CategoricalEncoder().fit_transform(train)
    for column in out.columns:
        assert pd.api.types.is_integer_dtype(out[column])


# --- ordinal ----------------------------------------------------------------


def test_ordinal_uses_position_in_categories(train):
    out = CategoricalEncoder(method="ordinal").fit_transform(train)

    assert list(out.columns) == ["age", "city", "size"]
    assert out["city"].tolist() == [1, 0, 1]
    assert out["size"].tolist() == [2, 0, 1]
    assert pd.api.types.is_integer_dtype(out["city"])


# --- which columns are encoded ----------------------------------------------


def test_numeric_columns_pass_through_unchanged(train):
    out = CategoricalEncoder().fit_transform(train)
    pd.testing.assert_series_equal(out["age"], train["age"])


@pytest.mark.parametrize("dtype", [object, "string", "category"])
def test_object_string_and_category_columns_are_encoded(dtype):
    X = pd.DataFrame({"c": pd.Series(["b", "a", "b"], dtype=dtype)})
    encoder = CategoricalEncoder().fit(X)
    assert encoder.categories_ == {"c": ["a", "b"]}
    assert encoder.transform(X)["c_b"].tolist() == [1, 0, 1]


def test_ordered_category_keeps_declared_order_and_unused_levels():
    X = pd.DataFrame(
        {
            "level": pd.Categorical(
                ["low", "high", "low"],
                categories=["low", "medium", "high"],
                ordered=True,
            )
        }
    )
    encoder = CategoricalEncoder(method="ordinal").fit(X)
    assert encoder.categories_ == {"level": ["low", "medium", "high"]}
    assert encoder.transform(X)["level"].tolist() == [0, 2, 0]


def test_mixed_type_column_is_encoded_independent_of_row_order():
    X = pd.DataFrame({"c": ["b", 1, "a", "1"]})
    for values in (X, X.iloc[::-1]):
        encoder = CategoricalEncoder(method="ordinal").fit(values)
        assert encoder.categories_ == {"c": [1, "1", "a", "b"]}
    assert encoder.transform(X)["c"].tolist() == [3, 0, 2, 1]


def test_mixed_type_unknown_values_are_reported():
    encoder = CategoricalEncoder().fit(pd.DataFrame({"c": ["a", "b"]}))
    with pytest.raises(ValueError, match=r"\[2, 'z'\]"):
        encoder.transform(pd.DataFrame({"c": ["z", 2, "a"]}))


def test_categories_do_not_depend_on_row_order(train):
    reversed_train = train.iloc[::-1]
    assert (
        CategoricalEncoder().fit(train).categories_
        == CategoricalEncoder().fit(reversed_train).categories_
    )


# --- unknown categories -----------------------------------------------------


def test_unknown_category_raises_naming_column_and_values(train):
    encoder = CategoricalEncoder().fit(train)
    test = pd.DataFrame({"age": [1, 2], "city": ["Bursa", "Adana"], "size": ["S"] * 2})
    with pytest.raises(ValueError, match=r"'city'.*\['Adana', 'Bursa'\]"):
        encoder.transform(test)


def test_unknown_category_ignored_gives_zeros_for_onehot(train):
    encoder = CategoricalEncoder(handle_unknown="ignore").fit(train)
    test = pd.DataFrame({"age": [1], "city": ["Bursa"], "size": ["S"]})
    out = encoder.transform(test)
    assert out.loc[0, "city_Ankara"] == 0
    assert out.loc[0, "city_Izmir"] == 0
    assert out.loc[0, "size_S"] == 1


def test_unknown_category_ignored_gives_minus_one_for_ordinal(train):
    encoder = CategoricalEncoder(method="ordinal", handle_unknown="ignore")
    encoder.fit(train)
    test = pd.DataFrame({"age": [1], "city": ["Bursa"], "size": ["S"]})
    out = encoder.transform(test)
    assert out.loc[0, "city"] == -1
    assert out.loc[0, "size"] == 2


def test_categories_come_from_training_data_only(train):
    test = pd.DataFrame({"age": [1], "city": ["Bursa"], "size": ["S"]})
    encoder = CategoricalEncoder(handle_unknown="ignore").fit(train)
    encoder.transform(test)
    assert "Bursa" not in encoder.categories_["city"]
    assert "city_Bursa" not in encoder.get_feature_names_out()


# --- validation -------------------------------------------------------------


@pytest.mark.parametrize(
    "params",
    [{"method": "binary"}, {"handle_unknown": "skip"}],
)
def test_invalid_parameters_are_rejected_in_fit_not_init(train, params):
    encoder = CategoricalEncoder(**params)  # does not raise
    with pytest.raises(ValueError, match="binary|skip"):
        encoder.fit(train)
    with pytest.raises(NotFittedError):
        encoder.transform(train)


def test_missing_values_in_fit_raise(train):
    X = train.assign(city=["Izmir", None, "Ankara"])
    with pytest.raises(ValueError, match="'city'.*impute"):
        CategoricalEncoder().fit(X)


def test_missing_values_in_transform_raise(train):
    encoder = CategoricalEncoder(handle_unknown="ignore").fit(train)
    X = train.assign(city=["Izmir", None, "Ankara"])
    with pytest.raises(ValueError, match="'city'.*impute"):
        encoder.transform(X)


def test_onehot_name_colliding_with_existing_column_raises():
    X = pd.DataFrame({"color": ["red", "blue"], "color_red": [1, 0]})
    with pytest.raises(ValueError, match="color_red"):
        CategoricalEncoder().fit(X)


def test_colliding_names_are_fine_for_ordinal():
    X = pd.DataFrame({"color": ["red", "blue"], "color_red": [1, 0]})
    out = CategoricalEncoder(method="ordinal").fit_transform(X)
    assert list(out.columns) == ["color", "color_red"]


# --- get_feature_names_out --------------------------------------------------


@pytest.mark.parametrize("method", ["onehot", "ordinal"])
def test_get_feature_names_out_matches_transform(train, method):
    encoder = CategoricalEncoder(method=method).fit(train)
    assert encoder.get_feature_names_out() == list(encoder.transform(train).columns)


def test_get_feature_names_out_accepts_matching_input_features(train):
    encoder = CategoricalEncoder(method="ordinal").fit(train)
    assert encoder.get_feature_names_out(np.array(["age", "city", "size"])) == [
        "age",
        "city",
        "size",
    ]


def test_get_feature_names_out_rejects_mismatched_input_features(train):
    encoder = CategoricalEncoder().fit(train)
    with pytest.raises(ValueError, match="input_features"):
        encoder.get_feature_names_out(["city", "age", "size"])


def test_get_feature_names_out_before_fit_raises():
    with pytest.raises(NotFittedError):
        CategoricalEncoder().get_feature_names_out()


# --- transform contract -----------------------------------------------------


def test_transform_returns_new_frame_and_does_not_modify_input(train):
    original = train.copy()
    out = CategoricalEncoder().fit_transform(train)
    assert out is not train
    pd.testing.assert_frame_equal(train, original)


def test_transform_keeps_index(train):
    X = train.set_axis([10, 10, 20])
    out = CategoricalEncoder().fit_transform(X)
    assert out.index.tolist() == [10, 10, 20]
    assert out["city_Ankara"].tolist() == [0, 1, 0]


# --- scikit-learn compatibility ---------------------------------------------


def test_clone_of_fitted_encoder_is_unfitted_copy(train):
    encoder = CategoricalEncoder(method="ordinal").fit(train)
    copy = clone(encoder)
    assert copy.get_params() == {"method": "ordinal", "handle_unknown": "error"}
    assert not hasattr(copy, "categories_")


def test_cross_val_score_runs_in_sklearn_pipeline(xy):
    X, y = xy
    pipe = make_pipeline(
        CategoricalEncoder(handle_unknown="ignore"), LogisticRegression()
    )
    scores = cross_val_score(pipe, X, y, cv=3, error_score="raise")
    assert len(scores) == 3
    assert not np.isnan(scores).any()


def test_grid_search_tunes_method(xy):
    X, y = xy
    pipe = make_pipeline(
        CategoricalEncoder(handle_unknown="ignore"), LogisticRegression()
    )
    search = GridSearchCV(
        pipe,
        {"categoricalencoder__method": ["onehot", "ordinal"]},
        cv=3,
        error_score="raise",
    )
    search.fit(X, y)
    assert search.best_params_["categoricalencoder__method"] in ("onehot", "ordinal")
