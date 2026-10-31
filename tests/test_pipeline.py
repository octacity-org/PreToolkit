import numpy as np
import pandas as pd
import pytest
from sklearn.base import clone
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from preprocessing_toolkit import BasePreprocessor, NotFittedError, Pipeline


class Shift(BasePreprocessor):
    """Subtracts the column means learned in fit, then adds ``offset``."""

    def __init__(self, offset=0.0):
        self.offset = offset

    def _fit(self, X, y=None):
        self.means_ = X.mean()

    def _transform(self, X):
        self.transform_calls = getattr(self, "transform_calls", 0) + 1
        return X - self.means_ + self.offset


class Rename(BasePreprocessor):
    """Adds ``prefix`` to every column name, like an encoder creating columns."""

    def __init__(self, prefix="r_"):
        self.prefix = prefix

    def _fit(self, X, y=None):
        pass

    def _transform(self, X):
        return X.add_prefix(self.prefix)

    def get_feature_names_out(self, input_features=None):
        names = super().get_feature_names_out(input_features)
        return [self.prefix + name for name in names]


class Explode(BasePreprocessor):
    def _fit(self, X, y=None):
        raise RuntimeError("boom")

    def _transform(self, X):
        return X


class RememberY(BasePreprocessor):
    def _fit(self, X, y=None):
        self.y_ = y

    def _transform(self, X):
        return X


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


# --- step names -------------------------------------------------------------


def test_named_and_unnamed_steps():
    first, second = Shift(), Rename()
    pipe = Pipeline([("center", first), second])
    assert list(pipe.named_steps) == ["center", "rename"]
    assert pipe.named_steps["center"] is first
    assert pipe.named_steps["rename"] is second


def test_repeated_unnamed_classes_get_numbered():
    pipe = Pipeline([Shift(), Rename(), Shift()])
    assert list(pipe.named_steps) == ["shift-1", "rename", "shift-2"]


@pytest.mark.parametrize(
    "steps, match",
    [
        ([("shift", Shift()), Shift()], "unique.*'shift'"),
        ([("a", Shift()), ("a", Rename())], "unique.*'a'"),
        ([("my__shift", Shift())], "__"),
        ([("steps", Shift())], "steps"),
        ([Shift(), StandardScaler()], "BasePreprocessor"),
        ([Shift(), "passthrough"], "BasePreprocessor"),
        ([], "at least one step"),
    ],
)
def test_invalid_steps_raise_in_fit_and_get_params(train, steps, match):
    pipe = Pipeline(steps)  # __init__ never validates
    with pytest.raises(ValueError, match=match):
        pipe.fit(train)
    with pytest.raises(ValueError, match=match):
        pipe.get_params()


def test_steps_must_be_a_list(train):
    with pytest.raises(ValueError, match="list"):
        Pipeline(Shift()).fit(train)


def test_init_stores_steps_unchanged():
    steps = [Shift(), ("r", Rename())]
    assert Pipeline(steps).steps is steps


# --- fit / transform --------------------------------------------------------


def test_fit_transform_matches_running_steps_by_hand(train):
    by_hand = Rename().fit_transform(Shift(offset=1.0).fit_transform(train))
    out = Pipeline([Shift(offset=1.0), Rename()]).fit_transform(train)
    pd.testing.assert_frame_equal(out, by_hand)


def test_transform_uses_what_each_step_learned_from_train(train):
    test = pd.DataFrame({"a": [100.0], "b": [100.0]})
    pipe = Pipeline([Shift(), Rename()]).fit(train)
    out = pipe.transform(test)
    # train means are 2 and 20; the test set's own mean must not be used
    assert out.loc[0, "r_a"] == 98.0
    assert out.loc[0, "r_b"] == 80.0


def test_fit_returns_self_and_fits_steps_in_place(train):
    shift = Shift()
    pipe = Pipeline([("shift", shift)])
    assert pipe.fit(train) is pipe
    assert shift.means_.tolist() == [2.0, 20.0]


def test_fit_transform_runs_the_chain_once(train):
    shift = Shift()
    Pipeline([shift, Rename()]).fit_transform(train)
    assert shift.transform_calls == 1


def test_y_is_passed_to_every_step(train):
    remember = RememberY()
    Pipeline([Shift(), remember]).fit(train, y=[0, 1, 0])
    assert remember.y_ == [0, 1, 0]


def test_transform_before_fit_raises(train):
    with pytest.raises(NotFittedError):
        Pipeline([Shift()]).transform(train)


def test_failed_refit_leaves_pipeline_unfitted(train):
    pipe = Pipeline([("shift", Shift())]).fit(train)
    pipe.set_params(shift=Explode())
    with pytest.raises(RuntimeError, match="boom"):
        pipe.fit(train)
    assert not pipe.__sklearn_is_fitted__()
    with pytest.raises(NotFittedError):
        pipe.transform(train)


def test_input_columns_are_checked(train):
    pipe = Pipeline([Shift()]).fit(train)
    with pytest.raises(ValueError, match=r"missing: \['b'\]"):
        pipe.transform(train[["a"]])


# --- get_feature_names_out --------------------------------------------------


def test_feature_names_pass_through_every_step(train):
    pipe = Pipeline([Shift(), Rename(prefix="x_"), Rename(prefix="y_")]).fit(train)
    assert pipe.get_feature_names_out() == ["y_x_a", "y_x_b"]
    assert pipe.get_feature_names_out() == list(pipe.transform(train).columns)


def test_get_feature_names_out_before_fit_raises():
    with pytest.raises(NotFittedError):
        Pipeline([Shift()]).get_feature_names_out()


# --- get_params / set_params ------------------------------------------------


def test_get_params_deep_lists_steps_and_their_parameters():
    shift, rename = Shift(offset=2.0), Rename()
    steps = [("center", shift), rename]
    params = Pipeline(steps).get_params()
    assert params == {
        "steps": steps,
        "center": shift,
        "center__offset": 2.0,
        "rename": rename,
        "rename__prefix": "r_",
    }


def test_get_params_shallow_returns_only_steps():
    steps = [Shift()]
    assert Pipeline(steps).get_params(deep=False) == {"steps": steps}


def test_set_params_reaches_into_a_step():
    pipe = Pipeline([("center", Shift())])
    assert pipe.set_params(center__offset=5.0) is pipe
    assert pipe.named_steps["center"].offset == 5.0


def test_replacing_an_unnamed_step_keeps_its_name():
    pipe = Pipeline([Shift(), Rename()])
    new = Rename(prefix="z_")
    pipe.set_params(shift=new)
    assert list(pipe.named_steps) == ["shift", "rename"]
    assert pipe.named_steps["shift"] is new
    pipe.set_params(shift__prefix="q_")  # the address still works
    assert new.prefix == "q_"


def test_replacing_a_step_does_not_modify_the_users_list():
    steps = [Shift()]
    pipe = Pipeline(steps)
    pipe.set_params(shift=Rename())
    assert isinstance(steps[0], Shift)
    assert pipe.steps is not steps


def test_set_params_can_replace_all_steps():
    pipe = Pipeline([Shift()])
    pipe.set_params(steps=[("r", Rename())])
    assert list(pipe.named_steps) == ["r"]


@pytest.mark.parametrize(
    "params",
    [
        {"center__offset": 9.0, "nope": Shift()},
        {"center__offset": 9.0, "nope__offset": 1.0},
        {"center__offset": 9.0, "center__nope": 1.0},
        {"center__offset": 9.0, "center": "passthrough"},
    ],
)
def test_invalid_set_params_raises_and_changes_nothing(params):
    shift = Shift()
    steps = [("center", shift)]
    pipe = Pipeline(steps)
    with pytest.raises(ValueError):
        pipe.set_params(**params)
    assert pipe.steps is steps
    assert shift.offset == 0.0


def test_invalid_nested_set_params_changes_nothing():
    first = Shift()
    outer = Pipeline([("a", first), ("inner", Pipeline([("c", Shift())]))])
    with pytest.raises(ValueError, match="passthrough"):
        outer.set_params(a__offset=1.0, inner__c="passthrough")
    assert first.offset == 0.0


def test_nested_pipelines_expose_and_set_deep_parameters():
    inner = Pipeline([("center", Shift())])
    outer = Pipeline([("inner", inner)])
    assert outer.get_params()["inner__center__offset"] == 0.0
    outer.set_params(inner__center__offset=3.0)
    assert inner.named_steps["center"].offset == 3.0


# --- scikit-learn compatibility ---------------------------------------------


def test_clone_returns_unfitted_pipeline_with_cloned_steps(train):
    pipe = Pipeline([("center", Shift(offset=4.0)), Rename()]).fit(train)
    copy = clone(pipe)
    assert not copy.__sklearn_is_fitted__()
    assert copy.get_params()["center__offset"] == 4.0
    assert copy.named_steps["center"] is not pipe.named_steps["center"]
    assert not hasattr(copy.named_steps["center"], "means_")


def test_cross_val_score_runs_in_sklearn_pipeline(xy):
    X, y = xy
    model = make_pipeline(Pipeline([Shift(), Rename()]), LogisticRegression())
    scores = cross_val_score(model, X, y, cv=3, error_score="raise")
    assert len(scores) == 3
    assert not np.isnan(scores).any()


def test_grid_search_tunes_a_nested_parameter(xy):
    X, y = xy
    model = make_pipeline(Pipeline([("center", Shift())]), LogisticRegression())
    search = GridSearchCV(
        model, {"pipeline__center__offset": [0.0, 1.0]}, cv=3, error_score="raise"
    )
    search.fit(X, y)
    assert search.best_params_["pipeline__center__offset"] in (0.0, 1.0)


def test_grid_search_can_swap_a_step(xy):
    X, y = xy
    model = make_pipeline(Pipeline([Shift()]), LogisticRegression())
    candidates = [Shift(), Rename()]
    search = GridSearchCV(
        model, {"pipeline__shift": candidates}, cv=3, error_score="raise"
    )
    search.fit(X, y)
    best = search.best_estimator_[0]
    assert list(best.named_steps) == ["shift"]  # the address survives the swap
