"""Shared contract for all preprocessing blocks."""

import inspect
from abc import ABC, abstractmethod

import pandas as pd


class NotFittedError(ValueError, AttributeError):
    """Raised when a block is used before it has been fitted."""


class BasePreprocessor(ABC):
    """Abstract base class for preprocessing blocks (template method pattern).

    The public ``fit``, ``transform`` and ``fit_transform`` methods run the
    shared checks; subclasses implement only ``_fit`` and ``_transform``.

    Rules for subclasses:

    * ``__init__`` stores each argument unchanged in an attribute of the same
      name and does nothing else: no validation, no computation.
      ``get_params`` and ``sklearn.base.clone`` rely on this.
    * Everything learned in ``_fit`` is stored in attributes ending in ``_``
      (for example ``mean_`` or ``categories_``).
    """

    def fit(self, X, y=None):
        # A failed (re)fit must leave the block unfitted, not half-updated.
        self.__dict__.pop("feature_names_in_", None)
        X = _check_dataframe(X)
        self._fit(X, y)
        self.feature_names_in_ = list(X.columns)
        return self

    def transform(self, X):
        self._check_is_fitted()
        X = _check_dataframe(X)
        self._check_columns(X)
        return self._transform(X[self.feature_names_in_])

    def fit_transform(self, X, y=None):
        return self.fit(X, y).transform(X)

    def get_feature_names_out(self, input_features=None):
        """Return the output column names.

        By default a block keeps its input columns. Blocks that create or
        rename columns (for example the encoder) override this method.
        """
        self._check_is_fitted()
        if (
            input_features is not None
            and list(input_features) != self.feature_names_in_
        ):
            raise ValueError(
                f"input_features {list(input_features)} do not match the columns "
                f"seen in fit {self.feature_names_in_}."
            )
        return list(self.feature_names_in_)

    def get_params(self, deep=True):
        """Return the ``__init__`` arguments as ``{name: value}``.

        ``deep`` is accepted for scikit-learn compatibility. Blocks have no
        nested parameters, so the result is always flat; ``Pipeline``
        overrides this to expose its steps' parameters.
        """
        return {name: getattr(self, name) for name in self._get_param_names()}

    def set_params(self, **params):
        valid = self._get_param_names()
        unknown = [name for name in params if name not in valid]
        if unknown:
            raise ValueError(
                f"Invalid parameters {unknown} for {type(self).__name__}. "
                f"Valid parameters: {valid}."
            )
        for name, value in params.items():
            setattr(self, name, value)
        return self

    def __sklearn_is_fitted__(self):
        return hasattr(self, "feature_names_in_")

    def __sklearn_tags__(self):
        """Describe the block to scikit-learn >= 1.6 as an unsupervised transformer.

        Only scikit-learn calls this method, so importing it here keeps
        scikit-learn out of the runtime dependencies. Older versions never
        call it.

        This mirrors the layout of scikit-learn's ``Tags`` dataclasses: if a
        scikit-learn release changes that structure, this method must be
        updated.
        """
        from sklearn.utils import InputTags, Tags, TargetTags, TransformerTags

        return Tags(
            estimator_type=None,
            target_tags=TargetTags(required=False),
            transformer_tags=TransformerTags(),
            input_tags=InputTags(),
        )

    @abstractmethod
    def _fit(self, X, y=None):
        """Learn from ``X`` and store the results in attributes ending in ``_``."""

    @abstractmethod
    def _transform(self, X):
        """Return a new, transformed DataFrame."""

    @classmethod
    def _get_param_names(cls):
        if cls.__init__ is object.__init__:
            return []
        params = list(inspect.signature(cls.__init__).parameters.values())[1:]
        for param in params:
            if param.kind in (param.VAR_POSITIONAL, param.VAR_KEYWORD):
                raise TypeError(
                    f"{cls.__name__}.__init__ must not use *args or **kwargs; "
                    "list every parameter explicitly."
                )
        return [param.name for param in params]

    def _check_is_fitted(self):
        if not self.__sklearn_is_fitted__():
            raise NotFittedError(
                f"{type(self).__name__} is not fitted yet; call fit() first."
            )

    def _check_columns(self, X):
        fitted = set(self.feature_names_in_)
        missing = [col for col in self.feature_names_in_ if col not in X.columns]
        unexpected = [col for col in X.columns if col not in fitted]
        if missing or unexpected:
            raise ValueError(
                "Columns differ from those seen in fit "
                f"(missing: {missing}, unexpected: {unexpected})."
            )


def _check_dataframe(X):
    if not isinstance(X, pd.DataFrame):
        raise TypeError(f"Input must be a pandas DataFrame, got {type(X).__name__}.")
    duplicated = X.columns[X.columns.duplicated()].unique().tolist()
    if duplicated:
        raise ValueError(f"Input has duplicate column names: {duplicated}.")
    return X
