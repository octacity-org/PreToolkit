"""Encode categorical columns as numbers using categories learned during fitting."""

import numpy as np
import pandas as pd
from pandas.api.types import is_object_dtype, is_string_dtype

from preprocessing_toolkit.base import BasePreprocessor


class CategoricalEncoder(BasePreprocessor):
    """Encode non-numeric columns with one-hot or ordinal encoding.

    Only object, string and category columns are encoded; other columns pass
    through unchanged. Categories are learned only from ``fit`` data, sorted,
    and stored in ``categories_`` as ``{column: [category, ...]}``.

    Parameters
    ----------
    method : {"onehot", "ordinal"}, default="onehot"
        ``"onehot"`` replaces each encoded column with one 0/1 column per
        category, named ``<column>_<category>``. ``"ordinal"`` replaces it
        with the position of the category in ``categories_[column]``.
    handle_unknown : {"error", "ignore"}, default="error"
        What ``transform`` does with a category not seen in ``fit``:
        ``"error"`` raises a ``ValueError``; ``"ignore"`` encodes it as all
        zeros (one-hot) or ``-1`` (ordinal).
    """

    def __init__(self, method="onehot", handle_unknown="error"):
        self.method = method
        self.handle_unknown = handle_unknown

    def _fit(self, X, y=None):
        if self.method not in ("onehot", "ordinal"):
            raise ValueError(
                f"Unknown method: {self.method!r}. Use 'onehot' or 'ordinal'."
            )
        if self.handle_unknown not in ("error", "ignore"):
            raise ValueError(
                f"Unknown handle_unknown: {self.handle_unknown!r}. "
                "Use 'error' or 'ignore'."
            )

        categories = {}
        for column in X.columns:
            if _is_categorical(X[column]):
                _check_no_missing(X[column])
                categories[column] = sorted(X[column].drop_duplicates())

        names = _output_columns(X.columns, categories, self.method)
        duplicated = sorted({name for name in names if names.count(name) > 1})
        if duplicated:
            raise ValueError(
                f"One-hot column names collide with other columns: {duplicated}."
            )

        self.categories_ = categories

    def _transform(self, X):
        parts = []
        for column in X.columns:
            if column not in self.categories_:
                parts.append(X[[column]])
                continue

            values = X[column]
            _check_no_missing(values)
            categories = self.categories_[column]
            # Codes are the position in ``categories``; unseen values get -1.
            codes = pd.Index(categories).get_indexer(values).astype(np.int64)
            if self.handle_unknown == "error" and (codes == -1).any():
                unknown = sorted(values[codes == -1].drop_duplicates())
                raise ValueError(
                    f"Column {column!r} contains categories not seen in fit: {unknown}."
                )

            if self.method == "onehot":
                onehot = (codes[:, None] == np.arange(len(categories))).astype(np.int64)
                names = [f"{column}_{category}" for category in categories]
                parts.append(pd.DataFrame(onehot, index=X.index, columns=names))
            else:
                parts.append(pd.DataFrame({column: codes}, index=X.index))

        if not parts:
            return X.copy()
        return pd.concat(parts, axis=1)

    def get_feature_names_out(self, input_features=None):
        """Return the output column names, in the order ``transform`` produces."""
        # The base class checks fitted state and ``input_features``.
        columns = super().get_feature_names_out(input_features)
        return _output_columns(columns, self.categories_, self.method)


def _is_categorical(values):
    return (
        is_object_dtype(values)
        or is_string_dtype(values)
        or isinstance(values.dtype, pd.CategoricalDtype)
    )


def _check_no_missing(values):
    if values.isna().any():
        raise ValueError(
            f"Column {values.name!r} contains missing values; impute them first "
            "(for example with MissingValueImputer)."
        )


def _output_columns(columns, categories, method):
    names = []
    for column in columns:
        if column in categories and method == "onehot":
            names.extend(f"{column}_{category}" for category in categories[column])
        else:
            names.append(column)
    return names
