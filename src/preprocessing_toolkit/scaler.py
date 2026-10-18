"""Scale numeric columns to a common range."""

import pandas as pd

from preprocessing_toolkit.base import BasePreprocessor

_METHODS = ("standard", "minmax")


class FeatureScaler(BasePreprocessor):
    """Scale numeric columns as ``(x - center_) / scale_``.

    Parameters
    ----------
    method : {"standard", "minmax"}, default "standard"
        ``"standard"`` subtracts the mean and divides by the population
        standard deviation (``ddof=0``). ``"minmax"`` subtracts the minimum
        and divides by the range, mapping the training data to ``[0, 1]``.

    Non-numeric (including boolean) columns pass through unchanged. Missing
    values are ignored when learning statistics and stay missing. A constant
    column gets a scale of 1, so it becomes 0 instead of NaN or infinity.

    Attributes
    ----------
    center_ : dict
        ``{column: value subtracted}`` for each numeric column.
    scale_ : dict
        ``{column: value divided by}`` for each numeric column.
    """

    def __init__(self, method="standard"):
        self.method = method

    def _fit(self, X, y=None):
        if self.method not in _METHODS:
            raise ValueError(
                f"Unknown method {self.method!r}; expected one of {list(_METHODS)}."
            )
        columns = [
            col
            for col in X.columns
            if pd.api.types.is_numeric_dtype(X[col])
            and not pd.api.types.is_bool_dtype(X[col])
        ]
        numeric = X[columns]
        if self.method == "standard":
            center = numeric.mean()
            scale = numeric.std(ddof=0)
        else:
            center = numeric.min()
            scale = numeric.max() - center
        self.center_ = {col: float(center[col]) for col in numeric.columns}
        self.scale_ = {
            col: 1.0 if scale[col] == 0 else float(scale[col])
            for col in numeric.columns
        }

    def _transform(self, X):
        out = X.copy()
        for col, center in self.center_.items():
            out[col] = (X[col] - center) / self.scale_[col]
        return out
