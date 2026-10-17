"""Fill missing values using statistics learned during fitting."""

from pandas.api.types import is_numeric_dtype

from preprocessing_toolkit.base import BasePreprocessor


class MissingValueImputer(BasePreprocessor):
    """Impute numeric columns with ``strategy`` and other columns with their mode.

    Statistics are learned only from ``fit`` data and reused by ``transform``.
    """

    def __init__(self, strategy="mean"):
        self.strategy = strategy

    def _fit(self, X, y=None):
        if self.strategy not in ("mean", "median", "most_frequent"):
            raise ValueError(f"Unknown imputation strategy: {self.strategy!r}.")

        statistics = {}
        for column in X.columns:
            values = X[column].dropna()
            if values.empty:
                raise ValueError(f"Column {column!r} contains only missing values.")

            if is_numeric_dtype(X[column]) and self.strategy == "mean":
                statistic = values.mean()
            elif is_numeric_dtype(X[column]) and self.strategy == "median":
                statistic = values.median()
            else:
                counts = values.value_counts()
                candidates = counts[counts == counts.max()].index
                statistic = min(candidates)
            statistics[column] = statistic

        self.statistics_ = statistics

    def _transform(self, X):
        result = X.copy()
        for column, statistic in self.statistics_.items():
            values = result[column]
            if is_numeric_dtype(values):
                # Nullable integer/bool dtypes cannot hold fractional means.
                try:
                    result[column] = values.fillna(statistic)
                except TypeError:
                    result[column] = values.astype("Float64").fillna(statistic)
            else:
                result[column] = values.fillna(statistic)
        return result
