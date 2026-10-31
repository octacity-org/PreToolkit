"""Chain preprocessing blocks so that one ``fit`` / ``transform`` runs them all."""

from preprocessing_toolkit.base import BasePreprocessor, _check_dataframe


class Pipeline(BasePreprocessor):
    """Run blocks in order, feeding each block's output into the next.

    Parameters
    ----------
    steps : list
        Each item is a block or a ``(name, block)`` pair. An unnamed block
        is named after its lowercase class name; repeated class names get
        ``-1``, ``-2``, ... suffixes, as in scikit-learn's ``make_pipeline``.

    A step's name is its address: ``named_steps[name]`` returns the block and
    ``"<name>__<param>"`` reaches its parameters in ``get_params``,
    ``set_params`` and ``GridSearchCV``. Names must be unique, must not
    contain ``__`` and must not be ``"steps"``. Invalid steps raise
    ``ValueError`` wherever names are needed (``fit``, ``get_params``,
    ``set_params``, ``named_steps``), never in ``__init__``.

    ``fit`` fits the given block objects in place, as scikit-learn does, so
    reusing one block object in two pipelines makes them share its state.
    """

    def __init__(self, steps):
        self.steps = steps

    @property
    def named_steps(self):
        """``{name: block}`` in pipeline order."""
        return dict(_name_steps(self.steps))

    def fit(self, X, y=None):
        self.fit_transform(X, y)
        return self

    def fit_transform(self, X, y=None):
        # Same contract as BasePreprocessor.fit, but each step's
        # fit_transform output is reused instead of running the chain twice.
        # This also matters for blocks whose fit_transform differs from
        # fit().transform(), such as cross-fitted target encoders.
        self.__dict__.pop("feature_names_in_", None)
        X = _check_dataframe(X)
        Xt = X
        for _, block in _name_steps(self.steps):
            Xt = block.fit_transform(Xt, y)
        self.feature_names_in_ = list(X.columns)
        return Xt

    def _fit(self, X, y=None):
        # Unused: fit and fit_transform above replace BasePreprocessor.fit.
        raise NotImplementedError

    def _transform(self, X):
        for _, block in _name_steps(self.steps):
            X = block.transform(X)
        return X

    def get_feature_names_out(self, input_features=None):
        """Return the column names after the last step."""
        names = super().get_feature_names_out(input_features)
        for _, block in _name_steps(self.steps):
            names = block.get_feature_names_out(names)
        return names

    def get_params(self, deep=True):
        """Return ``steps`` and, if ``deep``, each step and its parameters.

        A step appears under its name and each of its parameters under
        ``"<name>__<param>"``.
        """
        params = {"steps": self.steps}
        if not deep:
            return params
        for name, block in _name_steps(self.steps):
            params[name] = block
            for key, value in block.get_params(deep=True).items():
                params[f"{name}__{key}"] = value
        return params

    def set_params(self, **params):
        """Set ``steps``, replace a step by name, or set ``<name>__<param>``.

        Everything is validated first: on an unknown step or parameter a
        ``ValueError`` is raised and nothing changes. A replaced step is
        stored as ``(name, block)`` so its name, and every address that uses
        it, stays the same. The list passed by the user is never modified.
        """
        steps, nested = self._plan_set_params(params)
        self.steps = steps
        blocks = dict(_name_steps(steps))
        for name, step_params in nested.items():
            blocks[name].set_params(**step_params)
        return self

    def _plan_set_params(self, params):
        """Validate ``set_params`` arguments without changing anything.

        Returns the new ``steps`` and ``{step name: {param: value}}``. Nested
        pipelines are validated recursively, so an error deep in the tree
        still leaves every step unchanged.
        """
        params = dict(params)
        steps = params.pop("steps", self.steps)
        named = _name_steps(steps)
        names = [name for name, _ in named]

        replacements = {key: v for key, v in params.items() if "__" not in key}
        unknown = [key for key in replacements if key not in names]
        if unknown:
            raise ValueError(f"Unknown steps {unknown}. Steps: {names}.")
        if replacements:
            steps = [
                (name, replacements[name]) if name in replacements else step
                for step, (name, _) in zip(steps, named, strict=True)
            ]
            named = _name_steps(steps)  # validates the new blocks

        blocks = dict(named)
        nested = {}
        for key, value in params.items():
            if "__" not in key:
                continue
            name, _, param = key.partition("__")
            if name not in blocks:
                raise ValueError(f"Unknown step {name!r} in {key!r}. Steps: {names}.")
            if param not in blocks[name].get_params(deep=True):
                raise ValueError(f"Step {name!r} has no parameter {param!r}.")
            nested.setdefault(name, {})[param] = value

        for name, step_params in nested.items():
            if isinstance(blocks[name], Pipeline):
                blocks[name]._plan_set_params(step_params)
        return steps, nested


def _name_steps(steps):
    """Return ``[(name, block)]`` for ``steps``, raising ``ValueError`` if invalid."""
    if not isinstance(steps, list):
        raise ValueError(
            f"steps must be a list of blocks or (name, block) pairs, "
            f"got {type(steps).__name__}."
        )
    if not steps:
        raise ValueError("A Pipeline needs at least one step.")

    given = []
    for i, step in enumerate(steps):
        if isinstance(step, tuple) and len(step) == 2 and isinstance(step[0], str):
            name, block = step
        else:
            name, block = None, step
        if not isinstance(block, BasePreprocessor):
            raise ValueError(
                f"Step {i} must be a BasePreprocessor or a (name, block) pair, "
                f"got {block!r}."
            )
        given.append((name, block))

    auto = [type(block).__name__.lower() for name, block in given if name is None]
    seen = {}
    named = []
    for name, block in given:
        if name is None:
            name = type(block).__name__.lower()
            if auto.count(name) > 1:
                seen[name] = seen.get(name, 0) + 1
                name = f"{name}-{seen[name]}"
        named.append((name, block))

    names = [name for name, _ in named]
    repeated = sorted({name for name in names if names.count(name) > 1})
    if repeated:
        raise ValueError(
            f"Step names must be unique; repeated: {repeated}. "
            "Name the steps explicitly with (name, block) pairs."
        )
    for name in names:
        if "__" in name:
            raise ValueError(f"Step name {name!r} must not contain '__'.")
        if name == "steps":
            raise ValueError("'steps' is reserved and cannot be a step name.")
    return named
