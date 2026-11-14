"""PreToolkit's release pipeline, compiled by OmniShip.

This file is the source of truth for ``omniship.yaml``, ``omniship.lock`` and
``.github/workflows/{check,build,ship}.yml``. After editing it, regenerate them
with::

    uvx --from omniship==0.6.0 omniship generate

Pushing a ``v*`` tag runs Check -> Build -> Ship. Pull requests are tested by
``ci.yml``; the Check stage here runs only as part of a release.
"""

from omniship import Pipeline
from omniship.plugins.github import (
    GitHubActions,
    GitHubRelease,
    GitHubRunner,
    GitHubWorkflow,
)
from omniship.plugins.python import PyPIPublish, Pytest, Ruff, Wheel

github = GitHubActions(
    default_runner=GitHubRunner.UBUNTU_24_04,
    # No pull request or push triggers: ci.yml already tests every change.
    check=GitHubWorkflow(file="check.yml", name="Check", triggers=[]),
)
pipeline = Pipeline(targets=[github])


@pipeline.check
def check(stage):
    stage.task(Ruff())
    stage.task(Pytest())


@pipeline.build
def build(stage):
    stage.task(Wheel())


@pipeline.ship
def ship(stage):
    # Trusted Publishing: PyPI trusts this workflow, so no token is stored.
    # The "release" environment requires a maintainer's approval.
    publish = stage.task(
        PyPIPublish(trusted_publishing=True),
        execution=github.job(environment="release"),
    )
    stage.task(
        GitHubRelease(repository="octacity-org/PreToolkit", notes="auto"),
        after=[publish],
    )
