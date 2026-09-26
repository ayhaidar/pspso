# Releasing to PyPI

PSPSO is distributed through the existing [`pspso` project on
PyPI](https://pypi.org/project/pspso/). The one package contains the Python API,
CLI, FastAPI service and compiled dashboard.

Normal pushes and pull requests only verify the repository. Publishing is
triggered by a non-prerelease GitHub Release and uses PyPI Trusted Publishing,
so no long-lived PyPI token is stored in GitHub.

## One-time repository setup

1. Create or enable the `main` branch and make it the default release branch.
2. In **Settings → Pages**, choose **GitHub Actions** as the Pages source. The
   documentation workflow deploys <https://ayhaidar.github.io/pspso/> after
   documentation-related changes reach `main`.
3. Create a GitHub environment named `pypi`. Add required reviewers if release
   approval should be enforced in GitHub.
4. On PyPI, add a Trusted Publisher to the existing `pspso` project with:

   | Field | Value |
   | --- | --- |
   | Owner | `ayhaidar` |
   | Repository | `pspso` |
   | Workflow | `publish.yml` |
   | Environment | `pypi` |

See the [PyPI publisher setup
guide](https://docs.pypi.org/trusted-publishers/adding-a-publisher/) and [GitHub
Pages workflow guide](https://docs.github.com/en/pages/getting-started-with-github-pages/using-custom-workflows-with-github-pages).

## Prepare a release

1. Update the version in `pyproject.toml` and `frontend/package.json`, then
   refresh `frontend/package-lock.json` and `uv.lock`.
2. Update the release notes, documentation and `RELEASE_CHECKLIST.md`.
3. Build the frontend because its production assets are included in the wheel:

   ```bash
   cd frontend
   npm ci
   npm run build
   cd ..
   ```

4. Run the release checks:

   ```bash
   uv sync --locked --all-extras
   uv run --no-sync ruff check .
   uv run --no-sync ruff format --check .
   uv run --no-sync mypy pspso
   uv run --no-sync pytest
   uv run --no-sync mkdocs build --strict
   uv run --no-sync python scripts/build_release.py
   uv run --no-sync twine check --strict dist/*.whl dist/*.tar.gz
   ```

The build script reads the version from `pyproject.toml`, creates a clean wheel
and source archive, and checks the bundled dashboard, commands, typing marker
and required source files. `twine check --strict` validates package metadata and
the README rendering used by PyPI.

## Publish

1. Merge the verified revision into `main` and wait for the verification and
   documentation workflows to pass.
2. Create a GitHub Release from the exact tag `v<version>`, such as `v1.0.0`.
   Marking it as a prerelease prevents publication.
3. Publish the GitHub Release. The publishing workflow repeats the release
   gates, checks that the tag matches the package version, rebuilds the
   distributions from that tag and uploads them through Trusted Publishing.
4. Install the released version in a fresh environment:

   ```bash
   uv tool install "pspso==1.0.0"
   pspso --version
   pspso-dashboard --version
   ```

5. Check the project description and files on PyPI, open the public
   documentation, start the dashboard, and complete the example in the
   [CLI workflow](../guides/cli.md).

PyPI versions cannot be replaced. If publication fails after accepting one
distribution file, diagnose the workflow before creating a new version.
