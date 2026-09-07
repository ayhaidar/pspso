# Getting Started

PSPSO includes the dashboard, CLI and Python API in one package. Python 3.10
through 3.12 is supported; using the installed dashboard does not require Node.js.

## Install from PyPI

Choose uv or pip. The uv tool installation is recommended for the dashboard and
CLI because it keeps PSPSO in an isolated environment:

=== "uv"

    ```bash
    uv tool install pspso
    ```

=== "pip"

    ```bash
    python -m pip install --upgrade pspso
    ```

[uv](https://docs.astral.sh/uv/guides/tools/) installs PSPSO
from PyPI and exposes its `pspso` and `pspso-dashboard` commands. To import the
Python API from an existing uv project, run `uv add pspso` in that project and
use `uv run` for its commands.
If needed, follow the [uv installation guide](https://docs.astral.sh/uv/getting-started/installation/) first.

The package includes the dashboard and CLI together with their Python
dependencies. You can check your installed version with:

=== "uv"

    ```bash
    uv tool list
    ```

=== "pip"

    ```bash
    python -m pip show pspso
    ```

Optional engines can be selected during installation. Choose one requirement
with the extras you need:

=== "uv"

    ```bash
    uv tool install "pspso[xgboost]"
    uv tool install "pspso[xgboost,lightgbm,torch]"
    ```

=== "pip"

    ```bash
    python -m pip install --upgrade "pspso[xgboost]"
    python -m pip install --upgrade "pspso[lightgbm]"
    python -m pip install --upgrade "pspso[torch]"
    ```

Inside an existing uv project, use `uv add "pspso[xgboost]"` or combine the
extras in one requirement.

The standard scikit-learn models are included in the base installation.

## Open the dashboard

After installing version 1.0, keep this command running in a terminal:

```bash
pspso-dashboard
```

Open [the dashboard](http://127.0.0.1:8000). The Overview page explains the
workflow; choose **Set up an experiment** to start. Closing the server terminal
stops the dashboard service.

If port 8000 is already in use, select another port:

```bash
pspso-dashboard --port 8002
```

Then open `http://127.0.0.1:8002`. CLI submissions to that service use
`pspso --api-url http://127.0.0.1:8002 ...`.

## First experiment

1. In **Data setup**, choose the **Breast cancer** example and inspect the data.
2. Use **Binary classification**, **ROC AUC**, and the default five-fold evaluation.
3. Review the feature preparation settings, then validate and continue.
4. In **Model & parameters**, choose **SVM** and continue.
5. In **Search engine**, choose **Random** with **6** trials, validate, and start.
6. Watch **Live experiments**, then open **Results** when the run completes.

The [dashboard guide](dashboard/workflow.md) explains each stage and setting.

## Use the CLI

In a second terminal, inspect the available commands and shared run history:

```bash
pspso --help
pspso run --help
pspso run list
```

These commands work directly after `uv tool install pspso` or a pip installation.
Inside a uv project that uses `uv add pspso`, prefix them with `uv run`.

The [CLI guide](guides/cli.md) includes a complete experiment you can copy into a
JSON file, validate and run. Dashboard and CLI commands should use the same
working directory or `PSPSO_HOME` to share a workspace.

For direct Python use, see the [notebook guide](guides/notebooks.md).
For development tools and MkDocs, see [source setup](development/setup.md).
