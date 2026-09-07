# CLI workflow

The CLI and dashboard use the same versioned specifications and durable workspace.
Follow [Getting started](../getting-started.md) for uv and pip installation.

After `uv tool install pspso`, run the commands exactly as shown. If PSPSO was
added to an existing project with `uv add pspso`, prefix each command with
`uv run`. Run both terminals from the same working directory.

## Create your first specification

Save the following as `experiment.json` in your working directory. It uses an
included dataset, so no source checkout or separate data download is needed:

```json
{
  "schema_version": 1,
  "dataset": {"source": "example", "name": "breast_cancer", "target_column": "target"},
  "task": "binary_classification",
  "metric": "roc_auc",
  "estimator": "random_forest",
  "fixed_params": {"n_estimators": 20},
  "search_space": {"max_depth": {"type": "int", "low": 2, "high": 5}},
  "strategy": "random",
  "evaluation": {"protocol": "cross_validation", "folds": 5},
  "runtime": {"max_trials": 3, "verbose": 0}
}
```

## Submit to the dashboard service

Start the local service in one terminal and leave it running:

```bash
pspso-dashboard --host 127.0.0.1 --port 8000
```

In a second terminal, using the same working directory:

```bash
pspso run validate experiment.json
pspso --api-url http://127.0.0.1:8000 run start experiment.json --wait
pspso run list
```

Use the `run_id` from the JSON output in these commands (replace `<run_id>`):

```bash
pspso run show <run_id>
pspso run cancel <run_id>
pspso run retry <run_id> --wait
```

`--api-url` is a global option. `PSPSO_API_URL` provides the default service URL;
otherwise it is `http://127.0.0.1:8000`. Start, retry and cancel always go through
the service. A stopped service produces an actionable error. Start returns after
queueing unless `--wait` is supplied. A waited failed, cancelled or interrupted
run returns a nonzero exit status.

## Datasets and experiments

With your own `data.csv`, these commands import a dataset and organize experiments.
Repository-only commands never start a manager or interrupt active jobs:

```bash
pspso dataset import data.csv --name "Customer churn v1"
pspso dataset list
pspso experiment create "Churn study" --tag baseline
pspso experiment list
pspso run validate experiment.json
pspso run list
pspso run show <run_id>
```

Use the same `PSPSO_HOME` as the dashboard when reading its workspace. The default workspace is `.pspso/v1`. JSON and YAML specifications are
accepted. Validation checks schema, parameters, optional dependencies, objective
compatibility, target labels and available evaluation rows before saving a run.

## Foreground execution

For a foreground run without an HTTP service:

```bash
pspso run start experiment.json --standalone --wait
```

Standalone mode requires `--wait`. The foreground CLI owns a supervised worker
until it exits, and Ctrl+C requests cancellation followed by bounded cleanup.
It cannot claim a workspace that already has a live service. For background
execution, submit to the long-lived dashboard service.

Queued attempts survive service restarts. A terminated active attempt remains
historical as interrupted; retry creates a new run linked to its parent. Log,
heartbeat, queue position and attempt details are available in `run show` and in
the dashboard's live view.

See the [CLI reference](../api/cli.md) for every command, option, environment
variable and exit status.
