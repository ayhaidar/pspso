# Local Experiments

`pspso` runs experiments locally. FastAPI remains the backend: the dashboard
sends requests to it, FastAPI validates and records them, and a managed child
Python process performs training. The browser never receives a shell or direct
machine access.

## Workspace

The default workspace is `.pspso/v1/` in the directory where the backend or CLI is
started:

| Location | Contents |
| --- | --- |
| `tracking.sqlite3` | Experiments, run specifications, attempts, lifecycle events, and summaries. |
| `datasets/` | Saved CSV copies named by content fingerprint. |
| `artifacts/runs/<run-id>/` | Submitted spec, result, environment manifest, and serializable model/preprocessor files when available. |

Do not place confidential datasets in a shared project workspace unless every
user of that workspace is permitted to access them.

## GUI Flow

1. Save or inspect a dataset in **Data Setup**.
2. Choose a model in **Model & Parameters**, then configure and validate **Search Engine**.
3. FastAPI records `validation_passed` and `run_queued` before it starts a
   local worker process.
4. The worker writes preparation, trial, epoch, warning, failure, and terminal
   events into SQLite. SSE replays those persisted events to the browser.
5. Review run artifacts and prediction samples after completion.

Cancellation stops the managed worker and leaves the event trail intact.
Retry creates a new run from the original specification; it does not claim to
resume an incomplete PSO population.

Search Engine can select multiple compatible models for a tournament through
`POST /api/v1/experiments/{experiment_id}/tournament`. It accepts two or more
validated run specs, freezes them under the same experiment, and sends them
through the same local queue for fair comparison.

## CLI Flow

```bash
uv run pspso dataset import data/customer_churn.csv --name customer-churn
uv run pspso dataset list
uv run pspso experiment create churn-baselines --tag local
uv run pspso run validate examples/scenarios/breast_cancer_svm_random.json
uv run pspso run start examples/scenarios/breast_cancer_svm_random.json --wait
uv run pspso run list
uv run pspso run show <run-id>
uv run pspso run cancel <run-id>
uv run pspso run retry <run-id>
```

Use `pspso run start <spec> --wait` when a command should wait for completion.
The local service must be running for start, cancel and retry. Use global
`--api-url` or `PSPSO_API_URL` to select it. Repository list/show commands do not
claim ownership. `--standalone --wait` provides supervised foreground execution
without an HTTP service. Queued attempts survive restart; active attempts
interrupted by shutdown remain available for explicit retry.

## Training Semantics

The dashboard defaults to five-fold CV. It reserves a test partition first, fits
preprocessing within each fold, selects by mean fold score, and refits the winner
on all development rows before final test evaluation. Exact row/fold indices and
the frozen dataset are saved with the specification and models.
When final refitting is disabled, the fitted fold models from the winning
candidate are retained as an averaged ensemble instead.

PSO is the default search engine. It supports true global-best and ring-local
topologies, a maximum trial count, wall-clock timeout, and no-improvement early
stopping. Random and grid searches are retained as reproducible comparisons.

PyTorch tabular MLPs are optional:

```bash
uv sync --extra torch
```

They emit `training_epoch` events with loss and device details. XGBoost and
LightGBM remain optional model recipes with their own extras.
