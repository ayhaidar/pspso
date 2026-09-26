# Dashboard Workflow

The dashboard separates experiment work into six stages. A versioned draft is
saved in the browser while it is being prepared; submitted runs and results are
stored in the `.pspso` workspace.

The opening **Overview** page explains PSPSO, its supported tasks and the full
workflow. Use **Set up an experiment** to begin, **View history** to revisit saved
runs, or the PSPSO logo to return to the overview.

## 1. Data Setup

Data Setup contains three numbered sections:

1. **Dataset and target:** choose a built-in, saved or CSV dataset and inspect it.
2. **Data profile and evaluation:** review data quality and target statistics;
   select cross-validation (five folds by default) or train/validation/test.
   Choose random or chronological ordering and inspect the proposed partitions.
3. **Feature engineering:** include or exclude columns, handle missing values,
   optionally clip numeric outliers, standardize numeric features and configure
   nominal one-hot or explicitly ordered ordinal encoding.

Chronological evaluation keeps future rows out of training. A selected time
column sorts the data and is excluded from features; otherwise the existing row
order must run from oldest to newest. An optional row gap separates evaluation
windows. Chronological cross-validation uses expanding training windows.

Inspection also reports source rows, duplicate counts, descriptive numeric
statistics, class distribution for classification, and target statistics for
regression. The proposed split summary uses the selected seed and stratification
setting, so its row and class counts match the partitions used by the worker.

Missing numeric features can use median, mean, most-frequent, or constant-value
imputation. Missing categorical features can use most-frequent or constant-label
imputation, and optional missingness indicators can preserve the fact that a
value was absent. These steps are learned from training rows only. Missing target
values remain a blocking error because inventing labels would invalidate the
evaluation.

## 2. Model And Parameters

The model catalog is filtered by task. Availability and capabilities come from
`GET /api/v1/estimators`, including optional dependency status, probability
outputs, feature importance, scaling advice, and epoch monitoring.

Fixed parameters apply to every candidate. Tunable parameters belong to the
search space. The interface deliberately prevents one parameter from appearing
in both groups.

## 3. Search Engine

Choose random, grid, or PSO search. The page shows the planned number of model
fits and the real local-worker concurrency before launch. PSO exposes particles,
iterations, topology, cognitive/social coefficients, and inertia. The complete
specification is validated before it enters the queue.

For PSO, the planned fit count is `particles x iterations`. Selecting PSO in the
dashboard clears a trial limit left over from random or grid search, so a hidden
old limit cannot stop the swarm after its first particle.
The candidate count is multiplied by the fold count for cross-validation, and
one additional fit is included when final refit is enabled. Evaluation choices
are shown here and edited in Data Setup. The saved-model control chooses between
one refitted winner and the already fitted winning fold ensemble. Both choices
produce a model artifact; the ensemble adds no extra fit.

## 4. Live Experiments

The live page is built from persisted events, not simulated state. It shows the
current worker activity, trial ledger, best-cost curve, logs, and timeline.
The visual canvas changes with the strategy: particles and topology for PSO,
sample positions for random search, and parameter cells for grid search. If SSE
disconnects, polling reloads the durable timeline.

The optimizer evaluates up to the configured number of concurrent candidates.
Active model fits and fold progress are counted independently of queued
candidates. Every successful candidate contributes one convergence point;
failed candidates remain in the ledger. Fit totals include all candidate folds
and the final refit when enabled.
The `iteration_completed` event marks the barrier after all particles in that
iteration have been evaluated.

## 5. Results

The default toolbox depends on the task and can be customized per experiment:

- Shared context: dataset dimensions, missingness, duplicates, target summary,
  and recorded train/validation/test sizes.

- Binary classification: ROC AUC, PR AUC, confusion matrix, sensitivity,
  specificity, threshold exploration, feature importance, and predictions.
- Multiclass classification: confusion matrix, per-class sensitivity and
  specificity, one-vs-rest ROC curves when probabilities exist, and predictions.
- Regression: RMSE, MAE, R2, actual-versus-predicted, residuals, feature
  importance, and largest row-level errors in the prediction table.

Diagnostics default to the untouched test split when one exists. Native tree or
coefficient importance is used first; permutation importance can be calculated
from the saved model when native importance is unavailable.

## 6. History

History combines GUI and CLI runs from SQLite. Runs can be filtered, reopened,
reviewed, or cloned back into Search Engine. Reopening Live Experiments replays
its ordered event history, including failures and cancellation.

## Validation

`POST /api/v1/workflow/validate` accepts `data`, `model`, `search`, or `full`
scope. `POST /api/v1/runs/validate` validates a complete `ExperimentSpec` for
CLI and programmatic clients.
