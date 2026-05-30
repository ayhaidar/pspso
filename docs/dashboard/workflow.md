# Dashboard Workflow

The dashboard is a guided builder for `RunRequest`, the backend request model
used to start optimization.

## Dataset

Choose one of:

- `breast_cancer`: built-in binary-classification dataset.
- `diabetes`: built-in regression dataset.
- CSV text: pasted tabular data with a target column.

The dashboard sends dataset information in:

```json
{
  "dataset": {
    "source": "example",
    "name": "breast_cancer",
    "target_column": "target"
  }
}
```

## Task And Metric

Tasks control valid metrics and estimators.

| Task | Valid metrics | Typical estimators |
| --- | --- | --- |
| Regression | RMSE | SVM, RandomForest, MLP, XGBoost, LightGBM |
| Binary classification | Accuracy, ROC AUC | SVM, RandomForest, LogisticRegression, MLP, XGBoost, LightGBM |

The frontend must hide invalid choices. For example, ROC AUC is invalid for
regression.

## Estimator

Estimator metadata comes from `GET /api/estimators`. The frontend should use
that response as its source of truth for:

- supported tasks;
- optional dependency status;
- default fixed parameters;
- default tunable search space;
- human-readable help text.

When the estimator changes, the dashboard must replace stale tunable params.
This prevents failures such as sending SVM's `kernel` parameter to
RandomForest.

## Fixed Params vs Tunable Params

Fixed params are applied to every trial. Tunable params are optimized by the
selected search strategy.

Example fixed params:

```json
{
  "random_state": 42,
  "n_jobs": -1
}
```

Example tunable params:

```json
{
  "n_estimators": { "type": "int", "low": 10, "high": 100 },
  "max_depth": { "type": "int", "low": 2, "high": 12 }
}
```

## Strategy

| Strategy | Meaning |
| --- | --- |
| Random | Samples a fixed number of parameter sets. Good default for quick runs. |
| Grid | Evaluates every combination. Safe for tiny spaces only. |
| PSO | Uses particle swarm optimization across the encoded search space. |

## Runtime

Runtime fields limit work and control feedback:

- `max_trials`: maximum number of trials for random/PSO.
- `particles`: PSO swarm size.
- `iterations`: PSO iterations.
- `verbose`: backend estimator verbosity where supported.

Always validate before starting a run.
