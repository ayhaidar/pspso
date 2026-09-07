# Troubleshooting

## Training Fails After It Worked Once

The most common cause is stale dashboard state. For example, if you start with
SVM and then change the estimator to RandomForest without replacing the search
space, the backend receives SVM params such as `kernel`, `C`, and `gamma`.
RandomForest does not accept those params, so every trial fails.

Fix:

1. Change estimator.
2. Let the dashboard reload estimator defaults from `GET /api/v1/estimators`.
3. Validate before starting.
4. Start only if validation passes.

## Invalid Metric For Task

Regression supports `rmse`. Binary classification supports `accuracy` and
`roc_auc`.

If validation reports a metric error, select a metric that belongs to the chosen
task.

## Missing Optional Backend

XGBoost and LightGBM are optional. Install only what you need:

```bash
uv sync --extra xgboost
uv sync --extra lightgbm
```

The backend reports missing optional dependencies before a run starts.

## CSV Target Column Not Found

Check that `target_column` exactly matches a CSV header. Column names are
case-sensitive.

## All Trials Fail

Look at the trial table. Each failed trial includes the estimator error. Common
causes:

- unsupported parameter name;
- invalid parameter value;
- incompatible task/metric;
- target values unsuitable for binary classification;
- missing optional backend.

## SSE Disconnects

The frontend should fall back to polling `GET /api/v1/runs/{run_id}`. Polling
keeps the status, best params, and final result visible even if the event stream
disconnects.
