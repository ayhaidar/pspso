# Scenarios

Scenario files live in `examples/scenarios/`. They are versioned `ExperimentSpec`
JSON payloads accepted by the CLI and dashboard API.

Run one from the command line:

```bash
uv run pspso run start examples/scenarios/breast_cancer_svm_random.json --wait
```

Or reproduce it in the dashboard by selecting the same dataset, task, estimator,
strategy, fixed params, and tunable params.

## Included Scenarios

| File | Dataset | Task | Estimator | Strategy |
| --- | --- | --- | --- | --- |
| `breast_cancer_svm_random.json` | Breast cancer | Binary classification | SVM | Random |
| `breast_cancer_random_forest_grid.json` | Breast cancer | Binary classification | RandomForest | Grid |
| `diabetes_svm_regression_random.json` | Diabetes | Regression | SVM | Random |
| `diabetes_random_forest_pso.json` | Diabetes | Regression | RandomForest | PSO |
| `csv_binary_classification_template.json` | CSV | Binary classification | SVM | Random |
| `csv_regression_template.json` | CSV | Regression | RandomForest | Random |
| `wine_random_forest_multiclass.json` | Wine | Multiclass classification | RandomForest | Random |

## Dashboard Reproduction

For a checked-in scenario:

1. Open the JSON file.
2. Match the dataset fields in the dashboard.
3. Select the same task, metric, estimator, and strategy.
4. Use the same fixed params and tunable params.
5. Validate.
6. Start the run only after validation succeeds.

## Multiclass and PyTorch

The Wine scenario demonstrates macro F1 and a reproducible multiclass run:

```bash
uv run pspso run start examples/scenarios/wine_random_forest_multiclass.json --wait
```

The optional PyTorch tabular MLP emits epoch events for the live timeline:

```bash
uv sync --extra torch
```

The frontend does not execute arbitrary code. Select a registered model recipe
or use the documented Python extension contract for a custom estimator.
