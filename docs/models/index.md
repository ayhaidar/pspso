# Model Registry and Capabilities

The dashboard and CLI expose only registered recipes with validated task and
parameter metadata.

| ID | Family | Tasks | Optional dependency |
| --- | --- | --- | --- |
| `linear_regression` | Linear baseline | Regression | None |
| `logistic_regression` | Linear baseline | Classification | None |
| `elastic_net` | Regularized linear | Regression | None |
| `random_forest` | Ensemble | All | None |
| `extra_trees` | Ensemble | All | None |
| `hist_gradient_boosting` | Boosting | All | None |
| `svm` | Kernel | All | None |
| `sklearn_mlp` | Neural network | All | None |
| `xgboost` | Boosting | All | `pspso[xgboost]` |
| `lightgbm` | Boosting | All | `pspso[lightgbm]` |
| `pytorch_mlp` | Neural network | All | `pspso[torch]` |

Use `list_estimators(task=...)` to discover canonical IDs and
`get_estimator_info(name)` to inspect capabilities, dependency status, fixed
defaults, typed search spaces, and accepted parameters.

Probability-dependent metrics are allowed only for recipes that provide scores
or probabilities. Native feature importance is used for tree models,
coefficients for linear models, and permutation importance as an explicit
fallback.
