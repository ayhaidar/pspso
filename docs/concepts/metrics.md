# Metrics and Optimization Cost

Users choose a human-facing metric. Internally, every strategy minimizes a cost:

| Metric | Task | Better | Internal cost |
| --- | --- | --- | --- |
| RMSE, MAE, log loss | Applicable task | Lower | Metric value |
| R2 | Regression | Higher | `-R2` |
| Accuracy, ROC AUC, PR AUC, macro F1 | Classification | Higher | `1 - metric` |

This explains why the best metric can rise while the optimization cost falls.
The dashboard emphasizes the selected metric and formats cost as an optimizer
diagnostic rather than a competing result.

Canonical metric IDs are `rmse`, `mae`, `r2`, `accuracy`, `roc_auc`, `pr_auc`,
`log_loss`, and `f1_macro`. Short aliases are not accepted in 1.0.

CV optimization uses the mean fold cost and preserves each fold's metric plus
the population standard deviation. `positive_label` explicitly chooses the
binary positive class; sensitivity, specificity and thresholded predictions use
that class. AUC and PR AUC use its scores. Residuals are actual minus predicted.

Objective validation checks task and target domains for every objective in a
search space. For example, Gamma targets must be positive, Poisson and Tweedie
targets must be nonnegative with a positive value, and squared-log targets must
exceed -1. The supported objective definitions follow the
[XGBoost parameters](https://xgboost.readthedocs.io/en/stable/parameter.html) and
[LightGBM parameters](https://lightgbm.readthedocs.io/en/stable/Parameters.html).
