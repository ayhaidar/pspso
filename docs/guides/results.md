# Results and Diagnostics

The Results page selects tools that are valid for the run's task and available
model outputs.

## Shared Tools

Metric summaries, convergence, dataset summaries, prediction tables, feature
importance and failure analysis are shared across tasks. History compares models
under an experiment filter; the live trial ledger retains candidate scores.

## Classification

Binary runs can show ROC and PR curves, AUC values, confusion matrix,
sensitivity, specificity, F1, accuracy, log loss, threshold diagnostics, and
probabilities in prediction downloads. Multiclass runs add per-class measures and
one-vs-rest ROC curves when probabilities are available.

## Regression

Regression runs show RMSE, MAE, R2, actual-versus-predicted values, residual
plots and individual prediction rows. Residuals are actual minus predicted.

Feature importance is identified as native tree importance, model coefficient,
or calculated permutation importance. Unsupported tools remain disabled with a
reason instead of rendering empty charts.

Results default to the untouched test split when available. CV summaries show
individual fold values, mean and standard deviation; they do not represent one
model's validation predictions. Select test for saved-model predictions. Holdout
validation predictions use the saved selection model from before final refit.
When cross-validation final refitting is disabled, PSPSO saves the winning fold
models as an ensemble and averages them for test predictions. Results also show
the selected trial beside the last completed candidate so a later, worse score is
not mistaken for the winner.

Export provides full prediction CSV, metrics JSON, specification, events,
manifest, model, environment, saved indices and worker log. CSV includes all rows
and raw features, with class probabilities where supported. Reading results
never retrains. Missing serialized models produce a visible error, and artifact
serialization warnings are shown above the report.
Older cross-validation runs that deliberately skipped refitting may predate saved
fold ensembles. Their Results page keeps metric tools available, disables model
tools, omits invalid model downloads, and can copy the saved setup into a new run.
