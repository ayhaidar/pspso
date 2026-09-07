# Data Preparation

## Bundled examples

The dashboard includes six versioned examples. Selecting one applies its
standard supervised-learning task, target column, and a task-appropriate default
metric. Source and license details remain visible beside the selector. These are
educational defaults rather than claims that one metric or model is universally
best for the dataset.

| Dataset | Source | Standard task | Target | Default metric | What it demonstrates |
| --- | --- | --- | --- | --- | --- |
| Breast Cancer Wisconsin (Diagnostic) | UCI via scikit-learn | Binary classification | Diagnosis | ROC AUC | Numeric diagnostic features |
| Diabetes Progression | Efron et al. via scikit-learn | Regression | One-year disease progression | RMSE | Continuous clinical outcome |
| Wine Recognition | UCI via scikit-learn | Multiclass classification | Cultivar | Accuracy | Three-class numeric data |
| Banknote Authentication | UCI | Binary classification | Genuine or forged | ROC AUC | A larger binary example |
| Auto MPG | CMU StatLib via UCI | Regression | City-cycle MPG | RMSE | Missing numeric values and a categorical origin |
| Palmer Penguins | Palmer Station LTER | Multiclass classification | Species | Accuracy | Categorical features, missing values, and unequal class sizes |

The three external CSV snapshots are packaged with PSPSO, so examples do not
depend on a network connection. Attribution, download dates, licenses, and the
small Auto MPG loading transformation are recorded in the packaged dataset
README.

## Inspection

Before model selection, inspect row and column counts, inferred types,
missingness, duplicates, target distribution, numeric summaries, and proposed
split composition. Classification targets must contain the expected number of
classes; missing target values are rejected.

## Missing Values

Numeric columns support median, mean, most-frequent, and constant imputation.
Categorical columns support most-frequent and constant-label imputation.
Missingness indicators can preserve information about whether a value was
originally absent.

## Preprocessing

The third section of Data Setup exposes preparation rules even when the current
sample has no missing values, so the saved pipeline also covers future inputs.

- **Automatic:** infer numeric versus nominal from the column's data type.
- **Numeric:** interpret values as numbers; invalid numeric text is rejected.
- **Nominal:** one-hot encode unordered labels, including numeric category codes.
- **Ordinal:** enter categories from lowest to highest, one per line. Encoding
  uses that explicit order; unlisted values receive −1.

Unknown nominal labels produce all-zero one-hot columns. Missing-value rules
run before encoding. Turning off categorical encoding excludes nominal and
ordinal inputs; feature inclusion checkboxes provide individual control. The
prediction target is encoded separately for classification.

Numeric outliers can be kept, clipped to training-set IQR bounds (default
multiplier 1.5), or clipped to training-set percentile bounds (default 1st/99th).
Clipping changes feature values without dropping rows or modifying target
values. Data-profile outlier counts use the ordinary 1.5-IQR definition over the
source data for inspection; these descriptive counts are not training limits.
Imputation, clipping, encoding and scaling are refitted inside every training
fold and persisted with the selected model. Infinite numeric feature values are
treated as missing; a feature that is entirely missing remains in the pipeline.

For CLI/API specifications, the same options live under `preprocessing`:

```json
{
  "preprocessing": {
    "numeric_imputation": "median",
    "outlier_method": "iqr",
    "outlier_iqr_multiplier": 1.5,
    "features": {
      "region_code": {"kind": "nominal"},
      "priority": {"kind": "ordinal", "categories": ["low", "medium", "high"]}
    }
  }
}
```

Encoding follows scikit-learn's [one-hot](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.OneHotEncoder.html)
and [ordinal](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.OrdinalEncoder.html)
transformers.

- Scaling is recommended for SVM, linear models, and neural networks.
- Categorical encoding is required for estimators that consume numeric arrays.
- Ignored columns should include identifiers, post-outcome fields, and obvious
  leakage sources.
- Every transformer is fitted on training rows and reused for validation, test,
  prediction, and saved-model analysis.

Imported CSV files are copied into `.pspso/v1/datasets/` and identified by a
content fingerprint. Selecting a saved dataset therefore refers to a stable
local version rather than a changing external path.
