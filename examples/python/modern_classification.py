"""Modern pspso API example for binary classification."""

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from pspso import Choice, FloatRange, OptimizationConfig, SearchSpace, optimize

X, y = load_breast_cancer(return_X_y=True)
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_val = scaler.transform(X_val)

result = optimize(
    X_train,
    y_train,
    estimator="svm",
    search_space=SearchSpace(
        {
            "kernel": Choice(["linear", "rbf"]),
            "C": FloatRange(0.1, 2.0, precision=1),
            "gamma": FloatRange(0.1, 1.0, precision=1),
        }
    ),
    config=OptimizationConfig(
        task="binary_classification",
        metric="roc_auc",
        strategy="random",
        max_trials=6,
        random_state=42,
    ),
    X_validation=X_val,
    y_validation=y_val,
)

print(result.best_params)
print(result.best_metric)
