"""Modern pspso API example for regression."""

from sklearn.datasets import load_diabetes
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

from pspso import EstimatorConfig, IntRange, OptimizationConfig, PSPSOOptimizer, SearchSpace

X, y = load_diabetes(return_X_y=True)
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

optimizer = PSPSOOptimizer(
    estimator=EstimatorConfig(
        factory=RandomForestRegressor,
        fixed_params={"random_state": 42, "n_jobs": -1},
    ),
    search_space=SearchSpace(
        {
            "n_estimators": IntRange(10, 30),
            "max_depth": IntRange(2, 6),
        }
    ),
    config=OptimizationConfig(task="regression", metric="rmse", strategy="random", max_trials=6),
)

result = optimizer.optimize(X_train, y_train, X_val, y_val)
print(result.best_params)
print(result.best_metric)
