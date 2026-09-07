# Notebook Workflow

## Quick In-Process Run

`optimize()` is the shortest supported notebook path. It does not start FastAPI
or write to disk unless tracking is requested.

```python
from sklearn.datasets import load_diabetes
from pspso import IntRange, OptimizationConfig, SearchSpace, optimize

X, y = load_diabetes(return_X_y=True)
search_space = SearchSpace({
    "n_estimators": IntRange(20, 80),
    "max_depth": IntRange(2, 10),
})
config = OptimizationConfig(
    task="regression",
    metric="rmse",
    strategy="pso",
    n_particles=4,
    n_iterations=3,
    random_state=42,
)
result = optimize(
    X,
    y,
    estimator="random_forest",
    search_space=search_space,
    config=config,
)
result
```

Use `result.summary()`, `result.trials_frame()`, `result.predict(X)`, and
`result.evaluate(X, y)` to inspect the outcome. `predict_proba()` is available
only when the fitted estimator supports probabilities.

## Tracked Notebook Run

```python
from pspso import TrackingConfig

result = optimize(
    X,
    y,
    estimator="random_forest",
    search_space=search_space,
    config=config,
    tracking=TrackingConfig(
        experiment_name="Diabetes baselines",
        run_name="PSO notebook run",
        tags=("notebook", "baseline"),
        snapshot_data=True,
    ),
)
print(result.run_id, result.artifacts)
```

Tracked calls execute in the notebook process and write to `.pspso/v1/`.
They appear in the same History page as GUI and CLI runs.

## Full Example

```python
--8<-- "examples/python/modern_classification.py"
```
