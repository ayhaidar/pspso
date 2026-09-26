# Model Recipe Plugins

The dashboard only lists model recipes registered by trusted Python code on the
local machine. It does not execute code pasted into the browser.

Create an installed module such as `my_project.pspso_recipes`:

```python
from sklearn.ensemble import AdaBoostClassifier

from pspso import FloatRange, IntRange, SearchSpace, register_recipe

register_recipe(
    "ada_boost",
    AdaBoostClassifier,
    tasks=["binary_classification", "multiclass_classification"],
    description="Local AdaBoost recipe for comparison experiments.",
    fixed_params={"binary_classification": {"random_state": 42}},
    search_spaces={
        "binary_classification": SearchSpace({
            "n_estimators": IntRange(25, 150),
            "learning_rate": FloatRange(0.01, 1.0, precision=2),
        })
    },
)
```

Configure the module before starting the dashboard or CLI:

```bash
set PSPSO_RECIPE_PLUGINS=my_project.pspso_recipes
uv run pspso-dashboard
```

Use a comma-separated module list for several trusted plugin modules. Plugins
run on the local backend, so treat them like any other installed Python code.
