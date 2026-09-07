# Typed Search Spaces

PSPSO 1.0 accepts only explicit domain objects:

```python
from pspso import Choice, FloatRange, IntRange, LogFloatRange, SearchSpace

space = SearchSpace({
    "kernel": Choice(["linear", "rbf"]),
    "max_depth": IntRange(2, 12),
    "subsample": FloatRange(0.7, 1.0, precision=2),
    "learning_rate": LogFloatRange(0.001, 0.3, precision=5),
})
```

| Domain | Use |
| --- | --- |
| `Choice` | Categories, booleans, and explicit alternatives. |
| `IntRange` | Inclusive integer values. |
| `FloatRange` | Linear continuous values rounded to a precision. |
| `LogFloatRange` | Positive values spanning several orders of magnitude. |

`decode()` converts an optimizer position to estimator parameters. `encode()`
performs the reverse conversion. `iter_grid()` produces decoded combinations,
`grid_size` reports their count, and `to_schema()` produces the API v1 JSON
representation.

List-based declarations are intentionally rejected because their meaning was
ambiguous and could silently interpret categories or numeric precision
incorrectly.
