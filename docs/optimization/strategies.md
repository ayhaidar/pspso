# Search Strategies

## Particle Swarm Optimization

Each particle represents an encoded parameter position. Every iteration fits
each particle, updates its personal best, then moves it using inertia (`w`), a
cognitive coefficient (`c1`), and a social coefficient (`c2`).

Global topology follows the best particle in the full swarm. Ring-local topology
follows the best of each particle and its immediate neighbours. A run plans
`particles x iterations` candidates unless `max_trials`, timeout, cancellation, or
early stopping ends it sooner.

## Random Search

Random search samples independent candidates from the same typed domains. A
saved seed makes the sequence reproducible. It is the recommended baseline for
judging whether PSO adds value for a dataset and budget.

## Grid Search

Grid search evaluates every discrete grid value deterministically. `Choice` and
`IntRange` produce exact values. Float domains produce values according to their
precision or bounded log grid. Check `SearchSpace.grid_size` before launching a
large grid.

`runtime.trial_workers` bounds concurrent candidates. Each candidate evaluates
its folds sequentially. PSO waits for an iteration barrier before updating
positions; random and grid search use bounded batches. Estimator `n_jobs` is
one under candidate parallelism. Planned fits are candidates multiplied by fold
count, plus one final refit when enabled. Multiple recorded runs can separately
execute concurrently through `PSPSO_MAX_WORKERS`; one active run is the default.
