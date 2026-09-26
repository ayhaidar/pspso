# PSPSO 1.0 development completion record

Status: version 1.0 is still under active development on the `dev` branch. This
record tracks completed implementation and verification work before the release
is merged and published.

Distribution target: the existing `pspso` project on PyPI. Public users install
with `uv tool install pspso` or `python -m pip install --upgrade pspso`.
GitHub verifies every candidate and publishes a non-prerelease GitHub Release
through PyPI Trusted Publishing. See `docs/development/releasing.md` for the
release procedure and one-time repository settings.

Scope: the supplied PSPSO 1.0 Finalization Plan, including the Python API, CLI,
FastAPI service, six-stage React dashboard, reproducible evaluation, packaging,
documentation, and all release gates. A checkbox is complete only after direct
verification; existing code alone is insufficient evidence.

## Execution reliability

- [x] API lifespan exclusively owns a durable manager; repository-only CLI
  commands do not claim a service or change live run state.
- [x] Atomic SQLite service leases, PID identity, heartbeats, attempts, queue
  recovery, timeout/cancellation state, parent linkage, and queue positions.
- [x] CLI start/retry/cancel use /api/v1, --api-url and PSPSO_API_URL.
- [x] Supervised API-free foreground --standalone --wait, useful exit codes.
- [x] Queued jobs survive restarts; interrupted attempts remain historical.
- [x] Cooperative cancellation, bounded process-tree cleanup, safe shutdown,
  process-start failure handling, and exactly one terminal outcome.
- [x] Persisted worker output plus structured run_log events.

## Scientific evaluation and optimization

- [x] Evaluation settings cover protocol, folds, shuffle, stratification, test
  share, seed, positive class, threshold, and final-refit policy.
- [x] Dashboard defaults to five-fold CV and untouched holdout test.
- [x] Independent preprocessing inside each fold; exact partition/fold indices
  persisted and used when reading results (including an unspecified seed).
- [x] Individual fold metrics, mean and standard deviation drive optimization.
- [x] Selected configuration refits on all development rows before test
  evaluation; saved models/preprocessors generate predictions.
- [x] Bounded candidate workers for all strategies; PSO iteration barrier;
  model threads constrained under candidate parallelism.
- [x] Accurate candidate/fold/iteration/worker/fit totals and progress.
- [x] Estimator objectives validated against task/target, positive-target
  booster objectives, and categorized failures.

## API and dashboard

- [x] Constrained requests/responses, search-space discriminators, canonical
  metric/status types, typed event payloads, generated TypeScript API types.
- [x] SSE IDs, heartbeats, Last-Event-ID replay, client deduplication and
  incremental history; refresh and reconnection preserve the active run.
- [x] Full compatible-model tournament under frozen data/splits/metric/budget.
- [x] Full prediction CSV, metrics JSON, spec, events, artifact manifest and
  model downloads; serialization warnings visible in the dashboard.
- [x] Configured local CORS and loopback default bind address.
- [x] Modular charts and lazy analytical views, checked production bundles.
- [x] Six workflow stages and accurate live candidate/model-fit presentation.

## Packaging, branding and documentation

- [x] Numbered, non-destructive v1 migrations; pre-1.0 workspaces untouched.
- [x] React static assets and py.typed in fresh 1.0.0 wheel; SPA fallback.
- [x] Environment records dependency/Python/platform/device/source versions,
  actual seeds and dataset fingerprint.
- [x] Canonical SVG logo/marks throughout; original raster preserved at
  docs/assets/pspso-logo-original.png and ambiguous root copy removed.
- [x] Documentation matches ownership, CLI/API, CV/refit, concurrency,
  recovery, artifacts, metrics and optional dependencies.
- [x] Diagrams for ownership, folds, parallel particles, SSE and artifacts.
- [x] Test-results documentation generated from machine-readable test reports;
  CI regenerates the same page on each verification run.
- [x] Clean dist, rebuilt final packages, fresh-wheel runtime verification.

## Required verification

- [x] Deterministic manager tests: leases, restart/recovery, cancellation,
  timeout, retry, process failure and terminal idempotence.
- [x] Live-service CLI tests, including nonmutating list/show during execution.
- [x] Deadline-based test polling up to 30 seconds.
- [x] Leakage/split/refit/positive-class/aggregation/parallel-candidate tests.
- [x] Actual lightweight training for every built-in model family.
- [x] Separate optional-backend CI jobs for XGBoost, LightGBM and PyTorch.
- [x] Vitest/RTL/MSW: workflow gates, recovery, tournaments, results, failures.
- [x] Playwright: regression, binary/multiclass, cancel, retry, refresh, downloads.
- [x] Zero project mypy errors, Ruff lint/format, overall coverage >=80% and
  manager/tracking/worker lifecycle coverage >=90%.
- [x] CI Python 3.10-3.12, frontend tests/build, strict MkDocs, package build,
  dependency audits and installed-wheel smoke test.
- [x] Installed wheel serves dashboard/docs, supports modern notebook imports,
  rejects legacy imports and installs no legacy scripts; no Node required.
- [x] Clean-checkout verification before any release tag.

## Evidence and findings

The initial audit found an unsafe Windows process-existence probe that could
terminate live Python processes. It now uses psutil. Managed execution was
rebuilt around atomic leases, persisted attempts and OS process containment.
Repository-only CLI operations do not start a manager. Regression, binary and
multiclass browser runs now use saved models and exact partitions for results.

Verified from an isolated clean Git snapshot on 6 September 2026:

- Python 3.12 with all model engines: 139 passed, no failures or skips.
- Python 3.10 and 3.11: each passed 127 core tests, with 12 optional-engine tests
  skipped because those engines were not installed in the core environments.
- Overall coverage: 88.36%; manager 93.18%, tracking 92.62%, worker 92.73%.
- Frontend: 16 Vitest/RTL/MSW tests pass. Four Playwright scenarios cover all
  tasks, refresh recovery, cancellation, retry and full artifact downloads.
- All built-in model families trained, predicted and serialized, including
  installed XGBoost, LightGBM and PyTorch.
- Ruff lint/format and mypy pass; strict MkDocs builds. OpenAPI regeneration
  matches the checked-in TypeScript declarations and the dependency lock checks.
- npm and Python dependency audits report no known vulnerabilities after patched
  dependency upgrades.
- Lazy chart chunks are about 422 kB and 188 kB, with no size or circular-chunk
  warnings. The application entry is about 210 kB before compression.
- The source archive contains frontend sources, test fixtures and release tools.
  A fresh Python 3.12 wheel installation serves the dashboard, SPA routes and API
  docs, exposes modern CLI entries and completes notebook training without Node.

The user's original working tree and its pre-existing changes are preserved.
Local verification ran on Windows; the GitHub workflow additionally configures
Ubuntu and separate optional-engine jobs.

Machine-readable evidence is retained in `.artifacts/release-checkout/.artifacts/`
and the readable test report is `docs/quality/test-results.md`. Candidate packages
in `dist/` are retained for internal checks and release publication.

## Data Setup and overview follow-up — 6 September 2026

The subsequent dashboard update adds an overview landing page and three
numbered Data Setup sections: source/target, profile/evaluation, and feature
engineering. Evaluation now exposes ordinary holdout and five-fold CV directly
beside the data profile, plus chronological holdout or expanding-window CV with
an optional time column and gap. Preprocessing supports nominal and ordinal
roles, explicit ordered categories, missing-value rules and training-only
IQR/percentile clipping. Saved predictions reuse those exact pipelines.

Current version verification: 158 Python tests, 25 frontend tests and
seven browser workflows pass. Overall coverage is 88.70%; manager and tracking
remain above 92%, and worker coverage is 91.23%. Ruff, mypy and the production frontend build pass. Browser
coverage includes a dated CSV with missing values, nominal numeric codes,
ordinal categories, clipping, a time gap and saved untouched-test predictions.
The initial clean-snapshot record above remains historical; current test reports
are in `.artifacts/` and `docs/quality/test-results.md`.

The public documentation now assumes installation by package name, includes a
complete CLI reference and deploys from `main` with GitHub Pages. A protected
release workflow verifies an exact `v<version>` tag on `main` before publishing
through the existing PyPI project's Trusted Publisher.

The example catalogue now adds bundled Banknote Authentication, Auto MPG and
Palmer Penguins data alongside the original scikit-learn examples. Every entry
records its source, license, standard task, target and default metric. Browser
coverage verifies that those defaults travel from Data Setup into the Search
step, and the installed-wheel smoke test requires all three CSV snapshots.
