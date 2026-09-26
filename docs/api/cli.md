# CLI reference

PSPSO installs two commands:

- `pspso` manages datasets, experiments and runs.
- `pspso-dashboard` starts the dashboard and its local run service.

Install them as an isolated tool with `uv tool install pspso`, or install PSPSO
in a Python environment with `python -m pip install --upgrade pspso`. In an
existing uv project that uses `uv add pspso`, prefix commands with `uv run`.

## `pspso`

```text
pspso [--api-url URL] COMMAND
pspso --version
```

`--api-url URL` selects the dashboard service used by managed start, cancel and
retry operations. Its default is `PSPSO_API_URL`, or
`http://127.0.0.1:8000` when the variable is unset. Place this global option
before the command name.

### Datasets

```text
pspso dataset import PATH [--name NAME]
pspso dataset list
```

`dataset import` reads a UTF-8 CSV file, fingerprints it, stores one local copy
and records a preview. `--name` sets its display name; the filename stem is used
when it is omitted. `dataset list` prints all datasets in the current workspace.

### Experiments

```text
pspso experiment create NAME [--description TEXT] [--tag TAG ...]
pspso experiment list
```

Repeat `--tag` to attach more than one tag. Experiments group related runs and
remain visible to the dashboard and CLI.

### Runs

```text
pspso run validate SPEC
pspso run start SPEC [--wait] [--standalone]
pspso run list
pspso run show RUN_ID
pspso run cancel RUN_ID
pspso run retry RUN_ID [--wait]
```

`SPEC` may be JSON, YAML or YML and must follow `ExperimentSpec` schema version
1. `run validate` checks the schema, data, model, metric, search space,
evaluation rows and optional dependencies without starting a run.

Managed `run start` sends the specification to the dashboard service and
returns after queueing. Add `--wait` to keep the terminal attached until the run
finishes. `--standalone` runs a supervised worker in the foreground without
HTTP and therefore requires `--wait`.

`run show` includes the run snapshot and attempt history. `run cancel` and
`run retry` require the dashboard service because it owns managed workers.
Retry creates a new run linked to the original one.

## `pspso-dashboard`

```text
pspso-dashboard [--host ADDRESS] [--port PORT] [--reload]
pspso-dashboard --version
```

| Option | Default | Meaning |
| --- | --- | --- |
| `--host` | `127.0.0.1` | Address used by the local web service. |
| `--port` | `8000` | HTTP port for the dashboard and API. |
| `--reload` | off | Restart when source files change; intended for development. |

Open `http://127.0.0.1:8000` after the service starts. OpenAPI documentation is
available at `http://127.0.0.1:8000/api/v1/docs`.

The default address keeps the single-user dashboard on the current computer.
Binding to a non-loopback address such as `0.0.0.0` is supported and prints a
warning because PSPSO does not add authentication or HTTPS.

## Workspace and output

PSPSO stores data in `.pspso/v1` under the current working directory. Set
`PSPSO_HOME` to use an absolute or shared workspace path. The dashboard and CLI
see the same history when they use the same workspace.

CLI results are JSON on standard output. Actionable errors are written to
standard error.

| Exit status | Meaning |
| --- | --- |
| `0` | The command completed successfully. |
| `1` | Validation failed, or a waited run did not complete successfully. |
| `2` | The command, input, service request or local operation failed. |
| `130` | The foreground command was interrupted with Ctrl+C. |

## Examples

```bash
pspso --version
pspso dataset import customers.csv --name "Customer churn v1"
pspso experiment create "Churn study" --tag baseline --tag production-data
pspso run validate experiment.yaml
pspso run start experiment.yaml --wait
pspso run show <run_id>
```

For a service on another local port:

```bash
pspso-dashboard --port 8002
pspso --api-url http://127.0.0.1:8002 run start experiment.json --wait
```

The [CLI workflow](../guides/cli.md) provides a complete specification and a
first run from start to finish.
