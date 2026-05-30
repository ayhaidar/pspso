# Frontend Backend Link

The dashboard uses a separate React/Vite development server and FastAPI backend.
The frontend calls relative `/api/...` URLs; Vite proxies those calls to
FastAPI.

## Development Proxy

`frontend/vite.config.ts` contains:

```ts
server: {
  proxy: {
    "/api": "http://127.0.0.1:8000"
  }
}
```

This means the browser talks to Vite at `http://127.0.0.1:5173`, while API
requests are forwarded to `http://127.0.0.1:8000`.

## Successful Run

```mermaid
sequenceDiagram
    participant User
    participant React
    participant FastAPI
    participant Optimizer

    User->>React: Configure run
    React->>FastAPI: POST /api/runs/validate
    FastAPI-->>React: { valid: true }
    React->>FastAPI: POST /api/runs
    FastAPI-->>React: { run_id, status }
    React->>FastAPI: GET /api/runs/{run_id}/events
    FastAPI->>Optimizer: Execute run in background
    Optimizer-->>FastAPI: ProgressEvent
    FastAPI-->>React: SSE trial/best/completed events
    React->>FastAPI: GET /api/runs/{run_id}/result
    FastAPI-->>React: OptimizationResult
```

## Validation Failure

```mermaid
sequenceDiagram
    participant React
    participant FastAPI

    React->>FastAPI: POST /api/runs/validate
    FastAPI-->>React: 400 grouped validation errors
    React-->>React: Show errors and keep Start disabled
```

## Training Failure After Start

```mermaid
sequenceDiagram
    participant React
    participant FastAPI
    participant Optimizer

    React->>FastAPI: POST /api/runs
    FastAPI-->>React: { run_id }
    React->>FastAPI: GET /api/runs/{run_id}/events
    FastAPI->>Optimizer: Start trials
    Optimizer-->>FastAPI: trial_failed
    Optimizer-->>FastAPI: run_failed
    FastAPI-->>React: SSE failure events
    React-->>React: Show trial error and suggested fix
```

## Endpoint Responsibilities

| Endpoint | Purpose |
| --- | --- |
| `GET /api/estimators` | Frontend metadata for tasks, metrics, params, dependencies. |
| `POST /api/runs/validate` | Validate the whole config before a run starts. |
| `POST /api/runs` | Start a background optimization run. |
| `GET /api/runs/{run_id}/events` | Stream live progress over Server-Sent Events. |
| `GET /api/runs/{run_id}` | Polling fallback and run snapshot. |
| `GET /api/runs/{run_id}/result` | Final optimization result. |
