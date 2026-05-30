# Frontend Development

The frontend lives in `frontend/` and is built with React, TypeScript, and Vite.

## Important Files

| File | Role |
| --- | --- |
| `frontend/src/App.tsx` | Dashboard state, forms, API calls, event handling. |
| `frontend/src/styles.css` | Dashboard layout and visual styling. |
| `frontend/vite.config.ts` | Dev proxy from `/api` to FastAPI. |

## State Flow

1. Fetch estimator metadata from `GET /api/estimators`.
2. Keep task, metric, estimator, fixed params, and search space synchronized
   with the metadata.
3. Build a `RunRequest`.
4. Validate with `POST /api/runs/validate`.
5. Start with `POST /api/runs`.
6. Subscribe to `GET /api/runs/{run_id}/events`.
7. Fall back to polling `GET /api/runs/{run_id}` if SSE disconnects.

## UI Principles

- Do not let users start a run with stale estimator params.
- Explain every task and metric near the controls.
- Keep fixed params separate from tunable params.
- Prefer structured range/choice controls over raw JSON.
- Keep advanced JSON editing available for expert use.

## Event Handling

The frontend should treat SSE as live updates and snapshots as durable state.
If an event is missed, polling can recover the current run state.
