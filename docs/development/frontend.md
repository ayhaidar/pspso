# Frontend Development

The frontend is a React, TypeScript, Vite, React Router, and Apache ECharts
application under `frontend/`.

## Structure

| Location | Role |
| --- | --- |
| `src/App.tsx` | Route definitions for the six stages. |
| `src/workflow.tsx` | Autosaved draft, metadata, selected run, SSE, and polling state. |
| `src/api.ts` | Typed FastAPI client. |
| `src/pages/` | One focused page for each workflow stage. |
| `src/components/` | Navigation, charts, execution canvas, and timeline. |
| `src/api-schema.d.ts` | Generated OpenAPI types, checked for drift in CI. |
| `src/types.ts` | Form defaults and view types derived from API contracts. |

## Connection Flow

1. The workflow provider loads model metadata, datasets, and experiments.
2. Setup edits update `WorkflowDraft`, which is saved to local storage.
3. Each guided stage calls `POST /api/v1/workflow/validate` before continuing.
4. Search Engine submits the frozen draft to `POST /api/v1/runs`.
5. Live Experiments consumes `/api/v1/runs/{run_id}/events` with `EventSource`.
6. A disconnected event stream switches to snapshot/history polling.
7. Results loads saved analysis and predictions; History loads every persisted run.

The frontend always uses relative `/api` paths. During development, Vite proxies
them to the configured FastAPI URL.

Regenerate contracts from the repository root with
`uv run python scripts/export_openapi.py .artifacts/openapi.json`, then run
`npm run api:types` in `frontend/`. Vitest, React Testing Library and MSW cover
gates, event replay, reconnects, tournaments, result tools and failures.
`npm run test:e2e` builds the production frontend and runs Playwright against an
isolated local service. Browser scenarios cover all three tasks, cancellation,
retry, refresh and artifact downloads. Route components and modular ECharts load
on demand; the selected run ID is preserved in the route.

## Result Tools

Analytical charts use ECharts. The execution canvas uses SVG nodes whose states
are derived from trial events. Tool availability must be derived from the task
and saved analysis payload; unsupported tools remain disabled with an explicit
reason instead of rendering an empty graph.

## Commands

```bash
cd frontend
npm install
npm run dev
npm run build
```
