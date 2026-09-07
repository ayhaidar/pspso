import { fireEvent, render, screen, waitFor } from "@testing-library/react";
import { MemoryRouter, useLocation } from "react-router-dom";
import { afterAll, afterEach, beforeAll, beforeEach, expect, it, vi } from "vitest";
import { HttpResponse, http, delay } from "msw";
import { setupServer } from "msw/node";
import { SearchPage } from "./SearchPage";
import { ResultsPage } from "./ResultsPage";
import { draftFromSpec } from "../workflow";
import type { RunAnalysis, RunEvent, WorkflowDraft } from "../types";

vi.mock("../workflow", async (original) => ({
  ...await original<typeof import("../workflow")>(), useWorkflow: () => context,
}));
vi.mock("../components/Charts", () => Object.fromEntries(
  ["ConfusionChart", "ConvergenceChart", "ImportanceChart", "PrecisionRecallChart", "RegressionChart", "RocChart"].map((name) => [name, () => <div>Chart</div>]),
));
const defaults = { fixed_params: { n_jobs: 1, n_estimators: 5 }, search_space: { n_estimators: { type: "choice", values: [5] } } };
const model = (label: string) => ({ label, tasks: ["regression"], dependency: { installed: true }, defaults: { regression: defaults } });
const context = {
  draft: draftFromSpec({ schema_version: 1, task: "regression", metric: "rmse", estimator: "random_forest", strategy: "random" }),
  metadata: { estimators: { random_forest: model("Random Forest"), extra_trees: model("Extra Trees") }, tasks: { regression: { label: "Regression", metrics: ["rmse", "mae"] } }, metrics: { rmse: { label: "RMSE" }, mae: { label: "MAE" } } },
  preview: null, experiments: [], busy: null, validations: {},
  updateDraft: vi.fn(), validateStage: vi.fn(), createExperiment: vi.fn(), startRun: vi.fn(), refreshCollections: vi.fn(), loadRun: vi.fn(),
  run: { run_id: "run-1", experiment_id: "experiment-1", experiment_name: "Tournament", estimator: "random_forest", strategy: "random", status: "completed", artifacts: { model: "model.joblib" } as Record<string, string>, request: { schema_version: 1, task: "regression", metric: "rmse", estimator: "random_forest", strategy: "random", evaluation: { protocol: "holdout", folds: 5, shuffle: true, positive_label: null, decision_threshold: 0.5, refit_best: true } } },
  events: [{ sequence_number: 1, timestamp: "2026-01-01T00:00:00Z", type: "run_warning", payload: { reason: "artifact_serialization", message: "Model serialization failed: custom object" } }] as RunEvent[],
  analysis: { task: "regression", metric: "rmse", train: { metrics: {} }, validation: { metrics: { rmse: 0.8 } }, test: { metrics: { rmse: 0.9 } }, feature_importance: { available: false, items: [], reason: "Unavailable" } } as RunAnalysis,
};
let submitted: WorkflowDraft[] = [];
let fail = false;
const server = setupServer(
  http.post("*/api/v1/experiments/experiment-1/tournament", async ({ request }) => {
    submitted = (await request.json() as { runs: WorkflowDraft[] }).runs;
    return fail ? HttpResponse.json({ detail: "A model failed validation" }, { status: 400 }) : HttpResponse.json({ experiment_id: "experiment-1", runs: [{ run_id: "created-1" }] });
  }),
  http.get("*/api/v1/experiments/experiment-1", () => HttpResponse.json({ result_layout: ["summary", "predictions"] })),
  http.put("*/api/v1/experiments/experiment-1/result-layout", () => HttpResponse.json({ result_layout: [] })),
  http.get("*/api/v1/runs/run-1/predictions", ({ request }) => fail
    ? HttpResponse.json({ detail: "Saved model is unavailable" }, { status: 400 })
    : HttpResponse.json({ split: new URL(request.url).searchParams.get("split"), task: "regression", row_count: 1, preview_rows: [{ row_index: 8, actual: 10, predicted: 11, residual: -1, features: {} }] })),
);
beforeAll(() => server.listen({ onUnhandledRequest: "error" }));
afterEach(() => server.resetHandlers());
afterAll(() => server.close());
beforeEach(() => {
  vi.clearAllMocks(); fail = false; submitted = [];
  localStorage.clear();
  context.draft.experiment_id = "experiment-1";
  context.run.artifacts = { model: "model.joblib" };
  context.run.request.evaluation = { protocol: "holdout", folds: 5, shuffle: true, positive_label: null, decision_threshold: 0.5, refit_best: true };
  context.events = [{ sequence_number: 1, timestamp: "2026-01-01T00:00:00Z", type: "run_warning", payload: { reason: "artifact_serialization", message: "Model serialization failed: custom object" } }];
  context.analysis.validation = { metrics: { rmse: 0.8 } };
  context.analysis.test = { metrics: { rmse: 0.9 } };
});
function Location() { return <output data-testid="location">{useLocation().pathname}</output>; }
function mount(element: React.ReactNode) { render(<MemoryRouter>{element}<Location/></MemoryRouter>); }

it("allows choosing the optimization metric for the current task", () => {
  mount(<SearchPage/>);
  fireEvent.change(screen.getByLabelText(/Optimization metric/), { target: { value: "mae" } });
  expect(context.updateDraft).toHaveBeenCalledWith({ metric: "mae" });
});

it("explains and selects the saved-model behavior for cross-validation", () => {
  mount(<SearchPage/>);
  const refit = screen.getByRole("button", { name: /Refit one winner/ });
  const evaluated = screen.getByRole("button", { name: /Keep evaluated models/ });
  expect(refit).toHaveAttribute("aria-pressed", "true");
  expect(refit).toHaveTextContent(/one newly fitted model/i);
  expect(evaluated).toHaveTextContent(/5-model ensemble/i);
  expect(evaluated).toHaveTextContent(/different predictor from a single refitted model/i);
  fireEvent.click(evaluated);
  expect(context.updateDraft).toHaveBeenCalledWith({ evaluation: { ...context.draft.evaluation, refit_best: false } });
});

it("creates compatible tournament recipes with a frozen dataset, evaluation and budget", async () => {
  mount(<SearchPage/>);
  fireEvent.click(screen.getByLabelText("Random Forest"));
  expect(screen.getByRole("button", { name: "Start experiment run" })).toBeDisabled();
  fireEvent.click(screen.getByLabelText("Extra Trees"));
  fireEvent.click(screen.getByRole("button", { name: "Start 2-model tournament" }));
  await waitFor(() => expect(screen.getByTestId("location")).toHaveTextContent("/live/created-1"));
  expect(submitted).toHaveLength(2);
  for (const key of ["dataset", "task", "metric", "split", "evaluation", "runtime"] as const) expect(submitted[0][key]).toEqual(submitted[1][key]);
  expect(submitted.map((run) => run.estimator)).toEqual(["random_forest", "extra_trees"]);
  expect(submitted[0].fixed_params).toEqual({ n_jobs: 1 });
});

it("shows tournament validation failures and keeps the setup editable", async () => {
  fail = true; mount(<SearchPage/>);
  fireEvent.click(screen.getByLabelText("Random Forest")); fireEvent.click(screen.getByLabelText("Extra Trees"));
  fireEvent.click(screen.getByRole("button", { name: "Start 2-model tournament" }));
  await waitFor(() => expect(screen.getByRole("alert")).toHaveTextContent("A model failed validation"));
  expect(screen.getByRole("button", { name: "Start 2-model tournament" })).toBeEnabled();
});

it("loads saved predictions, downloads the selected test partition and shows artifact warnings", async () => {
  mount(<ResultsPage/>);
  await screen.findByRole("button", { name: "Load rows" });
  expect(screen.getByText(/One newly refitted model · 1 additional fit/)).toBeVisible();
  expect(screen.getByText(/test result can differ from the candidate score because it is a new fit/i)).toBeVisible();
  fireEvent.click(screen.getByRole("button", { name: "Load rows" }));
  await screen.findByRole("cell", { name: "11.0000" });
  expect(screen.getByRole("alert")).toHaveTextContent("Model serialization failed");
  expect(screen.getByRole("link", { name: "predictions", hidden: true })).toHaveAttribute("href", "/api/v1/runs/run-1/exports/predictions?split=test");
});

it("waits for saved report settings before allowing tools to be toggled", async () => {
  server.use(http.get("*/api/v1/experiments/experiment-1", async () => {
    await delay(40);
    return HttpResponse.json({ result_layout: ["summary"] });
  }));
  mount(<ResultsPage/>);
  const predictions = screen.getByRole("button", { name: /Prediction table/ });
  expect(predictions).toBeDisabled();
  await waitFor(() => expect(predictions).toBeEnabled());
  fireEvent.click(predictions);
  expect(await screen.findByRole("button", { name: "Load rows" })).toBeVisible();
});

it("explains older CV runs without a model and prepares a repeatable saved-model run", async () => {
  context.run.artifacts = {};
  context.run.request.evaluation = { protocol: "cross_validation", folds: 5, shuffle: true, positive_label: null, decision_threshold: 0.5, refit_best: false };
  context.events = [
    { sequence_number: 1, timestamp: "2026-01-01T00:00:00Z", type: "best_updated", payload: { trial_id: 10, best_cost: 0.0041 } },
    { sequence_number: 2, timestamp: "2026-01-01T00:00:01Z", type: "trial_completed", payload: { trial_id: 20, cost: 0.0046 } },
  ];
  context.analysis.validation = { metrics: { rmse: 0.8 }, cross_validation: { mean: 0.8, standard_deviation: 0.1, folds: [{ fold: 1, metric: 0.8, cost: 0.8, train_metric: 0.7, train_metrics: {}, validation_metrics: {} }] } };
  delete context.analysis.test;
  mount(<ResultsPage/>);
  expect(await screen.findByText(/Trial 10 · cost 0.0041/)).toBeVisible();
  expect(screen.getByText(/Trial 20 finished later at cost 0.0046/)).toBeVisible();
  expect(screen.getByText(/earlier evaluation-only behavior/)).toBeVisible();
  expect(screen.queryByRole("link", { name: "model", hidden: true })).not.toBeInTheDocument();
  expect(screen.getByRole("button", { name: /Prediction table/ })).toBeDisabled();
  expect(screen.getByRole("heading", { name: "Cross-validation evidence" })).toBeVisible();
  fireEvent.click(screen.getByRole("button", { name: /Prepare the same run/ }));
  expect(context.updateDraft).toHaveBeenCalled();
  expect(screen.getByTestId("location")).toHaveTextContent("/search");
});
