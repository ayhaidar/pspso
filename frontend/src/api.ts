import type { ApiSchema, DatasetPreview, ExampleDataset, ExperimentDetail, ExperimentSummary, Metadata, PredictionPreview, RunAnalysis, RunEvent, RunSnapshot, SavedDataset, ValidationResult, WorkflowDraft } from "./types";

const API = "/api/v1";

async function request<T>(url: string, init?: RequestInit): Promise<T> {
  const response = await fetch(url, init);
  const body = await response.json();
  if (!response.ok) {
    const detail = Array.isArray(body.detail)
      ? body.detail.map((item: { msg?: unknown }) => String(item.msg ?? "Invalid setting").replace(/^Value error, /, "")).join(" ")
      : typeof body.detail === "string" ? body.detail : JSON.stringify(body.detail ?? body);
    throw new Error(detail);
  }
  return body as T;
}

// Split controls remain the form's single source of truth for evaluation aliases.
const spec = (draft: WorkflowDraft): ApiSchema["ExperimentSpec"] => ({ ...draft, evaluation: {
  ...draft.evaluation, stratify: draft.split.stratify, test_size: draft.split.test_size, random_state: draft.split.random_state,
} });
const json = (method: string, body: unknown): RequestInit => ({ method, headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) });

export const api = {
  metadata: () => request<Metadata>(`${API}/estimators`),
  exampleDatasets: () => request<ExampleDataset[]>(`${API}/datasets/examples`),
  datasets: () => request<SavedDataset[]>(`${API}/datasets`),
  inspectDataset: (draft: WorkflowDraft) => request<DatasetPreview>(`${API}/datasets/inspect`, json("POST", { ...draft.dataset, task: draft.task, split: draft.split, evaluation: spec(draft).evaluation })),
  saveDataset: (name: string, csv_text: string) => request<SavedDataset>(`${API}/datasets`, json("POST", { name, csv_text })),
  experiments: () => request<ExperimentSummary[]>(`${API}/experiments`),
  experiment: (id: string) => request<ExperimentDetail>(`${API}/experiments/${id}`),
  createExperiment: (name: string, description: string) => request<ExperimentSummary>(`${API}/experiments`, json("POST", { name, description, tags: [] })),
  saveResultLayout: (id: string, tools: string[]) => request<ExperimentDetail>(`${API}/experiments/${id}/result-layout`, json("PUT", { tools })),
  runs: () => request<RunSnapshot[]>(`${API}/runs`),
  validateWorkflow: (stage: "data" | "model" | "search" | "full", draft: WorkflowDraft) => request<ValidationResult>(`${API}/workflow/validate`, json("POST", { stage, request: spec(draft) })),
  validateRun: (draft: WorkflowDraft) => request<ValidationResult>(`${API}/runs/validate`, json("POST", spec(draft))),
  createRun: (draft: WorkflowDraft) => request<RunSnapshot>(`${API}/runs`, json("POST", spec(draft))),
  snapshot: (id: string) => request<RunSnapshot>(`${API}/runs/${id}`),
  history: (id: string, after = 0) => request<{ run_id: string; events: RunEvent[] }>(`${API}/runs/${id}/history?after=${after}`),
  analysis: (id: string) => request<RunAnalysis>(`${API}/runs/${id}/analysis`),
  predictions: (id: string, split: string) => request<PredictionPreview>(`${API}/runs/${id}/predictions?split=${split}&limit=50`),
  featureImportance: (id: string) => request<RunAnalysis["feature_importance"]>(`${API}/runs/${id}/feature-importance`, { method: "POST" }),
  artifacts: (id: string) => request<ApiSchema["ArtifactResponse"]>(`${API}/runs/${id}/artifacts`),
  cancel: (id: string) => request<{ run_id: string; cancel_requested: boolean }>(`${API}/runs/${id}/cancel`, { method: "POST" }),
  tournament: (id: string, runs: WorkflowDraft[]) => request<ApiSchema["TournamentResponse"]>(`${API}/experiments/${id}/tournament`, json("POST", { runs: runs.map(spec) })),
  exportUrl: (id: string, kind: "spec" | "result" | "analysis" | "events" | "predictions" | "metrics" | "manifest" | "model" | "environment" | "split_indices" | "log", split?: string) => `${API}/runs/${id}/exports/${kind}${split ? `?split=${encodeURIComponent(split)}` : ""}`,
  retry: (id: string) => request<RunSnapshot>(`${API}/runs/${id}/retry`, { method: "POST" })
};
