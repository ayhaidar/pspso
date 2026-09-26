import { createContext, ReactNode, useContext, useEffect, useMemo, useRef, useState } from "react";
import { api } from "./api";
import type { ApiSchema, JsonValue, DatasetPreview, ExampleDataset, ExperimentSummary, Metadata, MetricId, RunAnalysis, RunEvent, RunSnapshot, SavedDataset, Task, ValidationResult, WorkflowDraft } from "./types";

const STORAGE_KEY = "pspso.workflow-draft.api-v1";
const defaultDraft: WorkflowDraft = {
  schema_version: 1,
  experiment_id: null,
  dataset: { source: "example", name: "breast_cancer", csv_text: null, dataset_id: null, target_column: "target" },
  task: "binary_classification",
  metric: "roc_auc",
  estimator: "svm",
  fixed_params: {},
  search_space: {},
  strategy: "random",
  split: { validation_size: 0.2, test_size: 0.2, random_state: 42, stratify: true, method: "random", time_column: null, gap: 0 },
  preprocessing: {
    scale_numeric: true,
    encode_categorical: true,
    ignored_columns: [],
    numeric_imputation: "median",
    categorical_imputation: "most_frequent",
    numeric_fill_value: 0,
    categorical_fill_value: "Missing",
    add_missing_indicators: false,
    features: {}, outlier_method: "none", outlier_iqr_multiplier: 1.5,
    outlier_lower_quantile: 0.01, outlier_upper_quantile: 0.99
  },
  pso: { particles: 5, iterations: 4, c1: 1.49618, c2: 1.49618, w: 0.7298, topology: "global" },
  evaluation: { protocol: "cross_validation", folds: 5, shuffle: true, positive_label: null, decision_threshold: 0.5, refit_best: true },
  runtime: { max_trials: 6, timeout_seconds: null, early_stopping_rounds: null, trial_workers: 2, verbose: 0 }
};

export function dashboardSpec(run: RunSnapshot | null): ApiSchema["ExperimentSpec"] | null {
  return run?.request && !("source" in run.request) ? run.request : null;
}

export function draftFromSpec(spec: ApiSchema["ExperimentSpec"]): WorkflowDraft {
  return {
    ...defaultDraft, ...spec,
    experiment_id: spec.experiment_id ?? null,
    fixed_params: spec.fixed_params ?? {}, search_space: spec.search_space ?? {},
    dataset: { ...defaultDraft.dataset, ...spec.dataset },
    split: { ...defaultDraft.split, ...spec.split, random_state: spec.split?.random_state ?? 42 },
    preprocessing: { ...defaultDraft.preprocessing, ...spec.preprocessing },
    pso: { ...defaultDraft.pso, ...spec.pso },
    evaluation: { ...defaultDraft.evaluation, ...spec.evaluation },
    runtime: { ...defaultDraft.runtime, ...spec.runtime },
  };
}

type Stage = "data" | "model" | "search" | "full";
type WorkflowContextValue = {
  draft: WorkflowDraft;
  metadata: Metadata | null;
  exampleDatasets: ExampleDataset[];
  datasets: SavedDataset[];
  experiments: ExperimentSummary[];
  preview: DatasetPreview | null;
  validations: Partial<Record<Stage, ValidationResult>>;
  busy: string | null;
  message: string;
  run: RunSnapshot | null;
  events: RunEvent[];
  analysis: RunAnalysis | null;
  connection: "idle" | "sse" | "polling" | "closed";
  updateDraft: (next: Partial<WorkflowDraft> | ((current: WorkflowDraft) => WorkflowDraft)) => void;
  setTask: (task: Task, metric?: MetricId) => void;
  setEstimator: (estimator: string) => void;
  inspectDataset: () => Promise<void>;
  saveCsvDataset: (name: string) => Promise<void>;
  validateStage: (stage: Stage) => Promise<boolean>;
  createExperiment: (name: string, description: string) => Promise<string | null>;
  startRun: () => Promise<RunSnapshot | null>;
  loadRun: (runId: string) => Promise<void>;
  refreshCollections: () => Promise<void>;
  cancelRun: () => Promise<void>;
  retryRun: () => Promise<RunSnapshot | null>;
};

const WorkflowContext = createContext<WorkflowContextValue | null>(null);

function readDraft(): WorkflowDraft {
  try {
    const stored = localStorage.getItem(STORAGE_KEY);
    if (!stored) return defaultDraft;
    const parsed = JSON.parse(stored) as Partial<WorkflowDraft>;
    return {
      ...defaultDraft,
      ...parsed,
      dataset: { ...defaultDraft.dataset, ...parsed.dataset },
      split: { ...defaultDraft.split, ...parsed.split },
      preprocessing: { ...defaultDraft.preprocessing, ...parsed.preprocessing },
      pso: { ...defaultDraft.pso, ...parsed.pso },
      evaluation: { ...defaultDraft.evaluation, ...parsed.evaluation },
      runtime: { ...defaultDraft.runtime, ...parsed.runtime },
      schema_version: 1
    };
  } catch {
    return defaultDraft;
  }
}

export function WorkflowProvider({ children }: { children: ReactNode }) {
  const [draft, setDraft] = useState<WorkflowDraft>(readDraft);
  const [metadata, setMetadata] = useState<Metadata | null>(null);
  const [exampleDatasets, setExampleDatasets] = useState<ExampleDataset[]>([]);
  const [datasets, setDatasets] = useState<SavedDataset[]>([]);
  const [experiments, setExperiments] = useState<ExperimentSummary[]>([]);
  const [preview, setPreview] = useState<DatasetPreview | null>(null);
  const [validations, setValidations] = useState<Partial<Record<Stage, ValidationResult>>>({});
  const [busy, setBusy] = useState<string | null>(null);
  const [message, setMessage] = useState("");
  const [run, setRun] = useState<RunSnapshot | null>(null);
  const [events, setEvents] = useState<RunEvent[]>([]);
  const [analysis, setAnalysis] = useState<RunAnalysis | null>(null);
  const [connection, setConnection] = useState<"idle" | "sse" | "polling" | "closed">("idle");
  const dataSignature = JSON.stringify({ dataset: draft.dataset, task: draft.task });
  const evaluationSignature = JSON.stringify({ split: draft.split, evaluation: draft.evaluation });
  const previousEvaluationSignature = useRef(evaluationSignature);
  const previousDataSignature = useRef(dataSignature);
  const eventCursor = useRef(0);
  const selectedRun = useRef<string | null>(null);
  const loadGeneration = useRef(0);
  const terminal = !!run && ["completed", "failed", "cancelled", "interrupted"].includes(run.status);

  useEffect(() => {
    Promise.all([api.metadata(), api.exampleDatasets(), api.datasets(), api.experiments()])
      .then(([meta, examples, saved, experimentRows]) => {
        setMetadata(meta);
        setExampleDatasets(examples);
        setDatasets(saved);
        setExperiments(experimentRows);
        setDraft((current) => {
          if (Object.keys(current.search_space).length > 0) {
            return { ...current, fixed_params: constantsOnly(current.fixed_params, current.search_space) };
          }
          const defaults = meta.estimators[current.estimator]?.defaults[current.task];
          return defaults ? { ...current, fixed_params: constantsOnly(defaults.fixed_params, defaults.search_space), search_space: defaults.search_space } : current;
        });
      })
      .catch((error) => setMessage(error instanceof Error ? error.message : "The dashboard API is unavailable."));
  }, []);

  useEffect(() => {
    try { localStorage.setItem(STORAGE_KEY, JSON.stringify(draft)); }
    catch { setMessage("The browser could not save this draft. Save large CSV datasets to the workspace."); }
  }, [draft]);

  useEffect(() => {
    if (previousDataSignature.current !== dataSignature) setPreview(null);
    previousDataSignature.current = dataSignature;
  }, [dataSignature]);

  useEffect(() => {
    if (previousEvaluationSignature.current !== evaluationSignature) {
      setPreview((current) => current ? { ...current, split_summary: null } : null);
    }
    previousEvaluationSignature.current = evaluationSignature;
  }, [evaluationSignature]);

  useEffect(() => {
    if (draft.strategy === "pso" && draft.runtime.max_trials !== null) {
      setDraft((current) => ({
        ...current,
        runtime: { ...current.runtime, max_trials: null }
      }));
    }
  }, [draft.strategy, draft.runtime.max_trials]);

  useEffect(() => {
    if (!run?.run_id || terminal) return;
    setConnection("sse");
    const source = new EventSource(`/api/v1/runs/${run.run_id}/events?after=${eventCursor.current}`);
    const receive = (incoming: MessageEvent) => {
      if (selectedRun.current !== run.run_id) return;
      const event = JSON.parse(incoming.data) as RunEvent;
      eventCursor.current = Math.max(eventCursor.current, event.sequence_number ?? 0);
      setEvents((current) => mergeEvents(current, [event]));
      if (["run_completed", "run_failed", "run_cancelled", "run_interrupted"].includes(event.type)) void loadRun(run.run_id);
    };
    ["validation_passed", "run_queued", "worker_started", "dataset_prepared", "run_started", "trial_started", "fold_started", "fold_completed", "model_fit_started", "model_fit_completed", "model_fit_failed", "training_epoch", "trial_completed", "trial_failed", "iteration_completed", "best_updated", "final_refit_started", "final_refit_completed", "run_warning", "run_log", "run_cancel_requested", "run_completed", "run_failed", "run_cancelled", "run_interrupted"].forEach((name) => source.addEventListener(name, receive as EventListener));
    source.onopen = () => setConnection("sse");
    source.onerror = () => setConnection("polling");
    return () => source.close();
  }, [run?.run_id, terminal]);

  useEffect(() => {
    if (!run?.run_id || terminal) return;
    let alive = true;
    let pending = false;
    const timer = window.setInterval(async () => {
      if (pending) return;
      pending = true;
      try {
        const [snapshot, history] = await Promise.all([api.snapshot(run.run_id), api.history(run.run_id, eventCursor.current)]);
        if (!alive || selectedRun.current !== run.run_id) return;
        setRun((current) => reconcileRunSnapshot(current, snapshot));
        eventCursor.current = Math.max(eventCursor.current, ...history.events.map((event) => event.sequence_number ?? 0));
        setEvents((current) => mergeEvents(current, history.events));
        if (["completed", "failed", "cancelled", "interrupted"].includes(snapshot.status)) await loadRun(run.run_id);
      } catch { /* EventSource keeps trying to reconnect with Last-Event-ID. */ }
      finally { pending = false; }
    }, connection === "polling" ? 1200 : 5000);
    return () => { alive = false; window.clearInterval(timer); };
  }, [run?.run_id, terminal, connection]);

  function updateDraft(next: Partial<WorkflowDraft> | ((current: WorkflowDraft) => WorkflowDraft)) {
    setDraft((current) => typeof next === "function" ? next(current) : { ...current, ...next });
    setValidations({});
    setMessage("");
  }

  function setTask(task: Task, metric?: MetricId) {
    if (!metadata) return;
    const estimator = metadata.estimators[draft.estimator]?.tasks.includes(task)
      ? draft.estimator
      : Object.keys(metadata.estimators).find((name) => metadata.estimators[name].tasks.includes(task)) ?? draft.estimator;
    const defaults = metadata.estimators[estimator]?.defaults[task];
    updateDraft((current) => ({
      ...current, task, estimator, metric: metric && metadata.tasks[task].metrics.includes(metric) ? metric : metadata.tasks[task].metrics[0],
      fixed_params: constantsOnly(defaults?.fixed_params ?? {}, defaults?.search_space ?? {}), search_space: defaults?.search_space ?? {},
      split: { ...current.split, stratify: task !== "regression" },
      evaluation: { ...current.evaluation, positive_label: null }
    }));
  }

  function setEstimator(estimator: string) {
    if (!metadata) return;
    const task = metadata.estimators[estimator].tasks.includes(draft.task) ? draft.task : metadata.estimators[estimator].tasks[0];
    const defaults = metadata.estimators[estimator].defaults[task];
    updateDraft((current) => ({ ...current, estimator, task, metric: metadata.tasks[task].metrics.includes(current.metric) ? current.metric : metadata.tasks[task].metrics[0], fixed_params: constantsOnly(defaults.fixed_params, defaults.search_space), search_space: defaults.search_space }));
  }

  async function inspectDataset() {
    setBusy("dataset"); setMessage("");
    try { setPreview(await api.inspectDataset(draft)); }
    catch (error) { setMessage(error instanceof Error ? error.message : "Dataset inspection failed."); }
    finally { setBusy(null); }
  }

  async function saveCsvDataset(name: string) {
    if (!draft.dataset.csv_text) return;
    setBusy("dataset");
    try {
      const saved = await api.saveDataset(name, draft.dataset.csv_text);
      setDatasets(await api.datasets());
      updateDraft((current) => ({ ...current, dataset: { ...current.dataset, source: "stored", dataset_id: saved.dataset_id } }));
    } catch (error) { setMessage(error instanceof Error ? error.message : "Dataset could not be saved."); }
    finally { setBusy(null); }
  }

  async function validateStage(stage: Stage) {
    setBusy(stage); setMessage("");
    try {
      const result = await api.validateWorkflow(stage, draft);
      setValidations((current) => ({ ...current, [stage]: result }));
      return result.valid;
    } catch (error) {
      setMessage(error instanceof Error ? error.message : "Validation failed.");
      return false;
    } finally { setBusy(null); }
  }

  async function refreshCollections() {
    const [saved, experimentRows] = await Promise.all([api.datasets(), api.experiments()]);
    setDatasets(saved); setExperiments(experimentRows);
  }

  async function createExperiment(name: string, description: string) {
    try {
      const created = await api.createExperiment(name, description);
      await refreshCollections();
      updateDraft((current) => ({ ...current, experiment_id: created.experiment_id }));
      return created.experiment_id;
    } catch (error) { setMessage(error instanceof Error ? error.message : "Experiment could not be created."); return null; }
  }

  async function startRun() {
    if (!(await validateStage("full"))) return null;
    setBusy("start"); setEvents([]); setAnalysis(null);
    try {
      const created = await api.createRun(draft);
      selectedRun.current = created.run_id;
      eventCursor.current = 0;
      setRun(created); setConnection("idle");
      await refreshCollections();
      return created;
    } catch (error) { setMessage(error instanceof Error ? error.message : "Run could not start."); return null; }
    finally { setBusy(null); }
  }

  async function loadRun(runId: string) {
    const generation = ++loadGeneration.current;
    const changed = selectedRun.current !== runId;
    selectedRun.current = runId;
    if (changed) { eventCursor.current = 0; setEvents([]); setAnalysis(null); }
    try {
      const [snapshot, history] = await Promise.all([api.snapshot(runId), api.history(runId, eventCursor.current)]);
      if (generation !== loadGeneration.current || selectedRun.current !== runId) return;
      eventCursor.current = Math.max(eventCursor.current, ...history.events.map((event) => event.sequence_number ?? 0));
      setRun((current) => reconcileRunSnapshot(current, snapshot));
      setEvents((current) => mergeEvents(current, history.events));
      if (snapshot.status === "completed") {
        try {
          const loaded = await api.analysis(runId);
          if (generation === loadGeneration.current && selectedRun.current === runId) setAnalysis(loaded);
        } catch { if (selectedRun.current === runId) setAnalysis(null); }
      }
      if (["completed", "failed", "cancelled", "interrupted"].includes(snapshot.status)) setConnection("closed");
    } catch (error) { setMessage(error instanceof Error ? error.message : "Run could not be loaded."); }
  }

  async function cancelRun() {
    if (!run || run.cancel_requested) return;
    const runId = run.run_id;
    setBusy("cancel");
    setMessage("");
    try {
      const response = await api.cancel(runId);
      if (response.cancel_requested) {
        setRun((current) => current?.run_id === runId
          ? { ...current, cancel_requested: true }
          : current);
      }
      await loadRun(runId);
      if (!response.cancel_requested) {
        setMessage("The run had already stopped or was no longer owned by this dashboard process.");
      }
    }
    catch (error) { setMessage(error instanceof Error ? error.message : "Cancellation failed."); }
    finally { setBusy(null); }
  }
  async function retryRun() {
    if (!run) return null;
    try { const retry = await api.retry(run.run_id); await loadRun(retry.run_id); return retry; }
    catch (error) { setMessage(error instanceof Error ? error.message : "Retry failed."); return null; }
  }

  const value = useMemo(() => ({ draft, metadata, exampleDatasets, datasets, experiments, preview, validations, busy, message, run, events, analysis, connection, updateDraft, setTask, setEstimator, inspectDataset, saveCsvDataset, validateStage, createExperiment, startRun, loadRun, refreshCollections, cancelRun, retryRun }), [draft, metadata, exampleDatasets, datasets, experiments, preview, validations, busy, message, run, events, analysis, connection]);
  return <WorkflowContext.Provider value={value}>{children}</WorkflowContext.Provider>;
}

export function useWorkflow() {
  const value = useContext(WorkflowContext);
  if (!value) throw new Error("useWorkflow must be used inside WorkflowProvider");
  return value;
}

function constantsOnly(fixed: Record<string, JsonValue>, search: Record<string, unknown>) {
  return Object.fromEntries(Object.entries(fixed).filter(([name]) => !(name in search)));
}

export function mergeEvents(current: RunEvent[], incoming: RunEvent[]): RunEvent[] {
  if (incoming.length === 0) return current;
  const events = new Map<number | string, RunEvent>();
  for (const event of current) {
    events.set(event.sequence_number ?? `${event.type}-${event.timestamp}`, event);
  }
  let changed = false;
  for (const event of incoming) {
    const key = event.sequence_number ?? `${event.type}-${event.timestamp}`;
    if (events.has(key)) continue;
    events.set(key, event);
    changed = true;
  }
  if (!changed) return current;
  return [...events.values()].sort((a, b) => (a.sequence_number ?? 0) - (b.sequence_number ?? 0));
}

export function reconcileRunSnapshot(
  current: RunSnapshot | null,
  incoming: RunSnapshot,
): RunSnapshot {
  if (!current || current.run_id !== incoming.run_id) return incoming;
  return JSON.stringify(current) === JSON.stringify(incoming) ? current : incoming;
}
