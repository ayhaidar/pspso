import { FormEvent, ReactNode, useEffect, useMemo, useRef, useState } from "react";
import {
  Activity,
  AlertTriangle,
  CheckCircle2,
  ChevronRight,
  Clock3,
  Database,
  Download,
  FolderKanban,
  Play,
  RefreshCcw,
  Save,
  Settings,
  SlidersHorizontal,
  Table2
} from "lucide-react";
import { createRoot } from "react-dom/client";
import "./styles.css";

type SearchParam =
  | { type: "choice"; values: string[] }
  | { type: "int"; low: number; high: number }
  | { type: "float"; low: number; high: number; precision: number };

type EstimatorDefault = {
  fixed_params: Record<string, unknown>;
  search_space: Record<string, SearchParam>;
  allowed_params: string[];
};

type EstimatorMeta = {
  label: string;
  tasks: string[];
  optional_dependency: string | null;
  description: string;
  dependency: { required: string | null; installed: boolean; install: string | null };
  defaults: Record<string, EstimatorDefault>;
};

type Metadata = {
  estimators: Record<string, EstimatorMeta>;
  tasks: Record<string, { label: string; description: string; metrics: string[] }>;
  metrics: Record<string, { label: string; description: string }>;
};

type RunEvent = {
  sequence_number?: number;
  type: string;
  timestamp: string;
  payload: Record<string, any>;
};

type RunSnapshot = {
  run_id: string;
  experiment_id: string;
  experiment_name: string;
  experiment_description?: string | null;
  status: string;
  event_count: number;
  best?: Record<string, any> | null;
  result?: Record<string, any> | null;
  error?: string | null;
  created_at?: string | null;
  updated_at?: string | null;
  duration?: number | null;
  n_trials?: number;
  n_failures?: number;
  strategy?: string | null;
  task?: string | null;
  metric?: string | null;
  estimator?: string | null;
  request?: Record<string, any> | null;
};

type ValidationResult = {
  valid: boolean;
  errors: Record<string, string[]>;
};

type ExperimentSummary = {
  experiment_id: string;
  name: string;
  description: string;
  tags: string[];
  is_ad_hoc: boolean;
  created_at: string;
  run_count: number;
  latest_status: string | null;
  latest_run_id: string | null;
  latest_updated_at: string | null;
};

type ExperimentDetail = ExperimentSummary & { runs: RunSnapshot[] };

type PhaseState = "done" | "active" | "idle" | "failed";

type DatasetPreview = {
  columns: { name: string; dtype: string; missing: number; unique: number }[];
  row_count: number;
  preview: Record<string, any>[];
  target_candidates: string[];
};

type PredictionPreview = {
  run_id: string;
  split: "validation" | "train";
  task: string;
  metric: string;
  estimator: string;
  feature_columns: string[];
  target_mapping: Record<string, any> | null;
  row_count: number;
  preview_rows: Array<{
    row_index: string | number;
    actual: any;
    predicted: any;
    residual?: number;
    score?: number | null;
    features: Record<string, any>;
  }>;
};

const api = {
  metadata: () => fetch("/api/estimators").then((response) => response.json()),
  experiments: () => fetch("/api/experiments").then((response) => response.json()),
  experiment: (experimentId: string) =>
    fetch(`/api/experiments/${experimentId}`).then((response) => response.json()),
  createExperiment: (payload: Record<string, any>) =>
    fetch("/api/experiments", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload)
    }).then((response) => response.json()),
  validate: (payload: Record<string, any>) =>
    fetch("/api/runs/validate", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload)
    }).then((response) => response.json()),
  createRun: (payload: Record<string, any>) =>
    fetch("/api/runs", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload)
    }),
  inspectDataset: (payload: Record<string, any>) =>
    fetch("/api/datasets/inspect", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload)
    }).then((response) => response.json()),
  snapshot: (runId: string) => fetch(`/api/runs/${runId}`).then((response) => response.json()),
  history: (runId: string) => fetch(`/api/runs/${runId}/history`).then((response) => response.json()),
  artifacts: (runId: string) => fetch(`/api/runs/${runId}/artifacts`).then((response) => response.json()),
  predictions: (runId: string, split: "validation" | "train") =>
    fetch(`/api/runs/${runId}/predictions?split=${split}&limit=20`).then((response) => response.json())
};

function App() {
  const [metadata, setMetadata] = useState<Metadata | null>(null);
  const [experiments, setExperiments] = useState<ExperimentSummary[]>([]);
  const [selectedExperimentId, setSelectedExperimentId] = useState("");
  const [experimentDetail, setExperimentDetail] = useState<ExperimentDetail | null>(null);
  const [newExperimentName, setNewExperimentName] = useState("");
  const [newExperimentDescription, setNewExperimentDescription] = useState("");

  const [datasetSource, setDatasetSource] = useState("example");
  const [datasetName, setDatasetName] = useState("breast_cancer");
  const [csvText, setCsvText] = useState("");
  const [targetColumn, setTargetColumn] = useState("target");
  const [task, setTask] = useState("binary classification");
  const [metric, setMetric] = useState("roc_auc");
  const [estimator, setEstimator] = useState("svm");
  const [strategy, setStrategy] = useState("random");
  const [validationSize, setValidationSize] = useState(0.2);
  const [randomState, setRandomState] = useState(42);
  const [fixedParams, setFixedParams] = useState("{}");
  const [searchSpace, setSearchSpace] = useState<Record<string, SearchParam>>({});
  const [advancedMode, setAdvancedMode] = useState(false);
  const [advancedJson, setAdvancedJson] = useState("{}");
  const [particles, setParticles] = useState(5);
  const [iterations, setIterations] = useState(4);
  const [maxTrials, setMaxTrials] = useState(6);

  const [run, setRun] = useState<RunSnapshot | null>(null);
  const [events, setEvents] = useState<RunEvent[]>([]);
  const [selectedTrialId, setSelectedTrialId] = useState<number | null>(null);
  const [message, setMessage] = useState("");
  const [validation, setValidation] = useState<ValidationResult | null>(null);
  const [connection, setConnection] = useState<"idle" | "sse" | "polling" | "closed">("idle");
  const [datasetPreview, setDatasetPreview] = useState<DatasetPreview | null>(null);
  const [predictionPreview, setPredictionPreview] = useState<PredictionPreview | null>(null);
  const [predictionSplit, setPredictionSplit] = useState<"validation" | "train">("validation");

  const estimatorMeta = metadata?.estimators[estimator];
  const taskMeta = metadata?.tasks[task];
  const metricMeta = metadata?.metrics[metric];

  const trials = events
    .filter((event) => event.type === "trial_completed" || event.type === "trial_failed")
    .map((event) => event.payload);
  const latestFailure = [...events]
    .reverse()
    .find((event) => event.type === "trial_failed" || event.type === "run_failed");
  const best = useMemo(() => {
    const bestEvent = [...events].reverse().find((event) => event.type === "best_updated");
    return bestEvent?.payload ?? run?.best ?? null;
  }, [events, run]);
  const latestEvent = events.length > 0 ? events[events.length - 1] : null;
  const currentActivity = describeEvent(latestEvent);
  const phaseStates = buildPhaseStates(events, run?.status ?? "idle");
  const selectedExperimentRuns = experimentDetail?.runs ?? [];

  useEffect(() => {
    Promise.all([api.metadata(), api.experiments()]).then(([payload, experimentRows]) => {
      setMetadata(payload);
      setExperiments(experimentRows);
      const firstEstimator = payload.estimators.svm ? "svm" : Object.keys(payload.estimators)[0];
      const firstTask = payload.estimators[firstEstimator].tasks[0];
      const firstMetric = payload.tasks[firstTask].metrics[0];
      setEstimator(firstEstimator);
      setTask(firstTask);
      setMetric(firstMetric);
      applyDefaults(payload, firstEstimator, firstTask);
      if (experimentRows.length > 0) {
        setSelectedExperimentId(experimentRows[0].experiment_id);
      }
    });
  }, []);

  useEffect(() => {
    if (!selectedExperimentId) {
      setExperimentDetail(null);
      return;
    }
    api.experiment(selectedExperimentId).then((payload: ExperimentDetail) => {
      setExperimentDetail(payload);
    });
  }, [selectedExperimentId]);

  useEffect(() => {
    if (!run?.run_id || ["completed", "failed"].includes(run.status)) return;
    setConnection("sse");
    const source = new EventSource(`/api/runs/${run.run_id}/events`);
    const pushEvent = (messageEvent: MessageEvent) => {
      const event = JSON.parse(messageEvent.data) as RunEvent;
      setEvents((current) => {
        if (current.some((item) => item.sequence_number === event.sequence_number)) {
          return current;
        }
        return [...current, event];
      });
    };
    [
      "validation_passed",
      "dataset_prepared",
      "run_started",
      "trial_started",
      "trial_completed",
      "trial_failed",
      "iteration_completed",
      "best_updated",
      "run_warning"
    ].forEach((eventName) => source.addEventListener(eventName, pushEvent as EventListener));
    source.addEventListener("run_completed", (messageEvent) => {
      pushEvent(messageEvent as MessageEvent);
      refreshRun(run.run_id);
      refreshExperiments();
      source.close();
      setConnection("closed");
    });
    source.addEventListener("run_failed", (messageEvent) => {
      pushEvent(messageEvent as MessageEvent);
      refreshRun(run.run_id);
      refreshExperiments();
      source.close();
      setConnection("closed");
    });
    source.onerror = () => {
      setConnection("polling");
      source.close();
    };
    return () => source.close();
  }, [run?.run_id, run?.status]);

  useEffect(() => {
    if (!run?.run_id || connection !== "polling") return;
    const timer = window.setInterval(() => refreshRun(run.run_id), 1000);
    return () => window.clearInterval(timer);
  }, [run?.run_id, connection]);

  async function refreshExperiments() {
    const rows = await api.experiments();
    setExperiments(rows);
    if (selectedExperimentId) {
      const detail = await api.experiment(selectedExperimentId);
      setExperimentDetail(detail);
    }
  }

  function resetValidation() {
    setValidation(null);
    setMessage("");
  }

  function applyDefaults(source: Metadata, estimatorName: string, taskName: string) {
    const defaults = source.estimators[estimatorName]?.defaults[taskName];
    if (!defaults) return;
    setFixedParams(JSON.stringify(defaults.fixed_params, null, 2));
    setSearchSpace(defaults.search_space);
    setAdvancedJson(JSON.stringify(defaults.search_space, null, 2));
    resetValidation();
  }

  function handleTaskChange(taskName: string) {
    if (!metadata) return;
    const metricName = metadata.tasks[taskName].metrics[0];
    const estimatorName = metadata.estimators[estimator]?.tasks.includes(taskName)
      ? estimator
      : Object.entries(metadata.estimators).find(([, value]) => value.tasks.includes(taskName))?.[0] ?? estimator;
    setTask(taskName);
    setMetric(metricName);
    setEstimator(estimatorName);
    applyDefaults(metadata, estimatorName, taskName);
  }

  function handleEstimatorChange(estimatorName: string) {
    if (!metadata) return;
    const supportedTask = metadata.estimators[estimatorName].tasks.includes(task)
      ? task
      : metadata.estimators[estimatorName].tasks[0];
    setEstimator(estimatorName);
    if (supportedTask !== task) {
      setTask(supportedTask);
      setMetric(metadata.tasks[supportedTask].metrics[0]);
    }
    applyDefaults(metadata, estimatorName, supportedTask);
  }

  async function refreshRun(runId: string) {
    const [snapshot, history] = await Promise.all([api.snapshot(runId), api.history(runId)]);
    setRun(snapshot);
    setEvents(history.events);
    if (snapshot.experiment_id && snapshot.experiment_id !== selectedExperimentId) {
      setSelectedExperimentId(snapshot.experiment_id);
    } else if (snapshot.experiment_id) {
      const detail = await api.experiment(snapshot.experiment_id);
      setExperimentDetail(detail);
    }
    if (snapshot.status === "completed" || snapshot.status === "failed") {
      setConnection("closed");
    }
  }

  async function loadRun(runId: string) {
    setMessage("");
    setConnection("closed");
    setSelectedTrialId(null);
    setPredictionPreview(null);
    await refreshRun(runId);
  }

  async function validateCurrentConfig() {
    setMessage("");
    let payload: Record<string, any>;
    try {
      payload = buildPayload();
    } catch (error) {
      setValidation({
        valid: false,
        errors: { search_space: [error instanceof Error ? error.message : "Invalid JSON."] }
      });
      return false;
    }
    const result = await api.validate(payload);
    setValidation(result);
    return result.valid;
  }

  async function createExperiment() {
    if (!newExperimentName.trim()) {
      setMessage("Experiment name is required.");
      return;
    }
    const created = await api.createExperiment({
      name: newExperimentName.trim(),
      description: newExperimentDescription.trim(),
      tags: []
    });
    setNewExperimentName("");
    setNewExperimentDescription("");
    const rows = await api.experiments();
    setExperiments(rows);
    setSelectedExperimentId(created.experiment_id);
    const detail = await api.experiment(created.experiment_id);
    setExperimentDetail(detail);
  }

  async function startRun(event: FormEvent) {
    event.preventDefault();
    setMessage("");
    setEvents([]);
    setSelectedTrialId(null);
    const isValid = await validateCurrentConfig();
    if (!isValid) return;
    const response = await api.createRun(buildPayload());
    const body = await response.json();
    if (!response.ok) {
      if (body.detail && typeof body.detail === "object" && "errors" in body.detail) {
        setValidation(body.detail);
      } else {
        setMessage(typeof body.detail === "string" ? body.detail : "Run could not start.");
      }
      return;
    }
    setRun(body);
    setSelectedExperimentId(body.experiment_id);
    setConnection("idle");
    const history = await api.history(body.run_id);
    setEvents(history.events);
    await refreshExperiments();
  }

  function buildPayload() {
    return {
      experiment_id: selectedExperimentId || null,
      dataset: {
        source: datasetSource,
        name: datasetSource === "example" ? datasetName : null,
        csv_text: datasetSource === "csv" ? csvText : null,
        target_column: targetColumn
      },
      task,
      metric,
      estimator,
      fixed_params: JSON.parse(fixedParams || "{}"),
      search_space: advancedMode ? JSON.parse(advancedJson || "{}") : searchSpace,
      strategy,
      split: {
        validation_size: validationSize,
        random_state: randomState,
        stratify: task === "binary classification"
      },
      preprocessing: {
        scale_numeric: true,
        encode_categorical: true,
        ignored_columns: []
      },
      pso: {
        particles,
        iterations,
        c1: 1.49618,
        c2: 1.49618,
        w: 0.7298,
        topology: "global"
      },
      runtime: {
        max_trials: strategy === "grid" ? null : maxTrials,
        verbose: 0
      }
    };
  }

  function updateParam(name: string, next: SearchParam) {
    const updated = { ...searchSpace, [name]: next };
    setSearchSpace(updated);
    setAdvancedJson(JSON.stringify(updated, null, 2));
    resetValidation();
  }

  async function exportArtifacts() {
    if (!run?.run_id) return;
    const artifacts = await api.artifacts(run.run_id);
    const blob = new Blob([JSON.stringify(artifacts, null, 2)], { type: "application/json" });
    const url = URL.createObjectURL(blob);
    const link = document.createElement("a");
    link.href = url;
    link.download = `pspso-${run.run_id}-artifacts.json`;
    link.click();
    URL.revokeObjectURL(url);
  }

  async function loadDatasetPreview() {
    setMessage("");
    try {
      const preview = await api.inspectDataset({
        source: datasetSource,
        name: datasetSource === "example" ? datasetName : null,
        csv_text: datasetSource === "csv" ? csvText : null,
        target_column: targetColumn
      });
      if (preview.detail) {
        setMessage(typeof preview.detail === "string" ? preview.detail : "Dataset preview failed.");
        return;
      }
      setDatasetPreview(preview);
    } catch {
      setMessage("Dataset preview failed.");
    }
  }

  async function loadPredictionPreview() {
    if (!run?.run_id) return;
    setMessage("");
    try {
      const preview = await api.predictions(run.run_id, predictionSplit);
      if (preview.detail) {
        setMessage(typeof preview.detail === "string" ? preview.detail : "Prediction preview failed.");
        return;
      }
      setPredictionPreview(preview);
    } catch {
      setMessage("Prediction preview failed.");
    }
  }

  const metricOptions = metadata?.tasks[task]?.metrics ?? [];
  const estimatorOptions = metadata
    ? Object.entries(metadata.estimators).filter(([, value]) => value.tasks.includes(task))
    : [];

  return (
    <main className="app">
      <header className="topbar">
        <div>
          <h1>pspso dashboard</h1>
          <p>Build a run on the left, watch the system explain itself in the middle, and revisit saved runs on the right.</p>
        </div>
        <button className="secondary" onClick={exportArtifacts} disabled={!run?.run_id}>
          <Download size={16} />
          Export run
        </button>
      </header>

      <section className="introBand">
        <IntroCard title="1. Build the run" text="Choose the dataset, task, estimator, and search space. Validation tells you what is incompatible before training begins." />
        <IntroCard title="2. Watch execution" text="The monitor shows dataset preparation, trial attempts, best updates, and terminal state as ordered timeline events." />
        <IntroCard title="3. Compare history" text="Every run is saved under an experiment so you can reopen it, inspect failures, and compare outcomes after a refresh." />
      </section>

      <form className="workspace" onSubmit={startRun}>
        <section className="panel config">
          <div className="panelHeader">
            <SectionTitle icon={<FolderKanban size={18} />} title="Run builder" />
            <p className="panelLead">This side is for deciding what to train, how to search, and where the run should be saved.</p>
          </div>

          <BuilderSummary
            experimentName={experimentDetail?.name ?? "Ad hoc"}
            datasetName={datasetSource === "example" ? datasetName : "CSV upload"}
            estimatorName={estimatorMeta?.label ?? estimator}
            taskName={taskMeta?.label ?? task}
            strategy={strategy}
          />

          <div className="sectionCard">
            <SectionTitle icon={<FolderKanban size={18} />} title="Experiment" />
            <p className="sectionLead">Experiments group related runs together so you can reopen and compare them later.</p>
            <div className="grid two">
              <label>
                Save runs under
                <select value={selectedExperimentId} onChange={(event) => setSelectedExperimentId(event.target.value)}>
                  <option value="">Create ad hoc automatically</option>
                  {experiments.map((item) => (
                    <option key={item.experiment_id} value={item.experiment_id}>
                      {item.name}
                    </option>
                  ))}
                </select>
              </label>
              <div className="infoStat">
                <span>Runs saved</span>
                <strong>{experimentDetail?.run_count ?? 0}</strong>
              </div>
            </div>
            <InfoBox
              title={experimentDetail?.name ?? "Ad hoc experiment"}
              text={
                experimentDetail
                  ? experimentDetail.description || "Saved experiment for comparing multiple runs."
                  : "Leave experiment unselected to create a one-off run record automatically."
              }
            />
            <div className="subPanel">
              <h3>
                <Save size={16} />
                New experiment
              </h3>
              <label>
                Name
                <input value={newExperimentName} onChange={(event) => setNewExperimentName(event.target.value)} />
              </label>
              <label>
                Description
                <textarea
                  rows={3}
                  value={newExperimentDescription}
                  onChange={(event) => setNewExperimentDescription(event.target.value)}
                />
              </label>
              <button className="secondary" type="button" onClick={() => void createExperiment()}>
                <Save size={16} />
                Save experiment
              </button>
            </div>
          </div>

          <div className="sectionCard">
            <SectionTitle icon={<Database size={18} />} title="Dataset and task" />
            <p className="sectionLead">Choose the data source, the target column, and the kind of prediction problem the optimizer should solve.</p>
            <div className="grid two">
              <label>
                Source
                <select value={datasetSource} onChange={(event) => { setDatasetSource(event.target.value); resetValidation(); }}>
                  <option value="example">Example dataset</option>
                  <option value="csv">CSV text</option>
                </select>
              </label>
              <label>
                Example
                <select value={datasetName} onChange={(event) => { setDatasetName(event.target.value); resetValidation(); }}>
                  <option value="breast_cancer">Breast cancer</option>
                  <option value="diabetes">Diabetes</option>
                </select>
              </label>
            </div>
            {datasetSource === "csv" && (
              <label>
                CSV text
                <textarea value={csvText} onChange={(event) => { setCsvText(event.target.value); resetValidation(); }} rows={5} />
              </label>
            )}
            <div className="grid three">
              <label>
                Target
                <input value={targetColumn} onChange={(event) => { setTargetColumn(event.target.value); resetValidation(); }} />
              </label>
              <label>
                Task
                <select value={task} onChange={(event) => handleTaskChange(event.target.value)}>
                  {metadata && Object.entries(metadata.tasks).map(([name, item]) => (
                    <option key={name} value={name}>{item.label}</option>
                  ))}
                </select>
              </label>
              <label>
                Metric
                <select value={metric} onChange={(event) => { setMetric(event.target.value); resetValidation(); }}>
                  {metricOptions.map((name) => (
                    <option key={name} value={name}>{metadata?.metrics[name]?.label ?? name}</option>
                  ))}
                </select>
              </label>
            </div>
            <InfoBox title={taskMeta?.label ?? "Task"} text={taskMeta?.description ?? "Loading task metadata."} />
            <InfoBox title={metricMeta?.label ?? "Metric"} text={metricMeta?.description ?? "Choose a metric supported by the current task."} />
            <div className="buttonRow">
              <button className="secondary" type="button" onClick={() => void loadDatasetPreview()}>
                <Database size={16} />
                Show dataset preview
              </button>
            </div>
            <DatasetPreviewPanel preview={datasetPreview} />
          </div>

          <div className="sectionCard">
            <SectionTitle icon={<Settings size={18} />} title="Estimator and search plan" />
            <p className="sectionLead">Pick the model family and the search strategy, then set the runtime limits that control how much exploration happens.</p>
            <div className="grid three">
              <label>
                Estimator
                <select value={estimator} onChange={(event) => handleEstimatorChange(event.target.value)}>
                  {estimatorOptions.map(([name, item]) => (
                    <option key={name} value={name}>{item.label}</option>
                  ))}
                </select>
              </label>
              <label>
                Strategy
                <select value={strategy} onChange={(event) => { setStrategy(event.target.value); resetValidation(); }}>
                  <option value="random">Random</option>
                  <option value="grid">Grid</option>
                  <option value="pso">PSO</option>
                </select>
              </label>
              <label>
                Validation split
                <input type="number" min="0.05" max="0.8" step="0.05" value={validationSize} onChange={(event) => { setValidationSize(Number(event.target.value)); resetValidation(); }} />
              </label>
            </div>
            <InfoBox
              title={estimatorMeta?.label ?? "Estimator"}
              text={estimatorMeta ? `${estimatorMeta.description} Dependency: ${estimatorMeta.dependency.installed ? "available" : estimatorMeta.dependency.install}.` : "Loading estimator metadata."}
            />
            <div className="grid three">
              <label>
                Random seed
                <input type="number" value={randomState} onChange={(event) => { setRandomState(Number(event.target.value)); resetValidation(); }} />
              </label>
              <label>
                Max trials
                <input type="number" min="1" value={maxTrials} onChange={(event) => { setMaxTrials(Number(event.target.value)); resetValidation(); }} />
              </label>
              <label>
                Particles
                <input type="number" min="1" value={particles} onChange={(event) => { setParticles(Number(event.target.value)); resetValidation(); }} />
              </label>
            </div>
            <label>
              PSO iterations
              <input type="number" min="1" value={iterations} onChange={(event) => { setIterations(Number(event.target.value)); resetValidation(); }} />
            </label>
            <label>
              Fixed params JSON
              <textarea value={fixedParams} onChange={(event) => { setFixedParams(event.target.value); resetValidation(); }} rows={4} />
            </label>
          </div>

          <div className="sectionCard">
            <SectionTitle icon={<SlidersHorizontal size={18} />} title="Tunable parameters" />
            <p className="sectionLead">These are the hyperparameters the optimizer is allowed to change while searching for a better result.</p>
            <label className="inlineCheck">
              <input type="checkbox" checked={advancedMode} onChange={(event) => setAdvancedMode(event.target.checked)} />
              Advanced JSON editor
            </label>
            {advancedMode ? (
              <textarea value={advancedJson} onChange={(event) => { setAdvancedJson(event.target.value); resetValidation(); }} rows={8} />
            ) : (
              <div className="paramList">
                {Object.entries(searchSpace).map(([name, spec]) => (
                  <ParamEditor key={name} name={name} spec={spec} onChange={(next) => updateParam(name, next)} />
                ))}
              </div>
            )}
          </div>

          <div className="actionPanel">
            <div className="buttonRow">
              <button className="secondary" type="button" onClick={validateCurrentConfig}>
                <CheckCircle2 size={16} />
                Validate
              </button>
              <button className="primary" type="submit" disabled={validation?.valid !== true}>
                <Play size={16} />
                Start run
              </button>
            </div>
            <ValidationPanel validation={validation} />
            {message && <p className="error">{message}</p>}
          </div>
        </section>

        <div className="monitorColumn">
          <section className="panel monitor">
            <div className="panelHeader">
              <SectionTitle icon={<Activity size={18} />} title="Execution monitor" />
              <p className="panelLead">This area explains what the backend is doing now, what it has completed, and where the best result came from.</p>
            </div>
            <RunHeader run={run} />
            <PhaseStrip phases={phaseStates} />
            <div className="stats">
              <Stat label="Status" value={run?.status ?? "idle"} />
              <Stat label="Connection" value={connection} />
              <Stat label="Trials" value={String(run?.n_trials ?? trials.length)} />
              <Stat label="Best metric" value={best?.best_metric?.toFixed?.(4) ?? "-"} />
            </div>
            <div className="activityCard">
              <strong>Current activity</strong>
              <span>{currentActivity}</span>
            </div>
            <StrategyVisualizer
              strategy={String(run?.strategy ?? strategy)}
              events={events}
              particles={Number((run?.request as any)?.pso?.particles ?? particles)}
              maxTrials={Number((run?.request as any)?.runtime?.max_trials ?? maxTrials ?? 0)}
            />
            {latestFailure && (
              <div className="failure">
                <AlertTriangle size={17} />
                <span>{latestFailure.payload.error ?? latestFailure.payload.message ?? "Training failed."}</span>
              </div>
            )}
            {connection === "polling" && (
              <div className="failure muted">
                <RefreshCcw size={17} />
                <span>SSE disconnected. Polling run snapshots instead.</span>
              </div>
            )}
            <Timeline events={events} selectedTrialId={selectedTrialId} onSelectTrial={setSelectedTrialId} />
          </section>

          <section className="panel analytics">
            <div className="panelHeader">
              <SectionTitle icon={<Table2 size={18} />} title="Trial analysis" />
              <p className="panelLead">Use this section to connect the timeline to the actual trial metrics, parameter sets, and best result snapshot.</p>
            </div>
            <ScoreChart trials={trials} />
            <div className="best">
              <h3>
                <SlidersHorizontal size={16} />
                Best parameters
              </h3>
              <pre>{best?.best_params ? JSON.stringify(best.best_params, null, 2) : "{}"}</pre>
            </div>
            <div className="tableWrap">
              <table>
                <thead>
                  <tr>
                    <th>#</th>
                    <th>Status</th>
                    <th>Train</th>
                    <th>Validation</th>
                    <th>Duration</th>
                    <th>Params / error</th>
                  </tr>
                </thead>
                <tbody>
                  {trials.map((trial) => (
                    <tr
                      key={trial.trial_id}
                      className={selectedTrialId === trial.trial_id ? "activeRow" : ""}
                      onClick={() => setSelectedTrialId(trial.trial_id)}
                    >
                      <td>{trial.trial_id}</td>
                      <td>{trial.status}</td>
                      <td>{formatNumber(trial.train_metric)}</td>
                      <td>{formatNumber(trial.metric)}</td>
                      <td>{formatNumber(trial.duration)}s</td>
                      <td>
                        <code>{trial.error ? trial.error : JSON.stringify(trial.params)}</code>
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
            <div className="sectionCard">
              <SectionTitle icon={<Activity size={18} />} title="Prediction output" />
              <p className="sectionLead">For a completed run, inspect the model output on the saved train or validation split and compare actual versus predicted values.</p>
              <div className="grid two">
                <label>
                  Split
                  <select value={predictionSplit} onChange={(event) => setPredictionSplit(event.target.value as "validation" | "train")}>
                    <option value="validation">Validation</option>
                    <option value="train">Train</option>
                  </select>
                </label>
                <div className="buttonBlock">
                  <button className="secondary" type="button" onClick={() => void loadPredictionPreview()} disabled={!run?.result?.best_params}>
                    <Activity size={16} />
                    Show predictions
                  </button>
                </div>
              </div>
              <PredictionPreviewPanel preview={predictionPreview} />
            </div>
          </section>
        </div>

        <aside className="panel historyPanel">
          <div className="panelHeader">
            <SectionTitle icon={<Clock3 size={18} />} title="Saved runs" />
            <p className="panelLead">This column is for reopening previous runs from the selected experiment and comparing how different attempts behaved.</p>
          </div>
          <InfoBox
            title={experimentDetail?.name ?? "No experiment selected"}
            text={experimentDetail ? `${experimentDetail.run_count} stored run${experimentDetail.run_count === 1 ? "" : "s"} in this experiment.` : "Select an experiment or let the app create an ad hoc one for the next run."}
          />
          <div className="historyList">
            {selectedExperimentRuns.length === 0 && <div className="emptyState">No saved runs yet for this experiment.</div>}
            {selectedExperimentRuns.map((item) => (
              <button
                key={item.run_id}
                type="button"
                className={`historyItem ${run?.run_id === item.run_id ? "selected" : ""}`}
                onClick={() => void loadRun(item.run_id)}
              >
                <div>
                  <strong>{item.estimator} / {item.strategy}</strong>
                  <span>{item.status} | {item.metric}</span>
                </div>
                <ChevronRight size={16} />
              </button>
            ))}
          </div>
        </aside>
      </form>
    </main>
  );
}

function IntroCard({ title, text }: { title: string; text: string }) {
  return (
    <div className="introCard">
      <strong>{title}</strong>
      <span>{text}</span>
    </div>
  );
}

function SectionTitle({ icon, title }: { icon: ReactNode; title: string }) {
  return <h2>{icon}{title}</h2>;
}

function BuilderSummary({
  experimentName,
  datasetName,
  estimatorName,
  taskName,
  strategy
}: {
  experimentName: string;
  datasetName: string;
  estimatorName: string;
  taskName: string;
  strategy: string;
}) {
  return (
    <div className="summaryStrip">
      <SummaryPill label="Experiment" value={experimentName} />
      <SummaryPill label="Dataset" value={datasetName} />
      <SummaryPill label="Task" value={taskName} />
      <SummaryPill label="Estimator" value={estimatorName} />
      <SummaryPill label="Strategy" value={strategy.toUpperCase()} />
    </div>
  );
}

function DatasetPreviewPanel({ preview }: { preview: DatasetPreview | null }) {
  if (!preview) {
    return <div className="emptyState">Load a dataset preview to inspect columns, missing values, and sample rows.</div>;
  }
  return (
    <div className="previewPanel">
      <div className="previewStats">
        <SummaryPill label="Rows" value={String(preview.row_count)} />
        <SummaryPill label="Columns" value={String(preview.columns.length)} />
      </div>
      <div className="tableWrap compact">
        <table>
          <thead>
            <tr>
              <th>Column</th>
              <th>Type</th>
              <th>Missing</th>
              <th>Unique</th>
            </tr>
          </thead>
          <tbody>
            {preview.columns.map((column) => (
              <tr key={column.name}>
                <td>{column.name}</td>
                <td>{column.dtype}</td>
                <td>{column.missing}</td>
                <td>{column.unique}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      <div className="tableWrap compact">
        <table>
          <thead>
            <tr>
              {Object.keys(preview.preview[0] ?? {}).map((column) => (
                <th key={column}>{column}</th>
              ))}
            </tr>
          </thead>
          <tbody>
            {preview.preview.map((row, index) => (
              <tr key={index}>
                {Object.entries(row).map(([column, value]) => (
                  <td key={column}>{String(value)}</td>
                ))}
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}

function SummaryPill({ label, value }: { label: string; value: string }) {
  return (
    <div className="summaryPill">
      <span>{label}</span>
      <strong>{value}</strong>
    </div>
  );
}

function InfoBox({ title, text }: { title: string; text: string }) {
  return (
    <div className="infoBox">
      <strong>{title}</strong>
      <span>{text}</span>
    </div>
  );
}

function ValidationPanel({ validation }: { validation: ValidationResult | null }) {
  if (!validation) return null;
  if (validation.valid) {
    return <div className="valid"><CheckCircle2 size={17} /> Configuration is valid.</div>;
  }
  return (
    <div className="validationErrors">
      <strong>Configuration needs attention</strong>
      {Object.entries(validation.errors)
        .filter(([, messages]) => messages.length > 0)
        .map(([section, messages]) => (
          <div key={section}>
            <span>{section}</span>
            <ul>{messages.map((message) => <li key={message}>{message}</li>)}</ul>
          </div>
        ))}
    </div>
  );
}

function ParamEditor({ name, spec, onChange }: { name: string; spec: SearchParam; onChange: (next: SearchParam) => void }) {
  if (spec.type === "choice") {
    return (
      <div className="paramRow">
        <strong>{name}</strong>
        <span>Choice</span>
        <input
          value={spec.values.join(", ")}
          onChange={(event) => onChange({ type: "choice", values: event.target.value.split(",").map((value) => value.trim()).filter(Boolean) })}
        />
      </div>
    );
  }
  return (
    <div className="paramRow">
      <strong>{name}</strong>
      <span>{spec.type === "int" ? "Integer" : "Float"}</span>
      <input type="number" value={spec.low} onChange={(event) => onChange({ ...spec, low: Number(event.target.value) })} />
      <input type="number" value={spec.high} onChange={(event) => onChange({ ...spec, high: Number(event.target.value) })} />
      {spec.type === "float" && (
        <input type="number" min="0" value={spec.precision} onChange={(event) => onChange({ ...spec, precision: Number(event.target.value) })} />
      )}
    </div>
  );
}

function Stat({ label, value }: { label: string; value: string }) {
  return (
    <div className="stat">
      <span>{label}</span>
      <strong>{value}</strong>
    </div>
  );
}

function RunHeader({ run }: { run: RunSnapshot | null }) {
  if (!run) {
    return <div className="emptyState">Start or open a saved run to see its execution story.</div>;
  }
  return (
    <div className="runHeader">
      <div>
        <strong>{run.experiment_name}</strong>
        <span>{run.run_id}</span>
      </div>
      <div>
        <strong>{run.estimator} / {run.strategy}</strong>
        <span>{run.task} | {run.metric}</span>
      </div>
      <div>
        <strong>{run.status}</strong>
        <span>{formatDuration(run.duration)}</span>
      </div>
    </div>
  );
}

function PhaseStrip({ phases }: { phases: { label: string; state: PhaseState }[] }) {
  return (
    <div className="phaseStrip">
      {phases.map((phase) => (
        <div key={phase.label} className={`phaseChip ${phase.state}`}>
          <span>{phase.label}</span>
        </div>
      ))}
    </div>
  );
}

function StrategyVisualizer({
  strategy,
  events,
  particles,
  maxTrials
}: {
  strategy: string;
  events: RunEvent[];
  particles: number;
  maxTrials: number;
}) {
  const canvasRef = useRef<HTMLCanvasElement | null>(null);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const context = canvas.getContext("2d");
    if (!context) return;
    let frame = 0;
    let animation = 0;

    const resize = () => {
      const width = canvas.clientWidth || 760;
      const height = 260;
      const ratio = window.devicePixelRatio || 1;
      canvas.width = Math.floor(width * ratio);
      canvas.height = Math.floor(height * ratio);
      context.setTransform(ratio, 0, 0, ratio, 0, 0);
    };

    const drawNode = (x: number, y: number, state: string, label: string, pulse: number) => {
      const colors: Record<string, string> = {
        idle: "#d5dde0",
        active: "#e1ad2f",
        completed: "#1f7a64",
        failed: "#b44131"
      };
      const radius = state === "active" ? 16 + pulse * 4 : 14;
      context.beginPath();
      context.fillStyle = colors[state] ?? colors.idle;
      context.arc(x, y, radius, 0, Math.PI * 2);
      context.fill();
      context.fillStyle = "#ffffff";
      context.font = "12px IBM Plex Sans, sans-serif";
      context.textAlign = "center";
      context.textBaseline = "middle";
      context.fillText(label, x, y);
    };

    const trialEvents = events.filter((event) =>
      event.type === "trial_started" || event.type === "trial_completed" || event.type === "trial_failed"
    );

    const draw = () => {
      frame += 1;
      resize();
      const width = canvas.clientWidth || 760;
      const height = 260;
      const pulse = (Math.sin(frame / 12) + 1) / 2;

      context.clearRect(0, 0, width, height);
      context.fillStyle = "#f8fbfa";
      context.fillRect(0, 0, width, height);

      context.fillStyle = "#20313a";
      context.font = "600 14px IBM Plex Sans, sans-serif";
      context.textAlign = "left";
      context.fillText(strategyLabel(strategy), 18, 22);

      context.fillStyle = "#5d6b6d";
      context.font = "12px IBM Plex Sans, sans-serif";
      context.fillText(strategyDescription(strategy, particles), 18, 42);

      if (strategy === "pso") {
        drawPsoScene(context, width, height, trialEvents, particles, pulse, drawNode);
      } else if (strategy === "grid") {
        drawGridScene(context, width, height, trialEvents, maxTrials || trialEvents.length || 1, pulse, drawNode);
      } else {
        drawRandomScene(context, width, height, trialEvents, maxTrials || trialEvents.length || 1, pulse, drawNode);
      }

      animation = window.requestAnimationFrame(draw);
    };

    animation = window.requestAnimationFrame(draw);
    return () => window.cancelAnimationFrame(animation);
  }, [events, maxTrials, particles, strategy]);

  return (
    <div className="visualizerPanel">
      <h3>
        <Activity size={16} />
        Strategy visualizer
      </h3>
      <canvas ref={canvasRef} className="strategyCanvas" />
      <div className="visualizerLegend">
        <LegendSwatch color="#e1ad2f" label="running" />
        <LegendSwatch color="#1f7a64" label="completed" />
        <LegendSwatch color="#b44131" label="failed" />
        <LegendSwatch color="#d5dde0" label="queued" />
      </div>
    </div>
  );
}

function LegendSwatch({ color, label }: { color: string; label: string }) {
  return (
    <span className="legendSwatch">
      <i style={{ background: color }} />
      {label}
    </span>
  );
}

function drawPsoScene(
  context: CanvasRenderingContext2D,
  width: number,
  height: number,
  trialEvents: RunEvent[],
  particles: number,
  pulse: number,
  drawNode: (x: number, y: number, state: string, label: string, pulse: number) => void
) {
  const slotCount = Math.max(1, particles || 5);
  const latestIteration = Math.max(
    0,
    ...trialEvents.map((event) => Number(event.payload.iteration ?? 0)).filter((value) => Number.isFinite(value))
  );
  const slots = Array.from({ length: slotCount }, (_, index) => ({
    state: "idle",
    label: `P${index + 1}`
  }));

  for (const event of trialEvents) {
    const iteration = Number(event.payload.iteration ?? 0);
    if (iteration !== latestIteration) continue;
    const slot = Number(event.payload.particle_index ?? event.payload.strategy_slot ?? 0);
    if (slot < 0 || slot >= slots.length) continue;
    if (event.type === "trial_started") slots[slot].state = "active";
    if (event.type === "trial_completed") slots[slot].state = "completed";
    if (event.type === "trial_failed") slots[slot].state = "failed";
  }

  context.fillStyle = "#5d6b6d";
  context.fillText(`Iteration ${latestIteration + 1}`, 18, 66);
  const spacing = width / (slotCount + 1);
  for (let index = 0; index < slotCount; index += 1) {
    const x = spacing * (index + 1);
    const y = height / 2 + (index % 2 === 0 ? -18 : 18);
    context.beginPath();
    context.strokeStyle = "#d6dfdc";
    context.lineWidth = 2;
    context.moveTo(x, height / 2 + (index % 2 === 0 ? -18 : 18));
    if (index < slotCount - 1) {
      const nextX = spacing * (index + 2);
      const nextY = height / 2 + ((index + 1) % 2 === 0 ? -18 : 18);
      context.lineTo(nextX, nextY);
      context.stroke();
    }
    drawNode(x, y, slots[index].state, slots[index].label, pulse);
  }
}

function drawRandomScene(
  context: CanvasRenderingContext2D,
  width: number,
  height: number,
  trialEvents: RunEvent[],
  count: number,
  pulse: number,
  drawNode: (x: number, y: number, state: string, label: string, pulse: number) => void
) {
  const slotCount = Math.max(1, Math.min(count, 18));
  const slots = buildSlotStates(trialEvents, slotCount);
  context.fillStyle = "#5d6b6d";
  context.fillText("Random search explores scattered parameter samples.", 18, 66);
  for (let index = 0; index < slotCount; index += 1) {
    const seed = index + 1;
    const x = 60 + pseudo(seed * 13) * (width - 120);
    const y = 92 + pseudo(seed * 29) * (height - 124);
    drawSpark(context, x, y, 10 + pulse * 6);
    drawNode(x, y, slots[index] ?? "idle", `R${index + 1}`, pulse);
  }
}

function drawGridScene(
  context: CanvasRenderingContext2D,
  width: number,
  height: number,
  trialEvents: RunEvent[],
  count: number,
  pulse: number,
  drawNode: (x: number, y: number, state: string, label: string, pulse: number) => void
) {
  const slotCount = Math.max(1, Math.min(count, 16));
  const slots = buildSlotStates(trialEvents, slotCount);
  const size = Math.ceil(Math.sqrt(slotCount));
  const cell = Math.min(70, (width - 80) / size);
  const startX = (width - size * cell) / 2;
  const startY = 82;
  context.fillStyle = "#5d6b6d";
  context.fillText("Grid search marches through parameter cells in order.", 18, 66);
  for (let index = 0; index < slotCount; index += 1) {
    const row = Math.floor(index / size);
    const column = index % size;
    const x = startX + column * cell;
    const y = startY + row * cell;
    context.fillStyle = "#ffffff";
    context.strokeStyle = "#d6dfdc";
    context.lineWidth = 1.5;
    context.fillRect(x, y, cell - 8, cell - 8);
    context.strokeRect(x, y, cell - 8, cell - 8);
    drawNode(x + (cell - 8) / 2, y + (cell - 8) / 2, slots[index] ?? "idle", `${index + 1}`, pulse);
  }
}

function buildSlotStates(trialEvents: RunEvent[], slotCount: number) {
  const states = Array.from({ length: slotCount }, () => "idle");
  for (const event of trialEvents) {
    const slot = Number(event.payload.strategy_slot ?? event.payload.trial_id ?? 1) - (event.payload.strategy_slot == null ? 1 : 0);
    if (slot < 0 || slot >= slotCount) continue;
    if (event.type === "trial_started") states[slot] = "active";
    if (event.type === "trial_completed") states[slot] = "completed";
    if (event.type === "trial_failed") states[slot] = "failed";
  }
  return states;
}

function strategyLabel(strategy: string) {
  if (strategy === "pso") return "PSO particle map";
  if (strategy === "grid") return "Grid cell map";
  return "Random sampler map";
}

function strategyDescription(strategy: string, particles: number) {
  if (strategy === "pso") return `Each circle represents a particle. With ${particles} particles, the canvas tracks the current PSO iteration.`;
  if (strategy === "grid") return "Each cell represents one parameter combination visited by grid search.";
  return "Each circle is one sampled trial position chosen by random search.";
}

function drawSpark(context: CanvasRenderingContext2D, x: number, y: number, radius: number) {
  context.save();
  context.strokeStyle = "#ead3a1";
  context.lineWidth = 1;
  for (let angle = 0; angle < 8; angle += 1) {
    const theta = (Math.PI / 4) * angle;
    context.beginPath();
    context.moveTo(x + Math.cos(theta) * (radius * 0.6), y + Math.sin(theta) * (radius * 0.6));
    context.lineTo(x + Math.cos(theta) * radius, y + Math.sin(theta) * radius);
    context.stroke();
  }
  context.restore();
}

function pseudo(seed: number) {
  const value = Math.sin(seed * 999) * 10000;
  return value - Math.floor(value);
}

function PredictionPreviewPanel({ preview }: { preview: PredictionPreview | null }) {
  if (!preview) {
    return <div className="emptyState">Load prediction output from a completed run to inspect actual and predicted values.</div>;
  }
  return (
    <div className="previewPanel">
      <div className="previewStats">
        <SummaryPill label="Split" value={preview.split} />
        <SummaryPill label="Rows" value={String(preview.row_count)} />
      </div>
      <div className="tableWrap compact">
        <table>
          <thead>
            <tr>
              <th>Row</th>
              <th>Actual</th>
              <th>Predicted</th>
              {preview.task === "regression" ? <th>Residual</th> : <th>Score</th>}
            </tr>
          </thead>
          <tbody>
            {preview.preview_rows.map((row) => (
              <tr key={String(row.row_index)}>
                <td>{String(row.row_index)}</td>
                <td>{String(row.actual)}</td>
                <td>{String(row.predicted)}</td>
                {preview.task === "regression" ? (
                  <td>{typeof row.residual === "number" ? row.residual.toFixed(4) : "-"}</td>
                ) : (
                  <td>{typeof row.score === "number" ? row.score.toFixed(4) : "-"}</td>
                )}
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}

function Timeline({
  events,
  selectedTrialId,
  onSelectTrial
}: {
  events: RunEvent[];
  selectedTrialId: number | null;
  onSelectTrial: (trialId: number | null) => void;
}) {
  return (
    <div className="timeline">
      <h3>
        <Clock3 size={16} />
        Timeline
      </h3>
      {events.length === 0 && <div className="emptyState">Waiting for execution events.</div>}
      {events.map((event) => {
        const trialId = typeof event.payload.trial_id === "number" ? event.payload.trial_id : null;
        const selected = trialId !== null && selectedTrialId === trialId;
        return (
          <button
            key={`${event.sequence_number ?? event.timestamp}-${event.type}`}
            type="button"
            className={`timelineItem ${eventTone(event.type)} ${selected ? "selected" : ""}`}
            onClick={() => onSelectTrial(trialId)}
          >
            <div className="timelineMeta">
              <strong>{describeEvent(event)}</strong>
              <span>{formatTimestamp(event.timestamp)}</span>
            </div>
            <code>{summarizePayload(event)}</code>
          </button>
        );
      })}
    </div>
  );
}

function ScoreChart({ trials }: { trials: Record<string, any>[] }) {
  const points = trials.filter((trial) => typeof trial.metric === "number").map((trial) => trial.metric as number);
  if (points.length < 2) return <div className="chart empty">Waiting for trial metrics</div>;
  const min = Math.min(...points);
  const max = Math.max(...points);
  const range = max - min || 1;
  const path = points
    .map((value, index) => {
      const x = (index / (points.length - 1)) * 100;
      const y = 100 - ((value - min) / range) * 90 - 5;
      return `${x},${y}`;
    })
    .join(" ");
  return (
    <svg className="chart" viewBox="0 0 100 100" preserveAspectRatio="none">
      <polyline points={path} fill="none" stroke="currentColor" strokeWidth="2" />
    </svg>
  );
}

function describeEvent(event: RunEvent | null) {
  if (!event) return "Waiting for a run.";
  switch (event.type) {
    case "validation_passed":
      return "Configuration accepted and queued.";
    case "dataset_prepared":
      return `Dataset prepared with ${event.payload.train_rows} training rows and ${event.payload.validation_rows} validation rows.`;
    case "run_started":
      return `Optimizer started with ${event.payload.strategy} search.`;
    case "trial_started":
      return `Trial ${event.payload.trial_id} started.`;
    case "trial_completed":
      return `Trial ${event.payload.trial_id} completed with metric ${formatNumber(event.payload.metric)}.`;
    case "trial_failed":
      return `Trial ${event.payload.trial_id} failed.`;
    case "iteration_completed":
      return `Iteration ${event.payload.iteration} completed.`;
    case "best_updated":
      return `Best result updated at trial ${event.payload.trial_id}.`;
    case "run_completed":
      return "Run completed successfully.";
    case "run_failed":
      return "Run failed.";
    default:
      return event.type;
  }
}

function summarizePayload(event: RunEvent) {
  if (event.type === "trial_started" || event.type === "trial_completed" || event.type === "trial_failed") {
    return JSON.stringify(event.payload.params ?? { error: event.payload.error });
  }
  if (event.type === "best_updated") {
    return JSON.stringify(event.payload.best_params ?? {});
  }
  return JSON.stringify(event.payload);
}

function eventTone(eventType: string) {
  if (eventType === "run_failed" || eventType === "trial_failed") return "danger";
  if (eventType === "run_completed" || eventType === "best_updated") return "success";
  if (eventType === "validation_passed" || eventType === "dataset_prepared") return "info";
  return "neutral";
}

function formatNumber(value: any) {
  return typeof value === "number" ? value.toFixed(4) : "-";
}

function formatTimestamp(value: string | null | undefined) {
  if (!value) return "-";
  return new Date(value).toLocaleTimeString();
}

function formatDuration(value: number | null | undefined) {
  if (typeof value !== "number") return "In progress";
  return `${value.toFixed(2)}s`;
}

function buildPhaseStates(events: RunEvent[], status: string) {
  const has = (type: string) => events.some((event) => event.type === type);
  if (status === "failed") {
    return [
      { label: "Validated", state: has("validation_passed") ? "done" as PhaseState : "idle" as PhaseState },
      { label: "Prepared", state: has("dataset_prepared") ? "done" as PhaseState : "idle" as PhaseState },
      { label: "Searching", state: has("run_started") ? "done" as PhaseState : "idle" as PhaseState },
      { label: "Best update", state: has("best_updated") ? "done" as PhaseState : "idle" as PhaseState },
      { label: "Failed", state: "failed" as PhaseState }
    ];
  }
  if (status === "completed") {
    return [
      { label: "Validated", state: "done" as PhaseState },
      { label: "Prepared", state: "done" as PhaseState },
      { label: "Searching", state: "done" as PhaseState },
      { label: "Best update", state: has("best_updated") ? "done" as PhaseState : "idle" as PhaseState },
      { label: "Completed", state: "done" as PhaseState }
    ];
  }
  return [
    { label: "Validated", state: has("validation_passed") ? "done" as PhaseState : "idle" as PhaseState },
    { label: "Prepared", state: has("dataset_prepared") ? "done" as PhaseState : "idle" as PhaseState },
    { label: "Searching", state: has("run_started") ? "active" as PhaseState : "idle" as PhaseState },
    { label: "Best update", state: has("best_updated") ? "active" as PhaseState : "idle" as PhaseState },
    { label: "Finished", state: "idle" as PhaseState }
  ];
}

createRoot(document.getElementById("root")!).render(<App />);
