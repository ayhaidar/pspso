import { BarChart3, Check, Download, Plus, RefreshCcw, Table2, X } from "lucide-react";

import { useEffect, useMemo, useState } from "react";

import { useNavigate, useParams } from "react-router-dom";

import { api } from "../api";

import { ConfusionChart, ConvergenceChart, ImportanceChart, PrecisionRecallChart, RegressionChart, RocChart } from "../components/Charts";

import { Metric, PageHeading } from "../components/Layout";

import type { AnalysisSplit, PredictionPreview, RunAnalysis, RunEvent } from "../types";

import { dashboardSpec, draftFromSpec, useWorkflow } from "../workflow";



type ToolId = "dataset" | "summary" | "convergence" | "roc" | "pr" | "confusion" | "threshold" | "per-class" | "predictions-chart" | "residuals" | "importance" | "predictions" | "failures";

const labels: Record<ToolId, string> = { dataset: "Dataset summary", summary: "Metric summary", convergence: "Search convergence", roc: "ROC curve", pr: "Precision-recall curve", confusion: "Confusion matrix", threshold: "Threshold explorer", "per-class": "Per-class metrics", "predictions-chart": "Actual vs predicted", residuals: "Residual analysis", importance: "Feature importance", predictions: "Prediction table", failures: "Failure analysis" };



export function ResultsPage() {

  const { runId } = useParams();
  const navigate = useNavigate();

  const { run, events, analysis, loadRun, updateDraft } = useWorkflow();

  const [tools, setTools] = useState<ToolId[]>([]);
  const [layoutReady, setLayoutReady] = useState(false);

  const [error, setError] = useState("");

  const [picker, setPicker] = useState(true);

  const [splitName, setSplitName] = useState<"validation" | "test">("validation");

  const [predictions, setPredictions] = useState<PredictionPreview | null>(null);

  const [importance, setImportance] = useState<RunAnalysis["feature_importance"] | null>(null);

  useEffect(() => { if (runId && run?.run_id !== runId) void loadRun(runId); }, [runId]);

  useEffect(() => {

    if (!analysis || !run) return;

    const defaults = defaultTools(analysis.task);
    let active = true;
    setLayoutReady(false);

    api.experiment(run.experiment_id).then((experiment) => {
      if (!active) return;

      const saved = (experiment.result_layout?.length ? experiment.result_layout : defaults) as ToolId[];

      setTools(analysis.dataset && !saved.includes("dataset") ? ["dataset", ...saved] : saved);

    }).catch(() => { if (active) setTools(defaults); })
      .finally(() => { if (active) setLayoutReady(true); });

    setSplitName(analysis.test ? "test" : "validation");

    setImportance(analysis.feature_importance);
    return () => { active = false; };

  }, [analysis?.task, run?.run_id]);

  const split = analysis?.[splitName] as AnalysisSplit | undefined;
  const spec = dashboardSpec(run);
  const modelAvailable = Boolean(run?.artifacts?.model);
  const isCrossValidation = spec?.evaluation?.protocol === "cross_validation";
  const refitEnabled = Boolean(spec?.evaluation?.refit_best);
  const cvWithoutRefit = isCrossValidation && !refitEnabled;
  const selectedEvent = [...events].reverse().find((event) => event.type === "best_updated");
  const lastCandidate = [...events].reverse().find((event) => event.type === "trial_completed");
  const exportKinds = (["spec", "result", "metrics", "analysis", "events", "predictions", "manifest", "model", "environment", "split_indices", "log"] as const)
    .filter((kind) => modelAvailable || !["model", "predictions"].includes(kind));

  const supported = useMemo(() => {

    const choices = availableTools(analysis, split, modelAvailable);

    if (modelAvailable && split?.cross_validation) choices.set("predictions", "Select the untouched test split for saved-model predictions.");

    return choices;

  }, [analysis, split, modelAvailable]);

  function prepareSavedModelRun() {
    if (!spec) return;
    updateDraft(draftFromSpec(spec));
    navigate("/search");
  }

  async function toggleTool(id: ToolId) {

    const next = tools.includes(id) ? tools.filter((tool) => tool !== id) : [...tools, id];

    setTools(next);

    try { if (run) await api.saveResultLayout(run.experiment_id, next); }

    catch (error) { setError(error instanceof Error ? error.message : "Result layout could not be saved."); }

  }

  async function loadPredictions() {

    setError("");

    try { if (run) setPredictions(await api.predictions(run.run_id, splitName)); }

    catch (error) { setError(error instanceof Error ? error.message : "Predictions could not be loaded."); }

  }

  async function permutation() {

    setError("");

    try { if (run) setImportance(await api.featureImportance(run.run_id)); }

    catch (error) { setError(error instanceof Error ? error.message : "Feature importance could not be calculated."); }

  }

  if (!run || !analysis || !split) return <><PageHeading eyebrow="Stage 5 of 6" title="Results" description="Open a completed run to inspect its held-out metrics and diagnostics."/><div className="emptyState large">Results become available after a successful run.</div></>;

  return <>

    <PageHeading eyebrow="Stage 5 of 6" title="Results" description="Start with the dataset and metric summaries, then switch analyses on or off below. Every chart identifies the evaluation split it uses." actions={<><select value={splitName} onChange={(event) => { setSplitName(event.target.value as "validation" | "test"); setPredictions(null); setError(""); }}><option value="validation">Validation / CV summary</option>{analysis.test && <option value="test">Untouched test split</option>}</select><details className="exportMenu"><summary><Download size={16}/> Export</summary><div>{exportKinds.map((kind) => <a key={kind} href={api.exportUrl(run.run_id, kind, kind === "predictions" ? (split.cross_validation ? "test" : splitName) : undefined)} download>{kind}</a>)}</div></details><button className="secondary" onClick={() => setPicker(!picker)}>{picker ? <X size={16}/> : <Plus size={16}/>} {picker ? "Hide tools" : `Choose result tools (${tools.length})`}</button></>}/>

    {error && <div role="alert" className="validationSummary invalid">{error}</div>}

    {events.filter((event) => event.type === "run_warning" && event.payload.reason === "artifact_serialization").map((event) => <div role="alert" className="validationSummary invalid" key={event.sequence_number}>{String(event.payload.message)}</div>)}

    <section className="resultHeader"><div><span>Experiment</span><strong>{run.experiment_name}</strong></div><div><span>Dataset</span><strong>{String(run.request?.dataset?.name ?? run.request?.dataset?.dataset_id ?? "Imported data")}</strong></div><div><span>Model</span><strong>{run.estimator}</strong></div><div><span>Search</span><strong>{run.strategy}</strong></div><div><span>Evaluation</span><strong>{splitName}</strong></div></section>

    <section className={`selectionNotice ${modelAvailable ? "" : "attention"}`}>
      <div><span>Selected winner</span><strong>{selectedEvent ? `Trial ${String(selectedEvent.payload.trial_id ?? "—")} · cost ${format(selectedEvent.payload.best_cost)}` : `Best cost ${format(run.best?.best_cost)}`}</strong><p>{lastCandidate && selectedEvent && lastCandidate.payload.trial_id !== selectedEvent.payload.trial_id ? `Trial ${String(lastCandidate.payload.trial_id)} finished later at cost ${format(lastCandidate.payload.cost)}. It did not replace the better candidate.` : "The search keeps the best candidate even when later trials score worse."}</p></div>
      <div><span>Saved predictor · refit choice</span><strong>{modelAvailable ? cvWithoutRefit ? `${spec?.evaluation?.folds ?? "Fold"}-model cross-validation ensemble · no extra fit` : refitEnabled ? "One newly refitted model · 1 additional fit" : "Evaluated holdout winner · no extra fit" : "No model artifact from this run"}</strong><p>{modelAvailable ? cvWithoutRefit ? "Predictions average the winning candidate's fitted fold models. This is a different predictor from one model refitted on all development rows." : refitEnabled ? `After selection, PSPSO trained this model again using ${isCrossValidation ? "all development rows" : "the combined training and validation rows"}. Its test result can differ from the candidate score because it is a new fit.${run.estimator === "pytorch_mlp" || run.estimator === "sklearn_mlp" ? " For this neural network, the saved seed makes refitting repeatable, while the changed training rows can still produce different weights." : ""}` : "This is the model fitted on the training partition during holdout evaluation; it has not seen the validation rows." : cvWithoutRefit ? "This run used the earlier evaluation-only behavior, which discarded fold models when final refitting was off." : "This run did not produce a usable model artifact. Check the recorded warnings or failures for details."}</p></div>
      {!modelAvailable && cvWithoutRefit && <button className="secondary" onClick={prepareSavedModelRun}><RefreshCcw size={15}/> Prepare the same run</button>}
    </section>

    {picker && <section className="toolPicker"><header><div><strong>Results tools</strong><span>Choose what appears in this report. Selected tools are shown below and saved with the experiment.</span></div><span className="selectedToolCount">{tools.length} shown</span></header><div>{(Object.keys(labels) as ToolId[]).map((id) => { const reason = supported.get(id); return <button key={id} disabled={!layoutReady || reason !== true} className={tools.includes(id) ? "selected" : ""} onClick={() => void toggleTool(id)}><span>{tools.includes(id) ? <Check size={14}/> : <Plus size={14}/>} {labels[id]}</span><small>{reason !== true ? reason : tools.includes(id) ? "Shown in report" : "Add to report"}</small></button>; })}</div></section>}

    {split.cross_validation && <section className="surface"><h2>Cross-validation evidence</h2><p>Mean {format(split.cross_validation.mean)} · standard deviation {format(split.cross_validation.standard_deviation)} · each preprocessing pipeline was fitted within its fold.</p><div className="tableWrap"><table><thead><tr><th>Fold</th><th>Metric</th><th>Cost</th></tr></thead><tbody>{split.cross_validation.folds.map((fold) => <tr key={fold.fold}><td>{fold.fold}</td><td>{format(fold.metric)}</td><td>{format(fold.cost)}</td></tr>)}</tbody></table></div></section>}

    <div className="resultsGrid">{tools.filter((tool) => supported.get(tool) === true).map((tool) => <ResultTool key={tool} id={tool} split={split} analysis={analysis} events={events} importance={importance} predictions={predictions} loadPredictions={loadPredictions} permutation={permutation}/>)}</div>

  </>;

}



function ResultTool({ id, split, analysis, events, importance, predictions, loadPredictions, permutation }: { id: ToolId; split: AnalysisSplit; analysis: RunAnalysis; events: RunEvent[]; importance: RunAnalysis["feature_importance"] | null; predictions: PredictionPreview | null; loadPredictions: () => Promise<void>; permutation: () => Promise<void> }) {

  if (id === "dataset") return <Tool title="Dataset and split summary" wide><DatasetResultSummary dataset={analysis.dataset}/></Tool>;

  if (id === "summary") return <Tool title="Metric summary" wide><div className="metricStrip wide">{Object.entries(split.metrics).map(([name, value]) => <Metric key={name} label={name.replace(/_/g, " ")} value={format(value)}/>)}</div></Tool>;

  if (id === "convergence") return <Tool title="Search convergence"><p className="fieldHint">{convergenceExplanation(events)}</p><ConvergenceChart events={events}/></Tool>;

  if (id === "roc") return <Tool title="ROC curve"><RocChart split={split}/></Tool>;

  if (id === "pr") return <Tool title="Precision-recall curve"><PrecisionRecallChart split={split}/></Tool>;

  if (id === "confusion") return <Tool title="Confusion matrix"><ConfusionChart split={split}/></Tool>;

  if (id === "threshold") return <Tool title="Decision-threshold explorer" wide><ThresholdExplorer split={split}/></Tool>;

  if (id === "per-class") return <Tool title="Per-class performance" wide><div className="tableWrap"><table><thead><tr>{Object.keys(split.per_class?.[0] ?? {}).map((name) => <th key={name}>{name}</th>)}</tr></thead><tbody>{split.per_class?.map((row, index) => <tr key={index}>{Object.values(row).map((value, column) => <td key={column}>{typeof value === "number" ? format(value) : value}</td>)}</tr>)}</tbody></table></div></Tool>;

  if (id === "predictions-chart") return <Tool title="Actual versus predicted"><RegressionChart split={split} mode="predictions"/></Tool>;

  if (id === "residuals") return <Tool title="Residual analysis"><RegressionChart split={split} mode="residuals"/></Tool>;

  if (id === "importance") return <Tool title="Feature importance"><div className="toolActions"><span>{importance?.available ? `${importance.source} importance` : importance?.reason ?? "Native importance unavailable"}</span>{!importance?.available && <button className="secondary" onClick={() => void permutation()}><RefreshCcw size={15}/> Calculate permutation importance</button>}</div>{importance?.available ? <ImportanceChart items={importance.items}/> : <div className="emptyCompact">Run the permutation calculation to measure validation-score sensitivity to each feature.</div>}</Tool>;

  if (id === "predictions") return <Tool title="Prediction table" wide><div className="toolActions"><span>Actual values and saved-model output</span><button className="secondary" onClick={() => void loadPredictions()}><Table2 size={15}/> Load rows</button></div>{predictions ? <PredictionTable preview={predictions}/> : <div className="emptyCompact">Load up to 50 rows from the selected evaluation split.</div>}</Tool>;

  if (id === "failures") { const failures = events.filter((event) => event.type.includes("failed")); return <Tool title="Failure analysis" wide>{failures.length ? failures.map((event) => <div className="failureRow" key={event.sequence_number}><strong>{event.type.replace(/_/g, " ")}</strong><span>{String(event.payload.error ?? event.payload.reason ?? "No additional message")}</span></div>) : <div className="emptyCompact">No failed attempts were recorded.</div>}</Tool>; }

  return null;

}



function Tool({ title, wide, children }: { title: string; wide?: boolean; children: React.ReactNode }) { return <section className={`surface resultTool ${wide ? "wide" : ""}`}><h2>{title}</h2>{children}</section>; }

function convergenceExplanation(events: RunEvent[]) {
  const started = [...events].reverse().find((event) => event.type === "run_started")?.payload;
  const candidates = Number(started?.planned_trials ?? 0);
  const candidateFits = Number(started?.planned_candidate_fits ?? candidates);
  const folds = candidates > 0 ? Math.max(1, Math.round(candidateFits / candidates)) : 1;
  const completed = events.filter((event) => event.type === "trial_completed").length;
  return `${completed}${candidates ? ` of ${candidates}` : ""} candidate scores are plotted in completion order. ${folds === 1 ? "Each score comes from one validation fit." : `Each score averages ${folds} fold fits; those ${candidateFits} fits are not separate convergence points.`}`;
}

function ThresholdExplorer({ split }: { split: AnalysisSplit }) { const rows = split.threshold_diagnostics ?? []; const [index, setIndex] = useState(Math.floor(rows.length / 2)); const row = rows[index]; if (!row) return <div className="emptyCompact">Probability or decision scores are unavailable.</div>; return <><input className="thresholdSlider" type="range" min="0" max={rows.length - 1} value={index} onChange={(event) => setIndex(Number(event.target.value))}/><div className="metricStrip wide"><Metric label="Threshold" value={format(row.threshold)}/><Metric label="Sensitivity" value={format(row.sensitivity)}/><Metric label="Specificity" value={format(row.specificity)}/><Metric label="Precision" value={format(row.precision)}/><Metric label="F1" value={format(row.f1)}/><Metric label="Accuracy" value={format(row.accuracy)}/></div></>; }

function PredictionTable({ preview }: { preview: PredictionPreview }) { return <div className="tableWrap"><table><thead><tr><th>Row</th><th>Actual</th><th>Predicted</th><th>{preview.task === "regression" ? "Residual" : "Score / probability"}</th></tr></thead><tbody>{preview.preview_rows.map((row) => <tr key={row.row_index}><td>{row.row_index}</td><td>{String(row.actual)}</td><td>{preview.task === "regression" ? format(row.predicted) : String(row.predicted)}</td><td>{row.residual != null ? format(row.residual) : row.score != null ? format(row.score) : row.probabilities ? Object.entries(row.probabilities).map(([name, value]) => `${name}: ${format(value)}`).join(" · ") : "—"}</td></tr>)}</tbody></table></div>; }

function DatasetResultSummary({ dataset }: { dataset?: RunAnalysis["dataset"] }) { if (!dataset) return <div className="emptyCompact">Dataset statistics were not recorded for this older run.</div>; const target = dataset.target_summary; return <><div className="metricStrip wide"><Metric label="Rows" value={dataset.row_count}/><Metric label="Columns" value={dataset.summary.column_count}/><Metric label="Missing cells" value={dataset.summary.total_missing}/><Metric label="Duplicate rows" value={dataset.summary.duplicate_rows}/></div><div className="datasetResultGrid"><div><h3>Target</h3>{target?.distribution ? <div className="distributionList compact">{target.distribution.map((row) => <div key={String(row.label)}><div><strong>{String(row.label)}</strong><span>{row.count} · {row.percentage}%</span></div><div className="distributionTrack"><span style={{ width: `${row.percentage}%` }}/></div></div>)}</div> : <div className="summaryList">{Object.entries(target?.statistics ?? {}).filter(([name]) => ["count", "mean", "std", "min", "median", "max"].includes(name)).map(([name, value]) => <div key={name}><span>{name}</span><strong>{format(value)}</strong></div>)}</div>}</div><div><h3>Recorded partitions</h3><div className="summaryList">{Object.entries(dataset.split_summary?.partitions ?? {}).map(([name, partition]) => <div key={name}><span>{name}</span><strong>{partition.rows} rows · {partition.percentage.toFixed(2)}%</strong></div>)}</div></div></div></>; }

function defaultTools(task: string): ToolId[] { if (task === "regression") return ["dataset", "summary", "predictions-chart", "residuals", "convergence", "importance", "predictions"]; if (task === "multiclass_classification") return ["dataset", "summary", "confusion", "per-class", "roc", "importance", "predictions"]; return ["dataset", "summary", "roc", "pr", "confusion", "threshold", "importance", "predictions"]; }

function availableTools(analysis: RunAnalysis | null, split: AnalysisSplit | undefined, modelAvailable: boolean) { const map = new Map<ToolId, true | string>(); (Object.keys(labels) as ToolId[]).forEach((id) => map.set(id, true)); if (!analysis || !split) return map; if (analysis.task === "regression") ["roc", "pr", "confusion", "threshold", "per-class"].forEach((id) => map.set(id as ToolId, "Classification only")); else ["predictions-chart", "residuals"].forEach((id) => map.set(id as ToolId, "Regression only")); if (!split.roc_curve && !split.multiclass_roc) map.set("roc", "Probability scores unavailable"); if (!split.precision_recall_curve) map.set("pr", "Binary scores unavailable"); if (!split.threshold_diagnostics) map.set("threshold", "Binary scores unavailable"); if (!split.per_class) map.set("per-class", "Multiclass only"); if (!modelAvailable) { map.set("importance", "No model was saved for this older run"); map.set("predictions", "No model was saved for this older run"); } return map; }

function format(value: unknown) { if (value == null) return "—"; const number = Number(value); return Number.isFinite(number) ? number.toFixed(4) : "—"; }
