import { BarChart3, Copy, Filter, History, Radio, RefreshCcw } from "lucide-react";
import { useEffect, useMemo, useState } from "react";
import { useNavigate } from "react-router-dom";
import { api } from "../api";
import { PageHeading, StatusBadge } from "../components/Layout";
import type { RunSnapshot } from "../types";
import { dashboardSpec, draftFromSpec, useWorkflow } from "../workflow";

export function HistoryPage() {
  const navigate = useNavigate();
  const { experiments, loadRun, updateDraft } = useWorkflow();
  const [runs, setRuns] = useState<RunSnapshot[]>([]);
  const [error, setError] = useState("");
  const [experiment, setExperiment] = useState("all");
  const [query, setQuery] = useState("");
  const [status, setStatus] = useState("all");
  const [task, setTask] = useState("all");
  async function refresh() { try { setRuns(await api.runs()); setError(""); } catch (error) { setError(error instanceof Error ? error.message : "History could not be loaded."); } }
  useEffect(() => { void refresh(); }, []);
  const filtered = useMemo(() => runs.filter((run) => {
    const text = `${run.experiment_name} ${run.estimator} ${run.metric} ${run.strategy}`.toLowerCase();
    return text.includes(query.toLowerCase()) && (status === "all" || run.status === status) && (task === "all" || run.task === task) && (experiment === "all" || run.experiment_id === experiment);
  }), [runs, query, status, task, experiment]);
  async function open(run: RunSnapshot, results = false) { await loadRun(run.run_id); navigate(results ? `/results/${run.run_id}` : `/live/${run.run_id}`); }
  function clone(run: RunSnapshot) { const spec = dashboardSpec(run); if (!spec) return; updateDraft(draftFromSpec({ ...spec, experiment_id: run.experiment_id })); navigate("/search"); }
  return <>
    <PageHeading eyebrow="Stage 6 of 6" title="Experiment history" description="GUI and CLI runs share this durable workspace. Filter, compare, reopen, or clone any recorded specification." actions={<button className="secondary" onClick={() => void refresh()}><RefreshCcw size={16}/> Refresh</button>}/>
    {error && <div role="alert" className="validation error">{error}</div>}
    <section className="historySummary"><div><span>Experiments</span><strong>{experiments.length}</strong></div><div><span>Recorded runs</span><strong>{runs.length}</strong></div><div><span>Completed</span><strong>{runs.filter((run) => run.status === "completed").length}</strong></div><div><span>Needs attention</span><strong>{runs.filter((run) => ["failed", "interrupted"].includes(run.status)).length}</strong></div></section>
    <section className="surface historyFilters"><Filter size={18}/><select aria-label="Filter experiment" value={experiment} onChange={(event) => setExperiment(event.target.value)}><option value="all">All experiments</option>{experiments.map((item) => <option value={item.experiment_id} key={item.experiment_id}>{item.name}</option>)}</select><input placeholder="Search experiment, model, metric…" value={query} onChange={(event) => setQuery(event.target.value)}/><select value={status} onChange={(event) => setStatus(event.target.value)}><option value="all">All statuses</option><option value="completed">Completed</option><option value="running">Running</option><option value="queued">Queued</option><option value="failed">Failed</option><option value="cancelled">Cancelled</option><option value="interrupted">Interrupted</option></select><select value={task} onChange={(event) => setTask(event.target.value)}><option value="all">All tasks</option><option value="regression">Regression</option><option value="binary_classification">Binary classification</option><option value="multiclass_classification">Multiclass classification</option></select></section>
    <section className="surface"><div className="tableWrap"><table className="historyTable"><thead><tr><th>Experiment / run</th><th>Task</th><th>Model</th><th>Search</th><th>Best metric</th><th>Duration</th><th>Status</th><th>Actions</th></tr></thead><tbody>{filtered.map((run) => <tr key={run.run_id}><td><strong>{run.experiment_name}</strong><span>{run.run_id.slice(0, 12)} · {formatDate(run.created_at)}</span></td><td>{run.task?.replace(/_/g, " ")}</td><td>{run.estimator}</td><td>{run.strategy}</td><td>{format(run.best?.best_metric ?? run.result?.best_metric)}</td><td>{run.duration == null ? "—" : `${run.duration.toFixed(1)}s`}</td><td><StatusBadge status={run.status}/></td><td><div className="rowActions"><button title="Open execution" onClick={() => void open(run)}><Radio size={16}/></button>{run.status === "completed" && <button title="Open results" onClick={() => void open(run, true)}><BarChart3 size={16}/></button>}<button title="Clone specification" disabled={!dashboardSpec(run)} onClick={() => clone(run)}><Copy size={16}/></button></div></td></tr>)}</tbody></table>{!filtered.length && <div className="emptyState"><History size={20}/> No runs match these filters.</div>}</div></section>
  </>;
}
function format(value: unknown) { const number = Number(value); return Number.isFinite(number) ? number.toFixed(4) : "—"; }
function formatDate(value?: string | null) { return value ? new Date(value).toLocaleString() : "Unknown time"; }
