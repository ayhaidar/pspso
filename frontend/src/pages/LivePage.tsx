import { Activity, BarChart3, Ban, Clock3, Cpu, RefreshCcw } from "lucide-react";
import { useEffect, useMemo } from "react";
import { useNavigate, useParams } from "react-router-dom";
import { ConvergenceChart } from "../components/Charts";
import { EventTimeline, StrategyCanvas } from "../components/Execution";
import { HelpTip } from "../components/Help";
import { Metric, PageHeading, StatusBadge } from "../components/Layout";
import { dashboardSpec, useWorkflow } from "../workflow";
import type { RunEvent } from "../types";

export function LivePage() {
  const { runId } = useParams();
  const navigate = useNavigate();
  const { run, events, connection, busy, loadRun, cancelRun, retryRun } = useWorkflow();
  useEffect(() => { if (runId && run?.run_id !== runId) void loadRun(runId); }, [runId]);
  const spec = dashboardSpec(run);
  const trials = events.filter((event) => ["trial_completed", "trial_failed"].includes(event.type));
  const chartPoints = events.filter((event) => event.type === "trial_completed").length;
  const started = [...events].reverse().find((event) => event.type === "run_started");
  const plannedTrials = Number(started?.payload.planned_trials ?? spec?.runtime?.max_trials ?? 0);
  const plannedFits = Number(started?.payload.planned_fits ?? plannedTrials);
  const fitEvents = events.filter((event) => event.type === "model_fit_completed" || event.type === "model_fit_failed");
  const finalRefitFits = fitEvents.filter((event) => event.payload.phase === "final_refit");
  const candidateFitEvents = fitEvents.filter((event) => event.payload.phase !== "final_refit");
  const plannedCandidateFits = Number(started?.payload.planned_candidate_fits ?? Math.max(0, plannedFits - Number(spec?.evaluation?.refit_best ?? false)));
  const completedCandidateFits = candidateFitEvents.length || events.filter((event) => event.type === "fold_completed").length || trials.length;
  const finalRefitStarted = events.some((event) => event.type === "final_refit_started");
  const finalRefitCompleted = events.some((event) => event.type === "final_refit_completed");
  const finalRefitFailed = finalRefitFits.some((event) => event.type === "model_fit_failed") || events.some((event) => event.type === "run_warning" && event.payload.reason === "final_refit_failed");
  const finalRefitPlanned = plannedFits > plannedCandidateFits || finalRefitStarted;
  const finalRefitStatus = finalRefitCompleted ? "1 / 1" : finalRefitFailed ? "Failed" : finalRefitStarted ? "Running" : finalRefitPlanned ? "0 / 1" : "Not requested";
  const completedFits = completedCandidateFits + finalRefitFits.length;
  const activeFits = events.filter((event) => event.type === "model_fit_started" && !events.some((later) => ["model_fit_completed", "model_fit_failed"].includes(later.type) && later.payload.fit_id === event.payload.fit_id));
  const progress = plannedFits > 0 ? Math.min(100, Math.floor(completedFits / plannedFits * 100)) : 0;
  const active = [...events].reverse().find((event) => event.type === "trial_started" && !events.some((later) => later.sequence_number! > event.sequence_number! && ["trial_completed", "trial_failed"].includes(later.type) && later.payload.trial_id === event.payload.trial_id));
  const best = [...events].reverse().find((event) => event.type === "best_updated")?.payload ?? run?.best;
  const elapsed = useMemo(() => run?.created_at ? Math.max(0, Date.now() - new Date(run.created_at).getTime()) / 1000 : 0, [run?.created_at, events.length]);
  if (!run) return <><PageHeading eyebrow="Stage 4 of 6" title="Live experiments" description="Start a run from Search Engine or open a saved execution from History."/><div className="emptyState large">No run is selected.</div></>;
  const attempt = run.attempts?.[run.attempts.length - 1];
  const terminal = ["completed", "failed", "cancelled", "interrupted"].includes(run.status);
  const cancelling = !terminal && (run.cancel_requested || busy === "cancel");
  const foldsPerCandidate = plannedTrials > 0 ? Math.max(1, Math.round(plannedCandidateFits / plannedTrials)) : 1;
  return <>
    <PageHeading eyebrow="Stage 4 of 6" title="Live experiments" description="The canvas, metrics, and timeline are driven by persisted worker events. Queued work is never presented as active training." actions={<div className="headerButtons">{!terminal && <button className="dangerButton" disabled={cancelling} onClick={() => void cancelRun()}>{cancelling ? <RefreshCcw size={16}/> : <Ban size={16}/>} {cancelling ? "Cancelling…" : "Cancel"}</button>}{terminal && run.status !== "completed" && <button className="secondary" onClick={() => void retryRun().then((retry) => { if (retry) navigate(`/live/${retry.run_id}`); })}><RefreshCcw size={16}/> Retry</button>}{run.status === "completed" && <button className="primary" onClick={() => navigate(`/results/${run.run_id}`)}><BarChart3 size={16}/> View results</button>}</div>}/>
    <section className="runBanner"><div><span>Experiment</span><strong>{run.experiment_name}</strong></div><div><span>Run ID</span><strong>{run.run_id}</strong></div><div><span>Model</span><strong>{run.estimator}</strong></div><div><span>Strategy</span><strong>{run.strategy}</strong></div><div><span>Connection</span><strong>{connection}</strong></div><StatusBadge status={run.status}/></section>
    <section className="runBanner"><div><span>Queue position</span><strong>{run.queue_position ?? "—"}</strong></div><div><span>Worker</span><strong>{attempt?.worker_pid ?? "Waiting"}</strong></div><div><span>Heartbeat</span><strong>{attempt?.heartbeat_at ? new Date(attempt.heartbeat_at).toLocaleTimeString() : "—"}</strong></div><div><span>Attempt</span><strong>{run.attempts?.length ?? 0}</strong></div>{run.cancel_requested && <strong>Cancellation requested · waiting for cleanup</strong>}{run.parent_run_id && <div><span>Retry of</span><strong>{run.parent_run_id}</strong></div>}</section>
    <div className="metricStrip wide"><Metric label="Candidates" value={`${trials.length} / ${plannedTrials || "—"}`}/><Metric label="Candidate model fits" value={`${completedCandidateFits} / ${plannedCandidateFits || "—"}`}/><Metric label="Winner refit" value={finalRefitStatus}/><Metric label="Successful candidates" value={`${chartPoints}`}/><Metric label="Failures" value={`${run.n_failures ?? trials.filter((event) => event.type === "trial_failed").length}`}/><Metric label="Best metric" value={format(best?.best_metric)}/><Metric label="Elapsed" value={`${Math.round(run.duration ?? elapsed)}s`}/></div>
    <div className="runProgress" aria-label={`${completedCandidateFits} of ${plannedCandidateFits} planned candidate model fits finished`}><span style={{ width: `${progress}%` }}/><strong>{completedCandidateFits} / {plannedCandidateFits || "—"} candidate fits{finalRefitPlanned ? ` · Winner refit ${finalRefitStatus}` : ""} · {progress}%</strong></div>
    <section className="currentActivity"><Activity size={20}/><div><span>Current worker activity</span><strong>{terminal ? `Run ${run.status}` : activeFits.length ? `${activeFits.length} active model fit${activeFits.length === 1 ? "" : "s"}` : active ? `Evaluating trial ${active.payload.trial_id}` : terminal ? `Run ${run.status}` : "Waiting for the next worker event"}</strong><p>{active ? formatParams(active.payload.params) : latestSummary(events)}</p></div><Cpu size={22}/></section>
    {!terminal && activeFits.length > 0 && <section className="runBanner" aria-label="Active model fits">{activeFits.map((fit) => <div key={fit.payload.fit_id}><span>Worker slot {Number(fit.payload.worker_slot ?? 0) + 1}</span><strong>{fit.payload.phase === "final_refit" ? "Final refit" : `Candidate ${fit.payload.trial_id} / fold ${fit.payload.fold ?? 1}`}</strong></div>)}</section>}
    <div className="liveGrid"><section className="surface canvasSurface"><StrategyCanvas run={run} events={events}/></section><section className="surface chartSurface"><div className="chartHeading"><h2>Best cost by completed candidate <HelpTip label="Optimization cost">The best candidate cost found so far. Lower is better; higher-is-better metrics such as ROC AUC are converted internally.</HelpTip></h2><span>{chartPoints} of {plannedTrials || "—"} candidate scores · {foldsPerCandidate === 1 ? "1 validation fit each" : `${foldsPerCandidate} fold fits per score`}</span></div><ConvergenceChart events={events}/></section></div>
    <div className="liveGrid lower"><section className="surface"><h2>Chronological execution timeline</h2><EventTimeline events={events}/></section><section className="surface"><h2>Trial ledger</h2><TrialTable events={events}/></section></div>
  </>;
}

function TrialTable({ events }: { events: RunEvent[] }) { const rows = events.filter((event) => ["trial_completed", "trial_failed"].includes(event.type)); return <div className="tableWrap"><table><thead><tr><th>Trial</th><th>Status</th><th>Metric</th><th>Cost</th><th>Parameters</th></tr></thead><tbody>{rows.map((event) => <tr key={event.sequence_number}><td>{String(event.payload.trial_id ?? "—")}</td><td><StatusBadge status={event.type === "trial_failed" ? "failed" : "completed"}/></td><td>{format(event.payload.metric ?? event.payload.metric_value)}</td><td>{format(event.payload.cost)}</td><td><code>{formatParams(event.payload.params)}</code></td></tr>)}</tbody></table>{!rows.length && <div className="emptyCompact"><Clock3 size={16}/> Trials appear here after the worker starts.</div>}</div>; }
function format(value: unknown) { if (value == null) return "—"; const number = Number(value); return Number.isFinite(number) ? number.toFixed(4) : "—"; }
function formatParams(params: unknown) { return params && typeof params === "object" ? Object.entries(params as Record<string, unknown>).map(([key, value]) => `${key}=${value}`).join(", ") : "Parameters are recorded with the trial."; }
function latestSummary(events: RunEvent[]) { const event = events[events.length - 1]; return event ? event.type.replace(/_/g, " ") : "The run is queued in the local workspace."; }
