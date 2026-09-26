import { AlertTriangle, CheckCircle2, Clock3, LoaderCircle, Terminal } from "lucide-react";
import type { RunEvent, RunSnapshot } from "../types";

import { dashboardSpec } from "../workflow";

type TrialState = { id: number; state: "queued" | "active" | "completed" | "failed" | "stopped"; iteration: number; slot: number; metric?: number };

function trialStates(events: RunEvent[], expected: number): TrialState[] {
  const states = new Map<number, TrialState>();
  for (let id = 1; id <= expected; id += 1) states.set(id, { id, state: "queued", iteration: 0, slot: id - 1 });
  for (const event of events) {
    if (!["trial_started", "trial_completed", "trial_failed"].includes(event.type)) continue;
    const id = Number(event.payload.trial_id ?? states.size + 1);
    const current = states.get(id) ?? { id, state: "queued", iteration: 0, slot: id - 1 };
    states.set(id, {
      ...current,
      iteration: Number(event.payload.iteration ?? current.iteration),
      slot: Number(event.payload.particle_index ?? event.payload.strategy_slot ?? current.slot),
      state: event.type === "trial_started" ? "active" : event.type === "trial_failed" ? "failed" : "completed",
      metric: numeric(event.payload.metric ?? event.payload.metric_value ?? event.payload.score)
    });
  }
  return [...states.values()].sort((a, b) => a.id - b.id);
}

export function StrategyCanvas({ run, events }: { run: RunSnapshot; events: RunEvent[] }) {
  const request = dashboardSpec(run);
  const strategy = run.strategy ?? request?.strategy ?? "random";
  const announced = [...events].reverse().find((event) => event.type === "run_started")?.payload.planned_trials;
  const expected = Number(announced ?? (strategy === "pso"
    ? (request?.pso?.particles ?? 5) * (request?.pso?.iterations ?? 1)
    : request?.runtime?.max_trials ?? Math.max(1, run.n_trials ?? 1)));
  const states = trialStates(events, Math.min(expected, 40));
  if (["completed", "failed", "cancelled", "interrupted"].includes(run.status)) {
    for (const state of states) if (state.state === "active" || state.state === "queued") state.state = "stopped";
  }
  const completedCount = events.filter((event) => ["trial_completed", "trial_failed"].includes(event.type)).length;
  if (strategy === "grid") return <GridCanvas states={states} completedCount={completedCount} expected={expected} />;
  if (strategy === "pso") return <PsoCanvas states={states} completedCount={completedCount} particles={request?.pso?.particles ?? 5} iterations={request?.pso?.iterations ?? 1} topology={request?.pso?.topology ?? "global"} workers={request?.runtime?.trial_workers ?? 1} />;
  return <RandomCanvas states={states} completedCount={completedCount} expected={expected} />;
}

function PsoCanvas({ states, completedCount, particles, iterations, topology, workers }: { states: TrialState[]; completedCount: number; particles: number; iterations: number; topology: string; workers: number }) {
  const iteration = Math.max(0, ...states.filter((item) => item.state !== "queued").map((item) => item.iteration));
  const visible = Array.from({ length: particles }, (_, slot) => {
    const matches = states.filter((state) => state.iteration === iteration && state.slot === slot);
    return matches[matches.length - 1] ?? { id: slot + 1, state: "queued", iteration, slot };
  });
  const points = visible.map((state, index) => ({ ...state, x: 80 + index * (760 / Math.max(1, particles - 1)), y: 115 + (index % 2 ? 35 : -20) }));
  const active = visible.filter((state) => state.state === "active").length;
  return <div className="executionCanvas"><div className="canvasTitle"><strong>PSO particle field</strong><span>Iteration {Math.min(iteration + 1, iterations)} of {iterations} · {completedCount} of {particles * iterations} candidates finished · {active || 0} active</span></div><svg viewBox="0 0 920 230" role="img" aria-label="PSO particle execution state">
    {topology === "global" && points.map((point) => <line key={`g-${point.id}`} x1={460} y1={115} x2={point.x} y2={point.y} className="topologyLine" />)}
    {topology === "local" && points.map((point, index) => { const next = points[(index + 1) % points.length]; return <line key={`l-${point.id}`} x1={point.x} y1={point.y} x2={next.x} y2={next.y} className="topologyLine" />; })}
    {points.map((point) => <g key={point.id} className={`trialNode ${point.state}`}><circle cx={point.x} cy={point.y} r="25"/><text x={point.x} y={point.y + 4}>P{point.slot + 1}</text></g>)}
  </svg><div className="sequentialNotice">{workers > 1 ? `Up to ${workers} particles are evaluated concurrently. Each iteration waits for its active particles before the swarm moves.` : "Particles are evaluated sequentially. Increase Parallel candidates in Search Engine to train more than one at a time."}</div><CanvasLegend /></div>;
}

function RandomCanvas({ states, completedCount, expected }: { states: TrialState[]; completedCount: number; expected: number }) {
  const points = states.slice(0, 24).map((state, index) => ({ ...state, x: 55 + pseudo(index * 19 + 3) * 810, y: 50 + pseudo(index * 31 + 9) * 125 }));
  return <div className="executionCanvas"><div className="canvasTitle"><strong>Random sample map</strong><span>{completedCount} of {expected} sampled candidates finished</span></div><svg viewBox="0 0 920 230" role="img" aria-label="Random search execution state">
    {points.map((point) => <g key={point.id} className={`trialNode ${point.state}`}><circle cx={point.x} cy={point.y} r="19"/><text x={point.x} y={point.y + 4}>{point.id}</text></g>)}
  </svg><CanvasLegend /></div>;
}

function GridCanvas({ states, completedCount, expected }: { states: TrialState[]; completedCount: number; expected: number }) {
  const columns = Math.ceil(Math.sqrt(states.length));
  return <div className="executionCanvas"><div className="canvasTitle"><strong>Grid combination map</strong><span>{completedCount} of {expected} combinations finished</span></div><svg viewBox="0 0 920 230" role="img" aria-label="Grid search execution state">
    {states.slice(0, 30).map((state, index) => { const x = 55 + (index % columns) * Math.min(95, 800 / columns); const y = 45 + Math.floor(index / columns) * 62; return <g key={state.id} className={`gridNode ${state.state}`}><rect x={x} y={y} width="48" height="42" rx="3"/><text x={x + 24} y={y + 26}>{state.id}</text></g>; })}
  </svg><CanvasLegend /></div>;
}

function CanvasLegend() { return <div className="canvasLegend"><span className="queued">Queued</span><span className="active">Training</span><span className="completed">Completed</span><span className="failed">Failed</span></div>; }
function pseudo(seed: number) { const value = Math.sin(seed * 999) * 10000; return value - Math.floor(value); }
function numeric(value: unknown) { const result = Number(value); return Number.isFinite(result) ? result : undefined; }

export function EventTimeline({ events }: { events: RunEvent[] }) {
  return <div className="timeline">{[...events].reverse().slice(0, 80).map((event) => {
    const Icon = event.type.includes("failed") ? AlertTriangle : event.type.includes("completed") || event.type === "best_updated" ? CheckCircle2 : event.type.includes("started") ? LoaderCircle : event.type === "run_log" ? Terminal : Clock3;
    return <div className={`timelineRow ${event.type.includes("failed") ? "danger" : ""}`} key={`${event.sequence_number}-${event.type}`}><Icon size={16}/><time>{new Date(event.timestamp).toLocaleTimeString()}</time><div><strong>{event.type.replace(/_/g, " ")}</strong><span>{summarize(event)}</span></div></div>;
  })}</div>;
}

function summarize(event: RunEvent) {
  const payload = event.payload;
  if (payload.error) return String(payload.error);
  if (event.type === "validation_passed") return `Configuration accepted for ${payload.estimator ?? "the selected model"}`;
  if (event.type === "run_queued") return "Waiting for an available local worker";
  if (event.type === "worker_started") return `Local worker started${payload.pid ? ` · process ${payload.pid}` : ""}`;
  if (event.type === "dataset_prepared") return `${payload.train_rows ?? "—"} training rows · ${payload.validation_rows ?? "—"} validation rows · ${payload.features ?? "—"} prepared features`;
  if (event.type === "run_started") {
    const candidateFits = Number(payload.planned_candidate_fits ?? payload.planned_fits ?? 0);
    const winnerRefits = Math.max(0, Number(payload.planned_fits ?? candidateFits) - candidateFits);
    return `${String(payload.strategy ?? "search").toUpperCase()} started · ${payload.planned_trials ?? "—"} candidates · ${candidateFits || "—"} candidate fits${winnerRefits ? ` + ${winnerRefits} winner refit` : ""}`;
  }
  if (event.type === "trial_started") return `Trial ${payload.trial_id} · ${formatParams(payload.params)}`;
  if (event.type === "fold_started") return `Trial ${payload.trial_id} · fold ${payload.fold} of ${payload.total_folds} · worker ${Number(payload.worker_slot ?? 0) + 1}`;
  if (event.type === "fold_completed") return `Trial ${payload.trial_id} · fold ${payload.fold} finished · metric ${format(payload.metric)}`;
  if (event.type === "trial_completed") return `Trial ${payload.trial_id} · metric ${format(payload.metric ?? payload.metric_value ?? payload.score)} · cost ${format(payload.cost)}`;
  if (event.type === "best_updated") return `New best ${format(payload.best_metric)} · ${formatParams(payload.best_params)}`;
  if (event.type === "training_epoch") return `Epoch ${payload.epoch} · train ${format(payload.train_loss)} · validation ${format(payload.validation_loss)}`;
  if (event.type === "final_refit_started") return "Refitting the selected configuration on the complete development partition";
  if (event.type === "final_refit_completed") return `Final model fitted on ${payload.rows ?? "all"} development rows`;
  if (event.type === "iteration_completed") return `Iteration ${Number(payload.iteration ?? 0) + 1} finished · best metric ${format(payload.best_metric)} · best cost ${format(payload.best_cost)}`;
  if (event.type === "run_completed") return `${payload.n_trials ?? "All"} trials finished in ${formatDuration(payload.duration)} · best metric ${format(payload.best_metric)}`;
  if (event.type === "run_cancelled") return String(payload.reason ?? "Run cancelled");
  if (event.type === "run_warning" && payload.reason === "early_stopping") return "Stopped early because the score did not improve";
  if (event.type === "run_log") return String(payload.message ?? payload.text ?? "Worker log recorded");
  return Object.entries(payload).slice(0, 3).map(([key, value]) => `${humanize(key)}: ${formatValue(value)}`).join(" · ");
}
function format(value: unknown) { const number = Number(value); return Number.isFinite(number) ? number.toFixed(4) : "n/a"; }
function formatDuration(value: unknown) { const number = Number(value); return Number.isFinite(number) ? `${number.toFixed(number < 10 ? 1 : 0)}s` : "recorded time"; }
function formatValue(value: unknown) { if (typeof value === "number") return Number.isInteger(value) ? String(value) : format(value); if (typeof value === "object" && value !== null) return "details recorded"; return String(value); }
function humanize(value: string) { return value.replace(/_/g, " "); }
function formatParams(value: unknown) { if (!value || typeof value !== "object") return "parameters recorded"; return Object.entries(value as Record<string, unknown>).slice(0, 3).map(([key, item]) => `${key}=${typeof item === "number" ? Number(item.toPrecision(4)) : item}`).join(", "); }
