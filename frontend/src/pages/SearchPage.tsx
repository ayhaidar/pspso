import { ArrowLeft, Clock, Grid3X3, Play, Save, Shuffle, Sparkles } from "lucide-react";
import { useEffect, useState } from "react";
import { useNavigate } from "react-router-dom";
import { api } from "../api";
import { PageHeading, ValidationSummary } from "../components/Layout";
import { HelpTip } from "../components/Help";
import type { MetricId, JsonValue, SearchParam, Strategy } from "../types";
import { useWorkflow } from "../workflow";

const strategies = [
  { id: "pso" as const, label: "Particle swarm", icon: Sparkles, text: "Particles share information while exploring continuous and categorical domains." },
  { id: "random" as const, label: "Random search", icon: Shuffle, text: "Independent reproducible samples provide a strong baseline with a fixed budget." },
  { id: "grid" as const, label: "Grid search", icon: Grid3X3, text: "Visit parameter combinations deterministically. Best for small discrete spaces." }
];

export function SearchPage() {
  const navigate = useNavigate();
  const { draft, metadata, preview, experiments, busy, refreshCollections, updateDraft, validateStage, createExperiment, startRun } = useWorkflow();
  const [name, setName] = useState("");
  const [description, setDescription] = useState("");
  const [tournamentModels, setTournamentModels] = useState<string[]>([]);
  const [launching, setLaunching] = useState(false);
  const [error, setError] = useState("");
  useEffect(() => { setTournamentModels([]); }, [draft.task]);
  const estimate = draft.strategy === "pso" ? draft.pso.particles * draft.pso.iterations : draft.runtime.max_trials ?? estimateGrid(draft.search_space);
  const candidateFits = estimate * (draft.evaluation.protocol === "cross_validation" ? draft.evaluation.folds : 1);
  const totalFits = candidateFits + (draft.evaluation.refit_best ? 1 : 0);
  const candidateFitLabel = `${candidateFits} candidate model fit${candidateFits === 1 ? "" : "s"}`;
  async function launch() {
    if (launching) return;
    setLaunching(true); setError("");
    try {
    if (tournamentModels.length < 2) {
      const created = await startRun();
      if (created) navigate(`/live/${created.run_id}`);
      return;
    }
    if (!metadata) return;
    let experimentId = draft.experiment_id;
    if (!experimentId) experimentId = await createExperiment(`Model tournament ${new Date().toLocaleString()}`, "Comparable runs created from the Search Engine.");
    if (!experimentId) return;
    const runs = tournamentModels.map((estimator) => {
      const defaults = metadata.estimators[estimator].defaults[draft.task];
      return { ...draft, experiment_id: experimentId, estimator, fixed_params: constantsOnly(defaults.fixed_params, defaults.search_space), search_space: defaults.search_space };
    });
    const created = await api.tournament(experimentId, runs);
    await refreshCollections();
    if (created.runs[0]) navigate(`/live/${created.runs[0].run_id}`);
    } catch (error) { setError(error instanceof Error ? error.message : "The tournament could not start."); }
    finally { setLaunching(false); }
  }
  async function saveExperiment() { if (!name.trim()) return; const id = await createExperiment(name.trim(), description.trim()); if (id) { setName(""); setDescription(""); } }
  function selectStrategy(strategy: Strategy) {
    const maxTrials = strategy === "pso" ? null : strategy === "random" ? 10 : 100;
    updateDraft({ strategy, runtime: { ...draft.runtime, max_trials: maxTrials } });
  }
  return <>
    <PageHeading eyebrow="Stage 3 of 6" title="Build the search engine" description="Choose how candidates are generated, set a transparent compute budget, validate the complete specification, and launch it into the local worker queue." />
    <section className="surface">
      <div className="strategyCards">{strategies.map((item) => { const Icon = item.icon; return <button type="button" key={item.id} className={`strategyCard ${draft.strategy === item.id ? "selected" : ""}`} onClick={() => selectStrategy(item.id)}><Icon size={22}/><strong>{item.label}</strong><span>{item.text}</span></button>; })}</div>
    </section>
    <div className="twoColumnPage searchConfiguration">
      <section className="surface primarySurface">
        <h2>Strategy controls</h2>
        {draft.strategy === "pso" ? <PsoControls/> : <BudgetControls strategy={draft.strategy}/>}
        <h2>Evaluation protocol</h2><label>Optimization metric<select value={draft.metric} onChange={(event) => updateDraft({ metric: event.target.value as MetricId })}>{(metadata?.tasks[draft.task]?.metrics ?? [draft.metric]).map((metric) => <option key={metric} value={metric}>{metadata?.metrics[metric]?.label ?? metric}</option>)}</select><small>{metadata?.metrics[draft.metric]?.description}</small></label>
        <p className="fieldHint">{draft.evaluation.protocol === "cross_validation" ? `${draft.evaluation.folds}-fold ${draft.split.method === "chronological" ? "expanding-window" : "cross"}-validation` : "Train / validation / test"} · {draft.split.method === "chronological" ? "chronological order" : "random split"}. <a href="/data#data-profile">Change evaluation in Data setup</a></p><div className="formGrid"><label>Parallel candidates<input type="number" min="1" max="64" value={draft.runtime.trial_workers} onChange={(event) => updateDraft({ runtime: { ...draft.runtime, trial_workers: Number(event.target.value) } })}/></label>{draft.task === "binary_classification" && <label>Decision threshold<input type="number" min="0" max="1" step="0.05" value={draft.evaluation.decision_threshold} onChange={(event) => updateDraft({ evaluation: { ...draft.evaluation, decision_threshold: Number(event.target.value) } })}/></label>}</div>
        <div className="formGrid">
          {draft.task === "binary_classification" && <label>Positive class<select aria-label="Positive class" value={JSON.stringify(draft.evaluation.positive_label)} onChange={(event) => updateDraft({ evaluation: { ...draft.evaluation, positive_label: JSON.parse(event.target.value) } })}><option value="null">Automatic · last ordered class</option>{preview?.target_summary?.distribution?.map((item) => <option key={String(item.label)} value={JSON.stringify(item.label)}>{String(item.label)}</option>)}</select><small>Inspect data to choose a class. AUC, sensitivity and the threshold use this selection.</small></label>}
        </div>
        <div className="modelOutputChoice" role="group" aria-labelledby="saved-model-choice-label">
          <header><div><span id="saved-model-choice-label">Saved model after evaluation</span><strong>Choose what is trained and saved after the search finds its winner</strong></div><small>The search winner and its candidate score stay the same with either choice.</small></header>
          <div className="modelOutputCards">
            <button type="button" aria-pressed={draft.evaluation.refit_best} className={`modelOutputCard ${draft.evaluation.refit_best ? "selected" : ""}`} onClick={() => updateDraft({ evaluation: { ...draft.evaluation, refit_best: true } })}>
              <span className="modelOutputTitle"><strong>Refit one winner</strong><em>1 additional fit</em></span>
              <span>Train a new single model on {draft.evaluation.protocol === "cross_validation" ? "all development rows after cross-validation" : "the combined training and validation rows"}.</span>
              <span className="modelOutputFact"><b>Saved:</b> one newly fitted model</span>
              <span className="modelOutputFact"><b>Score:</b> its test score may differ because this is a new training pass on more rows.</span>
            </button>
            <button type="button" aria-pressed={!draft.evaluation.refit_best} className={`modelOutputCard ${!draft.evaluation.refit_best ? "selected" : ""}`} onClick={() => updateDraft({ evaluation: { ...draft.evaluation, refit_best: false } })}>
              <span className="modelOutputTitle"><strong>Keep evaluated model{draft.evaluation.protocol === "cross_validation" ? "s" : ""}</strong><em>No extra fit</em></span>
              <span>{draft.evaluation.protocol === "cross_validation" ? `Save all ${draft.evaluation.folds} fitted fold models for the winning candidate and average their predictions.` : "Save the winning model already fitted on the training partition."}</span>
              <span className="modelOutputFact"><b>Saved:</b> {draft.evaluation.protocol === "cross_validation" ? `${draft.evaluation.folds}-model ensemble` : "one evaluated holdout model"}</span>
              <span className="modelOutputFact"><b>Score:</b> no new model initialization; {draft.evaluation.protocol === "cross_validation" ? "the ensemble is a different predictor from a single refitted model." : "the saved model has not seen validation rows."}</span>
            </button>
          </div>
          {(draft.estimator === "pytorch_mlp" || draft.estimator === "sklearn_mlp") && <p className="neuralRefitNote"><strong>Neural network note:</strong> refitting starts a new training pass. The saved seed makes it repeatable, while using more training rows can still produce different weights and a different test result.</p>}
        </div>
        <h2>Runtime guardrails</h2>
        <div className="formGrid"><label>Timeout in seconds<input type="number" min="1" placeholder="No timeout" value={draft.runtime.timeout_seconds ?? ""} onChange={(event) => updateDraft({ runtime: { ...draft.runtime, timeout_seconds: event.target.value ? Number(event.target.value) : null } })}/></label><label>Early stopping rounds<input type="number" min="1" placeholder="Disabled" value={draft.runtime.early_stopping_rounds ?? ""} onChange={(event) => updateDraft({ runtime: { ...draft.runtime, early_stopping_rounds: event.target.value ? Number(event.target.value) : null } })}/></label></div>
        {error && <div role="alert" className="validation error">{error}</div>}<ValidationSummary stage="search"/><ValidationSummary stage="full"/>
      </section>
      <aside className="surface launchPanel">
        <h2>Run summary</h2>
        <div className="summaryList"><Summary label="Dataset" value={draft.dataset.source === "example" ? draft.dataset.name ?? "Example" : draft.dataset.source}/><Summary label="Task" value={metadata?.tasks[draft.task]?.label ?? draft.task}/><Summary label="Model" value={metadata?.estimators[draft.estimator]?.label ?? draft.estimator}/><Summary label="Metric" value={metadata?.metrics[draft.metric]?.label ?? draft.metric}/><Summary label="Strategy" value={draft.strategy}/><Summary label="Candidate settings" value={draft.strategy === "pso" ? `${estimate} (${draft.pso.particles} particles × ${draft.pso.iterations} iterations)` : String(estimate)}/><Summary label="Candidate model fits" value={draft.evaluation.protocol === "cross_validation" ? `${candidateFits} (${estimate} candidates × ${draft.evaluation.folds} folds)` : `${candidateFits} (${estimate} candidates × 1 holdout)`}/><Summary label="Saved model" value={draft.evaluation.refit_best ? "Refitted winner · 1 additional fit" : draft.evaluation.protocol === "cross_validation" ? `${draft.evaluation.folds}-model fold ensemble` : "Evaluated holdout winner"}/><Summary label="Candidate scheduling" value={`${draft.runtime.trial_workers} concurrent worker${draft.runtime.trial_workers === 1 ? "" : "s"}`}/></div>
        <div className="experimentBox"><h3>Save under an experiment</h3><select value={draft.experiment_id ?? ""} onChange={(event) => updateDraft({ experiment_id: event.target.value || null })}><option value="">Create an ad hoc experiment</option>{experiments.map((item) => <option value={item.experiment_id} key={item.experiment_id}>{item.name} · {item.run_count} runs</option>)}</select><details><summary>Create a named experiment</summary><label>Name<input value={name} onChange={(event) => setName(event.target.value)}/></label><label>Description<textarea rows={3} value={description} onChange={(event) => setDescription(event.target.value)}/></label><button className="secondary" type="button" disabled={!name.trim()} onClick={() => void saveExperiment()}><Save size={16}/> Save experiment</button></details></div>
        <div className="experimentBox"><h3>Optional model tournament</h3><p className="fieldHint">Select two or more available models to queue comparable runs with the same data, metric, evaluation protocol, and budget.</p><div className="checkList">{Object.entries(metadata?.estimators ?? {}).filter(([, model]) => model.tasks.includes(draft.task) && model.dependency.installed).map(([id, model]) => <label key={id}><input type="checkbox" checked={tournamentModels.includes(id)} onChange={(event) => setTournamentModels((current) => event.target.checked ? [...current, id] : current.filter((item) => item !== id))}/>{model.label}</label>)}</div></div>
        {tournamentModels.length === 1 && <p role="status">Select at least two models for a tournament, or clear the selection for a single run.</p>}<div className="budgetNotice"><Clock size={18}/><p><strong>{candidateFitLabel}{draft.evaluation.refit_best ? " + 1 winner refit" : ""} planned per model</strong><span>{totalFits} total fit{totalFits === 1 ? "" : "s"}. {draft.strategy === "pso" ? `${draft.pso.particles} particles × ${draft.pso.iterations} iterations × ${draft.evaluation.protocol === "cross_validation" ? `${draft.evaluation.folds} folds` : "one holdout"}. Up to ${draft.runtime.trial_workers} particles train together.` : `Candidates use ${draft.evaluation.protocol === "cross_validation" ? `${draft.evaluation.folds} reproducible folds` : "one holdout split"}.`}</span></p></div>
        <button className="secondary fullButton" onClick={() => void validateStage("search")} disabled={busy === "search"}>Validate complete setup</button>
        <button className="primary fullButton" onClick={() => void launch()} disabled={launching || busy !== null || tournamentModels.length === 1}><Play size={17}/> {tournamentModels.length >= 2 ? `Start ${tournamentModels.length}-model tournament` : "Start experiment run"}</button>
        <button className="textButton" onClick={() => navigate("/model")}><ArrowLeft size={16}/> Back to model</button>
      </aside>
    </div>
  </>;
}

function PsoControls() { const { draft, updateDraft } = useWorkflow(); const change = (patch: Partial<typeof draft.pso>) => updateDraft({ pso: { ...draft.pso, ...patch } }); return <div className="formGrid"><label><span className="labelWithHelp">Particles <HelpTip label="Particles">Candidate parameter positions evaluated once in every iteration.</HelpTip></span><input type="number" min="2" value={draft.pso.particles} onChange={(event) => change({ particles: Number(event.target.value) })}/></label><label><span className="labelWithHelp">Iterations <HelpTip label="Iterations">Complete search rounds. Each round evaluates every particle.</HelpTip></span><input type="number" min="1" value={draft.pso.iterations} onChange={(event) => change({ iterations: Number(event.target.value) })}/></label><label><span className="labelWithHelp">Topology <HelpTip label="PSO topology">Global uses the swarm's overall best; ring local uses each particle's nearby neighbours.</HelpTip></span><select value={draft.pso.topology} onChange={(event) => change({ topology: event.target.value as "global" | "local" })}><option value="global">Global best</option><option value="local">Ring local</option></select></label><label>Cognitive coefficient c1<input type="number" step="0.1" value={draft.pso.c1} onChange={(event) => change({ c1: Number(event.target.value) })}/></label><label>Social coefficient c2<input type="number" step="0.1" value={draft.pso.c2} onChange={(event) => change({ c2: Number(event.target.value) })}/></label><label>Inertia w<input type="number" step="0.05" value={draft.pso.w} onChange={(event) => change({ w: Number(event.target.value) })}/></label></div>; }
function BudgetControls({ strategy }: { strategy: Strategy }) { const { draft, updateDraft } = useWorkflow(); return <div className="formGrid"><label>{strategy === "grid" ? "Maximum combinations" : "Number of trials"}<input type="number" min="1" value={draft.runtime.max_trials ?? 10} onChange={(event) => updateDraft({ runtime: { ...draft.runtime, max_trials: Number(event.target.value) } })}/></label><label>Sampling seed<input type="number" value={draft.split.random_state} onChange={(event) => updateDraft({ split: { ...draft.split, random_state: Number(event.target.value) } })}/></label></div>; }
function Summary({ label, value }: { label: string; value: string }) { return <div><span>{label}</span><strong>{value}</strong></div>; }
function estimateGrid(space: Record<string, SearchParam>) { return Math.max(1, Object.values(space).reduce((total: number, spec) => total * (spec.type === "choice" ? spec.values.length : spec.type === "int" ? Math.min(20, spec.high - spec.low + 1) : 5), 1)); }
function constantsOnly(fixed: Record<string, JsonValue>, search: Record<string, unknown>) { return Object.fromEntries(Object.entries(fixed).filter(([name]) => !(name in search))); }
