import { CheckCircle2, CircleHelp, X } from "lucide-react";

type Guide = {
  title: string;
  purpose: string;
  outcome: string;
  checklist: string[];
  terms: Array<{ name: string; meaning: string }>;
};

const guides: Record<number, Guide> = {
  1: {
    title: "Prepare trustworthy data",
    purpose: "Choose what the model should predict, inspect data quality, and freeze a reproducible evaluation split.",
    outcome: "A validated dataset, target, preparation plan, and train/validation/test split.",
    checklist: ["Inspect rows, columns, missingness, and target balance", "Choose cross-validation or a holdout, with chronological order when needed", "Set missing-value rules, outlier limits and category encoding"],
    terms: [
      { name: "Target", meaning: "The value or class the model learns to predict." },
      { name: "Validation", meaning: "Data used to compare candidate parameter settings during the search." },
      { name: "Test", meaning: "Untouched data used only for the final, less-biased report." }
    ]
  },
  2: {
    title: "Choose the model recipe",
    purpose: "Select a model compatible with the task, then decide which settings remain fixed and which settings the search may change.",
    outcome: "One available estimator with a valid and bounded parameter space.",
    checklist: ["Choose a compatible model family", "Review probability and interpretation support", "Separate fixed values from tunable ranges"],
    terms: [
      { name: "Fixed parameter", meaning: "A setting used unchanged for every candidate model." },
      { name: "Tunable parameter", meaning: "A setting proposed repeatedly by PSO, random, or grid search." },
      { name: "Probability output", meaning: "Class-confidence scores required for ROC, PR, AUC, and threshold tools." }
    ]
  },
  3: {
    title: "Set the search budget",
    purpose: "Choose how candidate parameters are generated and how much model-fitting work the experiment may perform.",
    outcome: "A validated, frozen run specification ready for the local worker.",
    checklist: ["Choose PSO, random, or grid", "Review the planned model-fit count", "Select an experiment and validate before launch"],
    terms: [
      { name: "Particle", meaning: "One candidate parameter position evaluated during each PSO iteration." },
      { name: "Iteration", meaning: "One complete pass in which every PSO particle is evaluated." },
      { name: "Topology", meaning: "The rule controlling which particle's best result influences another particle." }
    ]
  },
  4: {
    title: "Follow real execution",
    purpose: "Observe the local worker as it prepares data, fits candidate models, validates them, and records durable events.",
    outcome: "A completed result or a readable failure trail that can be retried.",
    checklist: ["Compare finished fits with the planned total", "Watch the current candidate and best score", "Use the timeline to locate warnings or failures"],
    terms: [
      { name: "Queued", meaning: "Waiting for the local worker or for the preceding candidate to finish." },
      { name: "Metric", meaning: "The human-facing measure being optimized, such as ROC AUC or RMSE." },
      { name: "Cost", meaning: "The internal minimization value; lower is better, even when the metric is higher-is-better." }
    ]
  },
  5: {
    title: "Interpret the held-out result",
    purpose: "Review final predictive quality, errors, diagnostics, and explanations on data not used to fit that model.",
    outcome: "A task-appropriate report whose tools and selected split are clearly identified.",
    checklist: ["Choose validation or untouched test results", "Select useful analysis tools", "Inspect predictions and failure cases, not only one score"],
    terms: [
      { name: "ROC AUC", meaning: "Ranking quality across all binary-classification thresholds; higher is better." },
      { name: "Sensitivity", meaning: "The share of actual positive cases correctly detected." },
      { name: "Specificity", meaning: "The share of actual negative cases correctly rejected." }
    ]
  },
  6: {
    title: "Revisit and compare experiments",
    purpose: "Find durable GUI and CLI runs, reopen their execution or results, and reuse a saved configuration.",
    outcome: "A traceable experiment history that supports comparison, cloning, retry, and export.",
    checklist: ["Filter to the relevant task or status", "Compare scores in the correct metric direction", "Open, clone, or retry the chosen run"],
    terms: [
      { name: "Experiment", meaning: "A named collection of related runs intended for comparison." },
      { name: "Run", meaning: "One frozen configuration executed by the worker." },
      { name: "Retry", meaning: "A new linked attempt that preserves the original run and its history." }
    ]
  }
};

export function StageGuide({ stage, onClose }: { stage: number; onClose: () => void }) {
  const guide = guides[stage];
  if (!guide) return null;
  return <section className="stageGuide" aria-label={`Stage ${stage} guide`}>
    <div className="stageGuideLead"><CircleHelp size={22}/><div><span>Quick guide</span><strong>{guide.title}</strong><p>{guide.purpose}</p></div></div>
    <div className="stageChecklist">{guide.checklist.map((item) => <div key={item}><CheckCircle2 size={15}/><span>{item}</span></div>)}</div>
    <dl>{guide.terms.map((term) => <div key={term.name}><dt>{term.name}</dt><dd>{term.meaning}</dd></div>)}</dl>
    <div className="stageOutcome"><span>Ready when</span><strong>{guide.outcome}</strong></div>
    <button type="button" className="iconButton stageGuideClose" onClick={onClose} title="Close stage guide" aria-label="Close stage guide"><X size={17}/></button>
  </section>;
}

export function HelpTip({ label, children }: { label: string; children: React.ReactNode }) {
  return <span className="helpTip"><button type="button" aria-label={`Explain ${label}`}><CircleHelp size={14}/></button><span role="tooltip"><strong>{label}</strong>{children}</span></span>;
}

export function guideSeenKey(stage: number) { return `pspso.stage-guide-seen.${stage}`; }
