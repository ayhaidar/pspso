import { AlertTriangle, ArrowRight, BarChart3, Columns3, Database, Eye, Save, Table2 } from "lucide-react";
import { useState } from "react";
import { useNavigate } from "react-router-dom";
import { PageHeading, ValidationSummary } from "../components/Layout";
import { EvaluationControls } from "../components/EvaluationControls";
import { FeaturePreparation } from "../components/FeaturePreparation";
import type { DatasetPreview, Task } from "../types";
import { useWorkflow } from "../workflow";

type InspectionView = "rows" | "columns" | "statistics";

export function DataPage() {
  const navigate = useNavigate();
  const [view, setView] = useState<InspectionView>("rows");
  const { draft, exampleDatasets, datasets, preview, busy, metadata, updateDraft, setTask, inspectDataset, saveCsvDataset, validateStage } = useWorkflow();
  const dataset = draft.dataset;
  const example = exampleDatasets.find((item) => item.name === dataset.name);
  const warnings = preview ? buildWarnings(preview, draft.task, dataset.target_column) : [];
  const targetMissing = preview?.target_summary?.missing ?? 0;
  async function continueFlow() { if (await validateStage("data")) navigate("/model"); }
  function updateDataset(patch: Partial<typeof dataset>) {
    const next = { ...dataset, ...patch };
    if (next.source === "csv") next.name = null;
    else if (next.source === "stored") next.name = datasets.find((item) => item.dataset_id === next.dataset_id)?.name ?? null;
    else next.name ??= "breast_cancer";
    updateDraft({ dataset: next, preprocessing: { ...draft.preprocessing, features: {}, ignored_columns: [] }, split: { ...draft.split, time_column: null } });
  }
  return <>
    <PageHeading eyebrow="Stage 1 of 6" title="Data setup" description="Choose your data, understand its quality, select an evaluation method, and prepare features before choosing a model." />
    <div className="dataSetupSteps" aria-label="Data setup sections"><a href="#dataset-source">1 <span>Dataset and target</span></a><a href="#data-profile">2 <span>Profile and evaluation</span></a><a href="#preparation-title">3 <span>Feature engineering</span></a></div>
    <section className="surface" id="dataset-source">
      <h2><Database size={19}/> 1. Dataset and target</h2>
      <div className="segmented">{(["example", "stored", "csv"] as const).map((source) => <button type="button" key={source} className={dataset.source === source ? "active" : ""} onClick={() => updateDataset({ source })}>{source === "csv" ? "Import CSV" : source}</button>)}</div>
      <div className="datasetSourceGrid"><div>
        {dataset.source === "example" && <><label>Example dataset<select value={dataset.name ?? "breast_cancer"} onChange={(event) => {
          const selected = exampleDatasets.find((item) => item.name === event.target.value);
          if (!selected) return;
          updateDataset({ name: selected.name, target_column: selected.target_column }); setTask(selected.task, selected.default_metric);
        }}>{exampleDatasets.length ? exampleDatasets.map((item) => <option value={item.name} key={item.name}>{item.label} · {metadata?.tasks[item.task]?.label ?? item.task}</option>) : <option value="breast_cancer">Loading example catalogue…</option>}</select></label>{example && <article className="datasetPreset"><header><div><span>{example.source}</span><strong>{example.label}</strong></div><b>{metadata?.tasks[example.task]?.label ?? example.task.replace(/_/g, " ")}</b></header><p>{example.description}</p><dl><div><dt>Standard task</dt><dd>{metadata?.tasks[example.task]?.label ?? example.task}</dd></div><div><dt>Target</dt><dd><code>{example.target_column}</code> · {example.target_description}</dd></div><div><dt>Size</dt><dd>{example.rows.toLocaleString()} rows · {example.features} features</dd></div><div><dt>Default metric</dt><dd>{metadata?.metrics[example.default_metric]?.label ?? example.default_metric}</dd></div></dl><footer><a href={example.license_url} target="_blank" rel="noreferrer">{example.license}</a><a href={example.source_url} target="_blank" rel="noreferrer">View dataset source</a></footer></article>}</>}
        {dataset.source === "stored" && <label>Saved dataset<select value={dataset.dataset_id ?? ""} onChange={(event) => updateDataset({ dataset_id: event.target.value })}><option value="">Select a saved version</option>{datasets.map((item) => <option value={item.dataset_id} key={item.dataset_id}>{item.name} · {item.profile.row_count} rows</option>)}</select></label>}
        {dataset.source === "csv" && <><label>CSV contents<textarea rows={6} value={dataset.csv_text ?? ""} onChange={(event) => updateDataset({ csv_text: event.target.value })} placeholder="feature_1,feature_2,target" /></label><button className="secondary" type="button" onClick={() => void saveCsvDataset(`Dataset ${new Date().toLocaleDateString()}`)} disabled={!dataset.csv_text}><Save size={16}/> Save as dataset version</button></>}
      </div><div className="formGrid">
        <label>Target column<select aria-label="Target column" value={dataset.target_column} onChange={(event) => updateDataset({ target_column: event.target.value })}>{(preview?.target_candidates ?? [dataset.target_column]).map((name) => <option key={name}>{name}</option>)}</select><small>The outcome or future value you want to predict.</small></label>
        <label>Task<select aria-label="Task" value={draft.task} onChange={(event) => setTask(event.target.value as Task)}>{metadata && Object.entries(metadata.tasks).map(([name, item]) => <option key={name} value={name}>{item.label}</option>)}</select></label>
      </div></div>
      <div className="footerActions"><span className="inspectionRequirement">Inspect data to reveal columns, statistics and feature options.</span><button className="secondary" type="button" onClick={() => void inspectDataset()} disabled={busy === "dataset"}><Eye size={16}/> {preview ? "Refresh inspection" : "Inspect data"}</button></div>
    </section>

    <section className="surface profileEvaluation" id="data-profile">
      <h2><BarChart3 size={19}/> 2. Data profile and evaluation</h2>
      {!preview ? <div className="emptyCompact">Inspect a dataset to see row counts, missing values, target balance and descriptive statistics. You can choose the evaluation method below now.</div> : <>
        <div className="metricStrip wide"><Metric label="Rows" value={preview.row_count}/><Metric label="Features" value={preview.columns.length - 1}/><Metric label="Missing cells" value={preview.summary.total_missing}/><Metric label="Duplicate rows" value={preview.summary.duplicate_rows}/></div>
        {!!warnings.length && <div className="warningList">{warnings.map((warning) => <p key={warning}>{warning}</p>)}</div>}
        <details className="profileDetails"><summary>Target profile and data statistics</summary><div className="targetProfile"><h3>Target profile</h3><TargetSummary preview={preview} task={draft.task}/></div>
          <div className="inspectionHeader"><div><h3>Tabular inspection</h3><p>Profile statistics describe the source data; preparation rules learn from training rows only.</p></div><div className="viewTabs" role="tablist">{(["rows", "columns", "statistics"] as InspectionView[]).map((name) => <button role="tab" aria-selected={view === name} className={view === name ? "active" : ""} key={name} onClick={() => setView(name)}>{name === "rows" ? <Table2 size={15}/> : name === "columns" ? <Columns3 size={15}/> : <BarChart3 size={15}/>} {name}</button>)}</div></div>
          {view === "rows" && <RowPreview preview={preview}/>}{view === "columns" && <ColumnProfile preview={preview} target={dataset.target_column}/>}{view === "statistics" && <NumericStatistics preview={preview}/>}
        </details>
      </>}
      <EvaluationControls/>
      {preview && <div className="proposedPartitions"><div className="sectionIntro"><h3>Proposed partitions</h3><button className="secondary" type="button" onClick={() => void inspectDataset()} disabled={busy === "dataset"}><Eye size={16}/> Update split preview</button></div><SplitSummary preview={preview} task={draft.task}/></div>}
    </section>
    <FeaturePreparation/>
    <section className="dataFinalize"><div><strong>Data checkpoint</strong><span>{preview ? "Validate the evaluation and feature preparation choices before selecting a model." : "Inspect the current dataset before continuing."}</span></div><ValidationSummary stage="data"/><button className="primary" type="button" onClick={() => void continueFlow()} disabled={!preview || busy === "data" || targetMissing > 0}>Validate and continue <ArrowRight size={16}/></button></section>
  </>;
}

function Metric({ label, value }: { label: string; value: number | string }) { return <div><span>{label}</span><strong>{value}</strong></div>; }

function TargetSummary({ preview, task }: { preview: DatasetPreview; task: Task }) {
  const target = preview.target_summary;
  if (!target) return <div className="emptyCompact">Select a valid target column and refresh the inspection.</div>;
  if (task === "regression" && target.statistics) return <div className="statGrid">{["count", "mean", "std", "min", "median", "max"].map((name) => <Metric key={name} label={name} value={format(target.statistics?.[name])}/>)}</div>;
  const distribution = target.distribution ?? [];
  return <div className="distributionList">{distribution.map((row) => <div key={String(row.label)}><div><strong>{String(row.label)}</strong><span>{row.count} rows · {row.percentage}%</span></div><div className="distributionTrack"><span style={{ width: `${row.percentage}%` }}/></div></div>)}</div>;
}

function SplitSummary({ preview, task }: { preview: DatasetPreview; task: Task }) {
  const summary = preview.split_summary;
  if (!summary?.available) return <div className="blockingNotice"><AlertTriangle size={17}/>{summary?.error ?? "Evaluation settings changed. Update the split preview to see the current partitions."}</div>;
  return <><p className="splitMethod">{summary.method === "chronological" ? `Chronological · ${summary.time_column ?? "existing row order"} · gap ${summary.gap} rows` : `Seed ${summary.random_state ?? "none"} · ${summary.stratified ? "class-stratified" : "unstratified"}`}</p><div className="splitPartitions">{Object.entries(summary.partitions ?? {}).map(([name, partition]) => <div className="splitPartition" key={name}><header><strong>{name}</strong><span>{partition.rows} rows · {partition.percentage}%</span></header>{task === "regression" ? <span>Mean target {format(partition.target?.statistics?.mean)}</span> : <div className="classCounts">{partition.target?.distribution?.map((item) => <span key={String(item.label)}>{String(item.label)}: <b>{item.count}</b></span>)}</div>}</div>)}</div></>;
}

function RowPreview({ preview }: { preview: DatasetPreview }) {
  const names = preview.columns.map((column) => column.name);
  return <div className="tableWrap"><table><thead><tr><th>Row</th>{names.map((name) => <th key={name}>{name}</th>)}</tr></thead><tbody>{preview.preview.map((row, index) => <tr key={index}><td>{index + 1}</td>{names.map((name) => <td key={name}>{formatCell(row[name])}</td>)}</tr>)}</tbody></table></div>;
}

function ColumnProfile({ preview, target }: { preview: DatasetPreview; target: string }) {
  return <div className="tableWrap"><table><thead><tr><th>Column</th><th>Role</th><th>Type</th><th>Missing</th><th>Unique</th></tr></thead><tbody>{preview.columns.map((column) => <tr key={column.name}><td><strong>{column.name}</strong></td><td>{column.name === target ? "Target" : "Feature"}</td><td>{column.dtype}</td><td>{column.missing}</td><td>{column.unique}</td></tr>)}</tbody></table></div>;
}

function NumericStatistics({ preview }: { preview: DatasetPreview }) {
  return <div className="tableWrap"><table><thead><tr><th>Column</th><th>Count</th><th>Missing</th><th>Mean</th><th>Std</th><th>Min</th><th>25%</th><th>Median</th><th>75%</th><th>Max</th><th>IQR outliers</th></tr></thead><tbody>{preview.numeric_summary.map((row) => <tr key={row.column}><td><strong>{row.column}</strong></td><td>{row.count}</td><td>{row.missing}</td><td>{format(row.mean)}</td><td>{format(row.std)}</td><td>{format(row.min)}</td><td>{format(row.q25)}</td><td>{format(row.median)}</td><td>{format(row.q75)}</td><td>{format(row.max)}</td><td>{row.outliers_iqr ?? 0}</td></tr>)}</tbody></table></div>;
}

function buildWarnings(preview: DatasetPreview, task: Task, target: string) {
  const warnings: string[] = [];
  const targetInfo = preview.columns.find((column) => column.name === target);
  if (!targetInfo) warnings.push("Select a target column that exists in this dataset.");
  else {
    if (targetInfo.missing) warnings.push(`The target contains ${targetInfo.missing} missing values and cannot be used yet.`);
    if (task === "binary_classification" && targetInfo.unique !== 2) warnings.push(`Binary classification needs exactly 2 target classes; this target has ${targetInfo.unique}.`);
    if (task === "multiclass_classification" && targetInfo.unique < 3) warnings.push("Multiclass classification needs at least 3 target classes.");
    if (task === "regression" && !/int|float|number/i.test(targetInfo.dtype)) warnings.push("Regression normally requires a numeric target.");
  }
  if (preview.row_count < 50) warnings.push("This is a small dataset; validation estimates may vary substantially between splits.");
  if (preview.summary.duplicate_rows) warnings.push(`${preview.summary.duplicate_rows} duplicate rows were detected; review whether they represent repeated observations.`);
  return warnings;
}

function format(value: unknown) { const number = Number(value); return Number.isFinite(number) ? number.toLocaleString(undefined, { maximumFractionDigits: 3 }) : "—"; }
function formatCell(value: unknown) { if (value === null || value === undefined || (typeof value === "number" && Number.isNaN(value))) return "Missing"; return String(value); }
