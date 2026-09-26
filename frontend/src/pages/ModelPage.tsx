import { ArrowLeft, ArrowRight, Check, CircleSlash2, Gauge, Info } from "lucide-react";
import { useState } from "react";
import { useNavigate } from "react-router-dom";
import { PageHeading, ValidationSummary } from "../components/Layout";
import { HelpTip } from "../components/Help";
import type { SearchParam } from "../types";
import { useWorkflow } from "../workflow";

export function ModelPage() {
  const navigate = useNavigate();
  const { draft, metadata, busy, updateDraft, setEstimator, validateStage } = useWorkflow();
  const [advanced, setAdvanced] = useState(false);
  if (!metadata) return <div className="emptyState">Loading model registry…</div>;
  const compatible = Object.entries(metadata.estimators).filter(([, item]) => item.tasks.includes(draft.task));
  const groups = [...new Set(compatible.map(([, item]) => item.group))];
  const current = metadata.estimators[draft.estimator];
  async function continueFlow() { if (await validateStage("model")) navigate("/search"); }
  return <>
    <PageHeading eyebrow="Stage 2 of 6" title="Model and parameters" description={`Choose a validated ${metadata.tasks[draft.task].label.toLowerCase()} recipe. Unavailable optional engines are identified before a run can start.`}/>
    <div className="modelLayout">
      <section className="surface modelCatalog">
        {groups.map((group) => <div className="modelGroup" key={group}><h2>{group}</h2><div className="modelCards">{compatible.filter(([, item]) => item.group === group).map(([name, item]) => <button type="button" key={name} disabled={!item.dependency.installed} className={`modelCard ${draft.estimator === name ? "selected" : ""}`} onClick={() => setEstimator(name)}>
          <div><strong>{item.label}</strong>{draft.estimator === name && <Check size={17}/>}</div><p>{item.description}</p><span className={item.dependency.installed ? "available" : "unavailable"}>{item.dependency.installed ? "Available" : `Unavailable · ${item.dependency.required}`}</span>
        </button>)}</div></div>)}
      </section>
      <aside className="surface parameterPanel">
        <div className="modelTitle"><div><span>{current.group}</span><h2>{current.label}</h2></div><Gauge size={22}/></div>
        {!current.dependency.installed && <div className="dependencyNotice"><CircleSlash2 size={18}/><div><strong>Engine is not installed</strong><code>{current.dependency.install}</code></div></div>}
        <div className="capabilityGrid"><Capability label="Probability output" enabled={current.capabilities.probability_output}/><Capability label="Feature importance" enabled={Boolean(current.capabilities.feature_importance)} detail={current.capabilities.feature_importance ?? "Permutation fallback"}/><Capability label="Scaling advised" enabled={current.capabilities.scaling_recommended}/><Capability label="Epoch monitoring" enabled={current.capabilities.epoch_progress}/></div>
        <h3>Fixed training parameters <HelpTip label="Fixed parameters">These settings remain identical for every model fitted in this run.</HelpTip></h3><p className="sectionNote">Applied to every candidate. These values are not optimized.</p>
        <div className="parameterRows">{Object.entries(draft.fixed_params).map(([name, value]) => <label key={name}><span>{name}</span><input value={String(value)} onChange={(event) => updateDraft({ fixed_params: { ...draft.fixed_params, [name]: parseValue(event.target.value, value) } })}/></label>)}{Object.keys(draft.fixed_params).length === 0 && <div className="emptyCompact">This recipe has no required fixed parameters.</div>}</div>
        <h3>Tunable search parameters <HelpTip label="Tunable parameters">The search engine proposes a value from each range for every candidate model.</HelpTip></h3><p className="sectionNote">The search engine will propose values inside these domains.</p>
        <div className="parameterRows">{Object.entries(draft.search_space).map(([name, spec]) => <SearchParameter key={name} name={name} spec={spec} onChange={(next) => updateDraft({ search_space: { ...draft.search_space, [name]: next } })}/>)}</div>
        <button className="textButton" type="button" onClick={() => setAdvanced(!advanced)}><Info size={15}/> {advanced ? "Hide" : "Show"} advanced specification</button>
        {advanced && <textarea className="codeEditor" rows={12} key={`${draft.estimator}-${draft.task}`} defaultValue={JSON.stringify({ fixed_params: draft.fixed_params, search_space: draft.search_space }, null, 2)} onBlur={(event) => { try { const parsed = JSON.parse(event.target.value); updateDraft({ fixed_params: parsed.fixed_params ?? {}, search_space: parsed.search_space ?? {} }); } catch { /* Preserve the last valid structured state. */ } }}/>}
        <ValidationSummary stage="model" />
        <div className="footerActions"><button className="secondary" onClick={() => navigate("/data")}><ArrowLeft size={16}/> Back</button><button className="primary" onClick={() => void continueFlow()} disabled={busy === "model" || !current.dependency.installed}>Validate and continue <ArrowRight size={16}/></button></div>
      </aside>
    </div>
  </>;
}

function Capability({ label, enabled, detail }: { label: string; enabled: boolean; detail?: string }) { return <div className={enabled ? "capability yes" : "capability"}><span>{label}</span><strong>{detail ?? (enabled ? "Supported" : "Not available")}</strong></div>; }
function parseValue(value: string, previous: unknown) { if (typeof previous === "number") return Number(value); if (typeof previous === "boolean") return value === "true"; return value; }

function SearchParameter({ name, spec, onChange }: { name: string; spec: SearchParam; onChange: (next: SearchParam) => void }) {
  if (spec.type === "choice") return <label className="parameterChoice"><span>{name}<small>Categories</small></span><input value={spec.values.join(", ")} onChange={(event) => onChange({ ...spec, values: event.target.value.split(",").map((value) => value.trim()).filter(Boolean) })}/></label>;
  return <div className="rangeEditor"><span>{name}<small>{spec.type === "log_float" ? "Logarithmic range" : spec.type === "int" ? "Integer range" : "Numeric range"}</small></span><label>Min<input type="number" value={spec.low} onChange={(event) => onChange({ ...spec, low: Number(event.target.value) })}/></label><label>Max<input type="number" value={spec.high} onChange={(event) => onChange({ ...spec, high: Number(event.target.value) })}/></label>{spec.type !== "int" && <label>Decimals<input type="number" min="0" max="8" value={spec.precision} onChange={(event) => onChange({ ...spec, precision: Number(event.target.value) })}/></label>}</div>;
}
