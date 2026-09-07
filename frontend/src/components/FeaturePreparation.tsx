import { AlertTriangle, WandSparkles } from "lucide-react";
import type { ApiSchema } from "../types";
import { useWorkflow } from "../workflow";

export function FeaturePreparation() {
  const { draft, preview, updateDraft } = useWorkflow();
  const settings = draft.preprocessing;
  const change = (patch: Partial<typeof settings>) => updateDraft({ preprocessing: { ...settings, ...patch } });
  const timeColumn = draft.split.method === "chronological" ? draft.split.time_column : null;
  const columns = preview?.columns.filter((column) => column.name !== draft.dataset.target_column) ?? [];
  const featureMissing = columns.filter((column) => !settings.ignored_columns.includes(column.name) && column.name !== timeColumn).reduce((sum, column) => sum + column.missing, 0);
  const featureRule = (name: string, patch: Partial<ApiSchema["FeatureSpec"]>) => change({ features: { ...settings.features, [name]: { ...(settings.features[name] ?? { kind: "auto", categories: [] }), ...patch } } });
  return <section className="surface featurePreparation" aria-labelledby="preparation-title">
    <div className="sectionIntro"><div><h2 id="preparation-title"><WandSparkles size={19}/> 3. Feature engineering</h2><p>Choose which inputs to use and how to prepare them. Imputation, clipping, scaling and learned categories are fitted on training rows inside each fold.</p></div></div>
    {preview && <div className="preparationSummary">{featureMissing ? `${featureMissing} missing feature cells will use the rules below.` : <><strong>No feature imputation needed</strong><span>The rules below also cover missing values in future predictions.</span></>}</div>}
    <div className="preparationGrid">
      <div className="preparationGroup"><h3>Missing values</h3>
        <label>Numeric missing values<select aria-label="Numeric missing values" value={settings.numeric_imputation} onChange={(event) => change({ numeric_imputation: event.target.value as typeof settings.numeric_imputation })}><option value="median">Median</option><option value="mean">Mean</option><option value="most_frequent">Most frequent</option><option value="constant">Constant value</option></select></label>
        {settings.numeric_imputation === "constant" && <label>Numeric fill value<input type="number" value={settings.numeric_fill_value} onChange={(event) => change({ numeric_fill_value: Number(event.target.value) })}/></label>}
        <label>Categorical missing values<select aria-label="Categorical missing values" value={settings.categorical_imputation} onChange={(event) => change({ categorical_imputation: event.target.value as typeof settings.categorical_imputation })}><option value="most_frequent">Most frequent</option><option value="constant">Constant label</option></select></label>
        {settings.categorical_imputation === "constant" && <label>Categorical fill label<input value={settings.categorical_fill_value} onChange={(event) => change({ categorical_fill_value: event.target.value })}/></label>}
        <label className="checkLabel"><input type="checkbox" checked={settings.add_missing_indicators} onChange={(event) => change({ add_missing_indicators: event.target.checked })}/> Add missingness indicators</label>
      </div>
      <div className="preparationGroup"><h3>Outliers and scale</h3>
        <label>Numeric outliers<select aria-label="Numeric outliers" value={settings.outlier_method} onChange={(event) => change({ outlier_method: event.target.value as typeof settings.outlier_method })}><option value="none">Keep original values</option><option value="iqr">Clip using the interquartile range (IQR)</option><option value="quantile">Clip to percentile bounds</option></select></label>
        {settings.outlier_method === "iqr" && <label>IQR multiplier<input type="number" min="0.1" step="0.1" value={settings.outlier_iqr_multiplier} onChange={(event) => change({ outlier_iqr_multiplier: Number(event.target.value) })}/><small>Bounds: Q1 − multiplier × IQR to Q3 + multiplier × IQR.</small></label>}
        {settings.outlier_method === "quantile" && <div className="formGrid"><label>Lower percentile<input type="number" min="0" max="49" value={settings.outlier_lower_quantile * 100} onChange={(event) => change({ outlier_lower_quantile: Number(event.target.value) / 100 })}/></label><label>Upper percentile<input type="number" min="51" max="100" value={settings.outlier_upper_quantile * 100} onChange={(event) => change({ outlier_upper_quantile: Number(event.target.value) / 100 })}/></label></div>}
        <p className="fieldHint">Clipping caps extreme feature values without removing rows or changing the target. Bounds are learned from training data only.</p>
        <label className="checkLabel"><input type="checkbox" checked={settings.scale_numeric} onChange={(event) => change({ scale_numeric: event.target.checked })}/> Standardize numeric features</label>
      </div>
    </div>
    <div className="sectionIntro featureTypesHeading"><div><h3>Feature types and encoding</h3><p>Nominal categories use one-hot encoding. Ordinal categories use the order you specify. Numeric category codes can be explicitly marked nominal or ordinal.</p></div><label className="checkLabel"><input type="checkbox" checked={settings.encode_categorical} onChange={(event) => change({ encode_categorical: event.target.checked })}/> Encode categorical features</label></div>
    {!settings.encode_categorical && <p className="blockingNotice"><AlertTriangle size={16}/> Categorical features are excluded while encoding is disabled.</p>}
    {!preview ? <div className="emptyCompact">Inspect a dataset to configure individual features.</div> : <div className="tableWrap"><table className="featureRules"><thead><tr><th>Use</th><th>Feature</th><th>Detected type / examples</th><th>Treat as</th><th>Encoding / order</th></tr></thead><tbody>{columns.map((column) => {
      const rule = settings.features[column.name] ?? { kind: "auto", categories: [] };
      const isTime = column.name === timeColumn;
      const ignored = settings.ignored_columns.includes(column.name);
      const numeric = /int|float|number|bool/i.test(column.dtype);
      return <tr key={column.name} className={ignored || isTime ? "excludedFeature" : ""}><td><input aria-label={`Use ${column.name}`} type="checkbox" checked={!ignored && !isTime} disabled={isTime} onChange={() => change({ ignored_columns: ignored ? settings.ignored_columns.filter((name) => name !== column.name) : [...settings.ignored_columns, column.name] })}/></td><td><strong>{column.name}</strong>{isTime && <small>Time ordering only</small>}</td><td><small>{column.dtype}</small><span className="featureExamples">{(column.examples ?? []).slice(0, 4).map(String).join(" · ")}</span></td><td><select aria-label={`Feature type for ${column.name}`} disabled={ignored || isTime} value={rule.kind} onChange={(event) => featureRule(column.name, { kind: event.target.value as ApiSchema["FeatureSpec"]["kind"] })}><option value="auto">Automatic · {numeric ? "numeric" : "nominal"}</option><option value="numeric">Numeric</option><option value="nominal">Nominal category</option><option value="ordinal">Ordinal category</option></select></td><td>{rule.kind === "ordinal" ? <label>Lowest → highest<textarea aria-label={`Ordered categories for ${column.name}`} disabled={ignored || isTime} rows={3} placeholder={"low\nmedium\nhigh"} value={(rule.categories ?? []).join("\n")} onChange={(event) => featureRule(column.name, { categories: event.target.value.split("\n") })}/><small>One category per line. Unlisted values receive −1.</small></label> : <span className="fieldHint">{rule.kind === "nominal" || rule.kind === "auto" && !numeric ? "One-hot · one column per category" : "Numeric values"}</span>}</td></tr>;
    })}</tbody></table></div>}
    {!!preview?.target_summary?.missing && <p className="blockingNotice"><AlertTriangle size={16}/> Missing target values cannot be imputed automatically. Remove or label those rows before training.</p>}
  </section>;
}
