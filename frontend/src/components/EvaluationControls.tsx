import { useWorkflow } from "../workflow";

export function EvaluationControls() {
  const { draft, preview, updateDraft } = useWorkflow();
  const chronological = draft.split.method === "chronological";
  const cv = draft.evaluation.protocol === "cross_validation";
  const changeSplit = (patch: Partial<typeof draft.split>) => updateDraft({ split: { ...draft.split, ...patch } });
  const changeEvaluation = (patch: Partial<typeof draft.evaluation>) => updateDraft({ evaluation: { ...draft.evaluation, ...patch } });
  return <div className="evaluationControls">
    <div className="sectionIntro"><div><h3>Choose how models are evaluated</h3><span>Keep the final test partition untouched while comparing candidate settings.</span></div></div>
    <div className="formGrid">
      <label>Evaluation method<select aria-label="Evaluation method" value={draft.evaluation.protocol} onChange={(event) => changeEvaluation({ protocol: event.target.value as "holdout" | "cross_validation" })}>
        <option value="cross_validation">Cross-validation</option><option value="holdout">Train / validation / test</option>
      </select><small>{cv ? "Five folds by default; each candidate is fitted once per fold." : "One training split and one validation split per candidate."}</small></label>
      <label>Row order<select aria-label="Row order" value={draft.split.method} onChange={(event) => {
        const ordered = event.target.value === "chronological";
        updateDraft({ split: { ...draft.split, method: ordered ? "chronological" : "random", stratify: !ordered && draft.task !== "regression", time_column: ordered ? draft.split.time_column : null, gap: ordered ? draft.split.gap : 0 }, evaluation: { ...draft.evaluation, shuffle: !ordered } });
      }}><option value="random">Random split · independent observations</option><option value="chronological">Chronological · time series / forecasting</option></select></label>
      {cv ? <label>Folds<input type="number" min="2" max="20" value={draft.evaluation.folds} onChange={(event) => changeEvaluation({ folds: Number(event.target.value) })}/></label>
        : <label>Validation share<input aria-label="Validation share" type="number" min="0.05" max="0.8" step="0.05" value={draft.split.validation_size} onChange={(event) => changeSplit({ validation_size: Number(event.target.value) })}/><small>Share of the full dataset used to select the winner.</small></label>}
      <label>Untouched test share<input aria-label="Untouched test share" type="number" min="0" max="0.8" step="0.05" value={draft.split.test_size} onChange={(event) => changeSplit({ test_size: Number(event.target.value) })}/><small>Use 0 only when a separate test report is not needed.</small></label>
      {chronological ? <>
        <label>Time column<select aria-label="Time column" value={draft.split.time_column ?? ""} onChange={(event) => changeSplit({ time_column: event.target.value || null })}><option value="">Use existing row order · oldest first</option>{preview?.columns.filter((column) => column.name !== draft.dataset.target_column).map((column) => <option key={column.name} value={column.name}>{column.name}</option>)}</select><small>A selected time column sorts the rows and is excluded from model features.</small></label>
        <label>Gap between partitions (rows)<input aria-label="Gap between partitions (rows)" type="number" min="0" step="1" value={draft.split.gap} onChange={(event) => changeSplit({ gap: Number(event.target.value) })}/><small>Leave a buffer before validation and test windows.</small></label>
      </> : <>
        <label>Random seed<input type="number" min="0" value={draft.split.random_state} onChange={(event) => changeSplit({ random_state: Number(event.target.value) })}/></label>
        {draft.task !== "regression" && <label className="checkLabel"><input type="checkbox" checked={draft.split.stratify} onChange={(event) => changeSplit({ stratify: event.target.checked })}/> Preserve class proportions</label>}
        {cv && <label className="checkLabel"><input type="checkbox" checked={draft.evaluation.shuffle} onChange={(event) => changeEvaluation({ shuffle: event.target.checked })}/> Shuffle folds with the saved seed</label>}
      </>}
    </div>
    <div className="evaluationExplanation">{chronological ? <><strong>{cv ? "Expanding-window cross-validation" : "Past → validation → future test"}</strong><p>{cv ? "Each fold trains on earlier observations and validates on the next window. Training history grows with each fold." : "The earliest rows train the model, later rows select its settings, and the latest rows form the untouched test."} Rows are never shuffled. For forecasting, supply a target aligned to the future period you want to predict.</p></> : <><strong>{cv ? `${draft.evaluation.folds}-fold cross-validation + final test` : `${Math.max(0, (1 - draft.split.validation_size - draft.split.test_size) * 100).toFixed(0)}% training / ${(draft.split.validation_size * 100).toFixed(0)}% validation / ${(draft.split.test_size * 100).toFixed(0)}% test`}</strong><p>{cv ? "Fold scores are averaged to select settings; the test set stays outside every fold." : "A single holdout uses fewer fits than cross-validation and keeps train, validation, and test rows separate."}</p></>}</div>
  </div>;
}
