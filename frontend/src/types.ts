import type { components } from "./api-schema";
export type ApiSchema = components["schemas"];
export type JsonValue = ApiSchema["JsonValue"];
export type Task = ApiSchema["ExperimentSpec"]["task"];
export type MetricId = ApiSchema["ExperimentSpec"]["metric"];
export type Strategy = ApiSchema["ExperimentSpec"]["strategy"];
export type SearchParam = ApiSchema["ChoiceSearchParam"] | ApiSchema["IntSearchParam"] | ApiSchema["FloatSearchParam"];
export type RunSnapshot = ApiSchema["RunResponse"];
export type RunEvent = ApiSchema["RunEventResponse"];
export type ValidationResult = ApiSchema["ValidationResponse"];
export type ExperimentSummary = ApiSchema["ExperimentResponse"];
export type ExperimentDetail = ExperimentSummary & { runs: RunSnapshot[] };
export type PredictionPreview = ApiSchema["PredictionResponse"];
export type EstimatorMeta = ApiSchema["EstimatorResponse"];
export type Metadata = ApiSchema["MetadataResponse"];

export type DatasetPreview = ApiSchema["DatasetPreviewResponse"];
export type ExampleDataset = ApiSchema["ExampleDatasetResponse"];
export type SavedDataset = ApiSchema["SavedDatasetResponse"];
export type AnalysisSplit = ApiSchema["AnalysisSplitResponse"];
export type RunAnalysis = ApiSchema["AnalysisResponse"];

// Forms materialize optional request defaults; all wire field types come from OpenAPI.
export type WorkflowDraft = Omit<Required<ApiSchema["ExperimentSpec"]>, "dataset" | "split" | "preprocessing" | "pso" | "evaluation" | "runtime" | "search_space"> & {
  dataset: Required<ApiSchema["DatasetSpec"]>;
  split: Required<ApiSchema["SplitSpec"]> & { random_state: number };
  preprocessing: Required<ApiSchema["PreprocessingSpec"]>;
  pso: ApiSchema["PsoSpec"];
  evaluation: Required<Omit<ApiSchema["EvaluationSpec"], "stratify" | "test_size" | "random_state">>;
  runtime: Required<ApiSchema["RuntimeSpec"]>;
  search_space: Record<string, SearchParam>;
};
