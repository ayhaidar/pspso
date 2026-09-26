"""Versioned API v1 contracts shared by FastAPI, the CLI, and workers."""

from __future__ import annotations

from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, JsonValue, model_validator

TaskId = Literal["regression", "binary_classification", "multiclass_classification"]
StrategyId = Literal["pso", "grid", "random"]
MetricId = Literal["rmse", "mae", "r2", "accuracy", "roc_auc", "pr_auc", "log_loss", "f1_macro"]
RunStatus = Literal["queued", "running", "completed", "failed", "cancelled", "interrupted"]


class ApiModel(BaseModel):
    """Base contract that rejects unknown fields in submitted specifications."""

    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)


class CsvPreviewRequest(ApiModel):
    csv_text: str


class DatasetSpec(ApiModel):
    source: Literal["example", "csv", "stored"] = "example"
    name: str | None = "breast_cancer"
    csv_text: str | None = None
    dataset_id: str | None = None
    target_column: str = "target"


class SplitSpec(ApiModel):
    validation_size: float = Field(default=0.2, gt=0, lt=1)
    test_size: float = Field(default=0.0, ge=0, lt=1)
    stratify: bool = True
    random_state: int | None = Field(default=42, ge=0, le=4294967295)
    method: Literal["random", "chronological"] = "random"
    time_column: str | None = None
    gap: int = Field(default=0, ge=0)


class FeatureSpec(ApiModel):
    kind: Literal["auto", "numeric", "nominal", "ordinal"] = "auto"
    categories: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_order(self) -> FeatureSpec:
        if self.kind == "ordinal" and not self.categories:
            raise ValueError("Ordinal features need categories ordered from lowest to highest.")
        if len(set(self.categories)) != len(self.categories):
            raise ValueError("Ordinal categories must be unique.")
        return self


class PreprocessingSpec(ApiModel):
    scale_numeric: bool = True
    encode_categorical: bool = True
    ignored_columns: list[str] = Field(default_factory=list)
    numeric_imputation: Literal["median", "mean", "most_frequent", "constant"] = "median"
    categorical_imputation: Literal["most_frequent", "constant"] = "most_frequent"
    numeric_fill_value: float = 0.0
    categorical_fill_value: str = "Missing"
    add_missing_indicators: bool = False
    features: dict[str, FeatureSpec] = Field(default_factory=dict)
    outlier_method: Literal["none", "iqr", "quantile"] = "none"
    outlier_iqr_multiplier: float = Field(default=1.5, gt=0)
    outlier_lower_quantile: float = Field(default=0.01, ge=0, lt=0.5)
    outlier_upper_quantile: float = Field(default=0.99, gt=0.5, le=1)


class DatasetInspectRequest(DatasetSpec):
    task: TaskId = "binary_classification"
    split: SplitSpec = Field(default_factory=SplitSpec)
    evaluation: EvaluationSpec | None = None


class PsoSpec(ApiModel):
    particles: int = Field(default=5, ge=1, le=512)
    iterations: int = Field(default=10, ge=1, le=10000)
    c1: float = 1.49618
    c2: float = 1.49618
    w: float = 0.7298
    topology: Literal["global", "local"] = "global"


class RuntimeSpec(ApiModel):
    max_trials: int | None = Field(default=None, ge=1)
    timeout_seconds: float | None = Field(default=None, gt=0)
    early_stopping_rounds: int | None = Field(default=None, ge=1)
    trial_workers: int = Field(default=1, ge=1, le=64)
    verbose: int = Field(default=0, ge=0, le=3)


class EvaluationSpec(ApiModel):
    """Scientifically reproducible candidate evaluation settings."""

    protocol: Literal["holdout", "cross_validation"] = "holdout"
    folds: int = Field(default=5, ge=2, le=20)
    shuffle: bool = True
    stratify: bool | None = None
    test_size: float | None = Field(default=None, ge=0, lt=1)
    random_state: int | None = Field(default=None, ge=0, le=4294967295)
    positive_label: str | int | float | bool | None = None
    decision_threshold: float = Field(default=0.5, ge=0, le=1)
    refit_best: bool = True


class ChoiceSearchParam(ApiModel):
    type: Literal["choice"]
    values: list[str | int | float | bool] = Field(min_length=1)


class IntSearchParam(ApiModel):
    type: Literal["int"]
    low: int
    high: int


class FloatSearchParam(ApiModel):
    type: Literal["float", "log_float"]
    low: float
    high: float
    precision: int = Field(default=4, ge=0, le=12)


SearchParameter = Annotated[
    ChoiceSearchParam | IntSearchParam | FloatSearchParam,
    Field(discriminator="type"),
]


class ExperimentSpec(ApiModel):
    """Complete, reproducible specification accepted by API v1 and the CLI."""

    schema_version: Literal[1] = 1
    experiment_id: str | None = None
    dataset: DatasetSpec = Field(default_factory=DatasetSpec)
    task: TaskId = "binary_classification"
    metric: MetricId = "roc_auc"
    estimator: str = "svm"
    fixed_params: dict[str, JsonValue] = Field(default_factory=dict)
    search_space: dict[str, SearchParameter] | None = None
    strategy: StrategyId = "random"
    split: SplitSpec = Field(default_factory=SplitSpec)
    preprocessing: PreprocessingSpec = Field(default_factory=PreprocessingSpec)
    pso: PsoSpec = Field(default_factory=PsoSpec)
    evaluation: EvaluationSpec = Field(default_factory=EvaluationSpec)
    runtime: RuntimeSpec = Field(default_factory=RuntimeSpec)

    @model_validator(mode="after")
    def synchronize_evaluation(self) -> ExperimentSpec:
        """Evaluation settings override legacy split fields when explicitly supplied."""
        for name in ("stratify", "test_size", "random_state"):
            value = getattr(self.evaluation, name)
            if value is None:
                setattr(self.evaluation, name, getattr(self.split, name))
            else:
                setattr(self.split, name, value)
        if self.split.method == "chronological":
            self.split.stratify = False
            self.evaluation.stratify = False
            self.evaluation.shuffle = False
        return self


class ExperimentCreateRequest(ApiModel):
    name: str
    description: str = ""
    tags: list[str] = Field(default_factory=list)


class TournamentRequest(ApiModel):
    runs: list[ExperimentSpec] = Field(min_length=2)


class DatasetCreateRequest(ApiModel):
    name: str
    csv_text: str


class WorkflowValidationRequest(ApiModel):
    stage: Literal["data", "model", "search", "full"] = "full"
    request: ExperimentSpec


class ResultLayoutRequest(ApiModel):
    tools: list[str] = Field(default_factory=list)


class ValidationResponse(BaseModel):
    valid: bool
    errors: dict[str, list[str]]
    stage: str | None = None


class ResponseModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class ProgressPayload(BaseModel):
    """Known progress fields are checked; recipes can add JSON-compatible details."""

    model_config = ConfigDict(extra="allow")
    __pydantic_extra__: dict[str, JsonValue] = Field(init=False)
    trial_id: int | None = None
    fit_id: int | None = None
    iteration: int | None = None
    fold: int | None = None
    total_folds: int | None = None
    worker_slot: int | None = None
    particle_index: int | None = None
    strategy_slot: int | None = None
    phase: Literal["candidate", "final_refit"] | None = None
    completed_fits: int | None = None
    failed_fits: int | None = None
    total_fits: int | None = None
    planned_fits: int | None = None
    planned_candidate_fits: int | None = None
    planned_trials: int | None = None
    metric: float | MetricId | None = None
    cost: float | None = None
    best_metric: float | None = None
    best_cost: float | None = None
    params: dict[str, JsonValue] | None = None
    best_params: dict[str, JsonValue] | None = None
    error: str | None = None
    reason: str | None = None
    message: str | None = None


class RunEventResponse(ResponseModel):
    sequence_number: int = Field(ge=1)
    type: str
    timestamp: str
    payload: ProgressPayload


class AttemptResponse(ResponseModel):
    attempt_id: str
    run_id: str
    parent_run_id: str | None = None
    status: RunStatus
    worker_pid: int | None = None
    worker_created_at: float | None = None
    started_at: str | None = None
    finished_at: str | None = None
    created_at: str
    service_id: str | None = None
    heartbeat_at: str | None = None
    timeout_seconds: float | None = None
    termination_requested: bool = False
    termination_reason: Literal["timeout", "cancelled", "interrupted"] | None = None


class BestResponse(ResponseModel):
    best_metric: float | None = None
    best_cost: float | None = None
    best_params: dict[str, JsonValue] | None = None


class TrialResponse(ResponseModel):
    trial_id: int
    params: dict[str, JsonValue]
    cost: float | None
    metric: float | None
    train_metric: float | None
    status: Literal["completed", "failed"]
    duration: float
    error: str | None = None
    iteration: int | None = None
    particle_index: int | None = None
    strategy_slot: int | None = None
    worker_slot: int | None = None
    validation_metrics: dict[str, float | None] = Field(default_factory=dict)
    train_metrics: dict[str, float | None] = Field(default_factory=dict)
    metric_std: float | None = None
    fold_metrics: list[dict[str, JsonValue]] = Field(default_factory=list)
    failure_category: str | None = None


class OptimizationResultResponse(BestResponse):
    duration: float
    trials: list[TrialResponse]
    strategy: StrategyId
    task: TaskId
    metric_name: MetricId
    failures: list[str] = Field(default_factory=list)
    best_position: list[float] | None = None
    optimizer_state: dict[str, JsonValue] = Field(default_factory=dict)
    run_id: str | None = None
    experiment_id: str | None = None
    artifacts: dict[str, str] = Field(default_factory=dict)


class PythonRunSpec(ResponseModel):
    schema_version: Literal[1]
    source: Literal["python"]
    run_name: str | None = None
    dataset: dict[str, JsonValue]
    task: TaskId
    metric: MetricId
    estimator: str
    fixed_params: dict[str, JsonValue]
    search_space: dict[str, JsonValue]
    strategy: StrategyId
    optimization: dict[str, JsonValue]


class ExperimentResponse(ResponseModel):
    experiment_id: str
    name: str
    description: str = ""
    tags: list[str] = Field(default_factory=list)
    run_count: int = 0
    is_ad_hoc: bool = False
    created_at: str
    latest_status: RunStatus | None = None
    latest_run_id: str | None = None
    latest_updated_at: str | None = None
    result_layout: list[str] = Field(default_factory=list)
    runs: list[RunResponse] = Field(default_factory=list)


class RunResponse(ResponseModel):
    run_id: str
    experiment_id: str
    experiment_name: str
    status: RunStatus
    event_count: int = 0
    experiment_description: str = ""
    best: BestResponse | None = None
    result: OptimizationResultResponse | None = None
    error: str | None = None
    created_at: str
    updated_at: str
    duration: float | None = None
    n_trials: int = 0
    n_failures: int = 0
    strategy: StrategyId | None = None
    task: TaskId | None = None
    metric: MetricId | None = None
    estimator: str | None = None
    request: ExperimentSpec | PythonRunSpec | None = None
    cancel_requested: bool = False
    artifacts: dict[str, str] = Field(default_factory=dict)
    parent_run_id: str | None = None
    attempts: list[AttemptResponse] = Field(default_factory=list)
    queue_position: int | None = None


class RunHistoryResponse(BaseModel):
    run_id: str
    events: list[RunEventResponse]


class PredictionRow(ResponseModel):
    row_index: str | int
    actual: JsonValue
    predicted: JsonValue
    residual: float | None = None
    score: float | None = None
    probabilities: dict[str, float] | None = None
    features: dict[str, JsonValue]


class PredictionResponse(ResponseModel):
    run_id: str
    split: Literal["validation", "train", "test"]
    task: TaskId
    metric: MetricId
    estimator: str
    row_count: int
    feature_columns: list[str]
    target_mapping: dict[int, JsonValue] | None = None
    preview_rows: list[PredictionRow]


class ArtifactResponse(BaseModel):
    experiment: ExperimentResponse
    run: RunResponse
    events: list[RunEventResponse]


class DependencyResponse(ResponseModel):
    required: str | None
    installed: bool
    install: str | None


class CapabilitiesResponse(ResponseModel):
    probability_output: bool
    feature_importance: str | None
    permutation_importance: bool
    scaling_recommended: bool
    epoch_progress: bool


class EstimatorDefaults(ResponseModel):
    fixed_params: dict[str, JsonValue]
    search_space: dict[str, SearchParameter]
    allowed_params: list[str]


class EstimatorResponse(ResponseModel):
    label: str
    tasks: list[TaskId]
    group: str
    optional_dependency: str | None
    install: str | None = None
    description: str
    dependency: DependencyResponse
    capabilities: CapabilitiesResponse
    defaults: dict[str, EstimatorDefaults]


class MetricResponse(ResponseModel):
    label: str
    description: str


class TaskResponse(MetricResponse):
    metrics: list[MetricId]


class MetadataResponse(ResponseModel):
    estimators: dict[str, EstimatorResponse]
    tasks: dict[TaskId, TaskResponse]
    metrics: dict[MetricId, MetricResponse]


class TournamentResponse(ResponseModel):
    experiment_id: str
    runs: list[RunResponse]


__all__ = [
    "CsvPreviewRequest",
    "DatasetCreateRequest",
    "DatasetInspectRequest",
    "DatasetSpec",
    "ArtifactResponse",
    "ExperimentCreateRequest",
    "EvaluationSpec",
    "ExperimentSpec",
    "ExperimentResponse",
    "MetadataResponse",
    "MetricId",
    "PreprocessingSpec",
    "PsoSpec",
    "ResultLayoutRequest",
    "PredictionResponse",
    "RunHistoryResponse",
    "RunResponse",
    "RunStatus",
    "SearchParameter",
    "RunEventResponse",
    "RuntimeSpec",
    "SplitSpec",
    "TaskId",
    "TournamentRequest",
    "ValidationResponse",
    "WorkflowValidationRequest",
]
