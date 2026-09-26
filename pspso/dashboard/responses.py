"""Structured dataset and diagnostic response contracts."""

from pydantic import Field, JsonValue

from .schemas import MetricId, ResponseModel, TaskId


class ColumnResponse(ResponseModel):
    name: str
    dtype: str
    missing: int
    unique: int
    examples: list[JsonValue] = Field(default_factory=list)


class DistributionRow(ResponseModel):
    label: str | int | float | bool
    count: int
    percentage: float


class TargetResponse(ColumnResponse):
    count: int
    distribution: list[DistributionRow] | None = None
    statistics: dict[str, JsonValue] | None = None


class PartitionResponse(ResponseModel):
    rows: int
    percentage: float
    target: TargetResponse


class SplitResponse(ResponseModel):
    available: bool
    error: str | None = None
    stratified: bool | None = None
    random_state: int | None = None
    method: str = "random"
    time_column: str | None = None
    gap: int = 0
    partitions: dict[str, PartitionResponse] = Field(default_factory=dict)


class TableSummaryResponse(ResponseModel):
    column_count: int
    numeric_columns: int
    categorical_columns: int
    total_missing: int
    duplicate_rows: int
    memory_bytes: int


class NumericSummaryResponse(ResponseModel):
    column: str | None
    count: int
    missing: int
    mean: float | None
    std: float | None
    min: float | None
    q25: float | None
    median: float | None
    q75: float | None
    max: float | None
    outliers_iqr: int = 0


class DatasetPreviewResponse(ResponseModel):
    columns: list[ColumnResponse]
    row_count: int
    preview: list[dict[str, JsonValue]]
    target_candidates: list[str]
    summary: TableSummaryResponse
    numeric_summary: list[NumericSummaryResponse]
    target_summary: TargetResponse | None = None
    split_summary: SplitResponse | None = None


class ExampleDatasetResponse(ResponseModel):
    name: str
    label: str
    task: TaskId
    default_metric: MetricId
    target_column: str
    target_description: str
    rows: int
    features: int
    source: str
    source_url: str
    license: str
    license_url: str
    description: str


class SavedDatasetResponse(ResponseModel):
    dataset_id: str
    name: str
    source_kind: str
    source_path: str | None
    fingerprint: str
    profile: DatasetPreviewResponse
    created_at: str


class ConfusionResponse(ResponseModel):
    labels: list[str]
    values: list[list[int]]
    true_negative: int | None = None
    false_positive: int | None = None
    false_negative: int | None = None
    true_positive: int | None = None


class RocResponse(ResponseModel):
    label: str | None = None
    fpr: list[float]
    tpr: list[float]
    thresholds: list[float | None] = Field(default_factory=list)
    auc: float


class PrecisionRecallResponse(ResponseModel):
    precision: list[float]
    recall: list[float]
    thresholds: list[float | None]
    auc: float


class RegressionDiagnostics(ResponseModel):
    actual: list[float]
    predicted: list[float]
    residuals: list[float]
    sampled: bool


class MulticlassRocResponse(ResponseModel):
    curves: list[RocResponse]
    average: str


class FoldResponse(ResponseModel):
    fold: int
    metric: float
    cost: float
    train_metric: float
    validation_metrics: dict[str, float | None]
    train_metrics: dict[str, float | None]


class CrossValidationResponse(ResponseModel):
    folds: list[FoldResponse]
    mean: float
    standard_deviation: float | None


class AnalysisSplitResponse(ResponseModel):
    selected_metric: MetricId | None = None
    value: float | None = None
    cost: float | None = None
    metrics: dict[str, float | None]
    confusion_matrix: ConfusionResponse | None = None
    roc_curve: RocResponse | None = None
    precision_recall_curve: PrecisionRecallResponse | None = None
    threshold_diagnostics: list[dict[str, float | None]] | None = None
    regression_diagnostics: RegressionDiagnostics | None = None
    multiclass_roc: MulticlassRocResponse | None = None
    per_class: list[dict[str, str | float | None]] | None = None
    cross_validation: CrossValidationResponse | None = None
    decision_threshold: float | None = None
    positive_label: str | None = None


class ImportanceItem(ResponseModel):
    feature: str
    importance: float
    std: float | None = None


class ImportanceResponse(ResponseModel):
    available: bool
    source: str | None = None
    reason: str | None = None
    split: str | None = None
    items: list[ImportanceItem]


class AnalysisResponse(ResponseModel):
    task: TaskId
    metric: MetricId
    dataset: DatasetPreviewResponse | None = None
    train: AnalysisSplitResponse
    validation: AnalysisSplitResponse
    test: AnalysisSplitResponse | None = None
    feature_importance: ImportanceResponse
