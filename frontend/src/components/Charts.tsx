import { BarChart, HeatmapChart, LineChart, ScatterChart } from "echarts/charts";
import {
  GridComponent,
  LegendComponent,
  TooltipComponent,
  VisualMapComponent,
} from "echarts/components";
import * as echarts from "echarts/core";
import { CanvasRenderer } from "echarts/renderers";
import ReactEChartsCore from "echarts-for-react/lib/core";
import { memo, useMemo } from "react";
import type { AnalysisSplit, RunEvent } from "../types";

echarts.use([
  BarChart,
  HeatmapChart,
  LineChart,
  ScatterChart,
  GridComponent,
  LegendComponent,
  TooltipComponent,
  VisualMapComponent,
  CanvasRenderer,
]);

const base = { animationDuration: 350, textStyle: { fontFamily: "Inter, Segoe UI, sans-serif", color: "#27343a" }, grid: { left: 48, right: 22, top: 30, bottom: 42 }, tooltip: { trigger: "axis" } };

export const ConvergenceChart = memo(function ConvergenceChart({ events }: { events: RunEvent[] }) {
  const option = useMemo(() => {
    const rows = events.filter((event) => event.type === "trial_completed");
    const planned = Number([...events].reverse().find((event) => event.type === "run_started")?.payload.planned_trials ?? 0);
    let best = Number.POSITIVE_INFINITY;
    const points = rows.map((event, index) => {
      const cost = Number(event.payload.cost);
      best = Math.min(best, cost);
      return {
        completed: index + 1,
        trialId: event.payload.trial_id ?? index + 1,
        candidateCost: cost,
        bestCost: best,
      };
    });
    return {
      ...base,
      tooltip: {
        trigger: "axis",
        formatter: (items: Array<{ dataIndex: number }>) => {
          const point = points[items[0]?.dataIndex ?? 0];
          if (!point) return "";
          return [
            `Candidate ${point.completed} completed`,
            `PSO trial ID: ${String(point.trialId)}`,
            `Candidate cost: ${point.candidateCost.toFixed(6)}`,
            `Best cost so far: ${point.bestCost.toFixed(6)}`,
          ].join("<br/>");
        },
      },
      xAxis: {
        type: "category",
        name: planned > 0 ? `Candidates completed (of ${planned})` : "Candidates completed",
        data: points.map((point) => point.completed),
      },
      yAxis: { type: "value", name: "Best cost so far", scale: true },
      series: [{
        name: "Best cost so far",
        type: "line",
        data: points.map((point) => point.bestCost),
        step: "end",
        smooth: false,
        symbolSize: 7,
        lineStyle: { color: "#087f8c", width: 3 },
        itemStyle: { color: "#087f8c" },
        areaStyle: { color: "rgba(8,127,140,.09)" },
      }],
    };
  }, [events]);
  return <Chart option={option} />;
}, sameConvergenceEvents);

export function RocChart({ split }: { split: AnalysisSplit }) {
  const curves = split.multiclass_roc?.curves ?? (split.roc_curve ? [{ label: `ROC AUC ${split.roc_curve.auc.toFixed(3)}`, ...split.roc_curve }] : []);
  return <Chart option={{ ...base, legend: { bottom: 0 }, xAxis: { type: "value", name: "False positive rate", min: 0, max: 1 }, yAxis: { type: "value", name: "True positive rate", min: 0, max: 1 }, series: [{ type: "line", data: [[0, 0], [1, 1]], symbol: "none", lineStyle: { color: "#aab6b8", type: "dashed" } }, ...curves.map((curve) => ({ name: curve.label, type: "line", showSymbol: false, data: curve.fpr.map((x, index) => [x, curve.tpr[index]]) }))] }} />;
}

export function PrecisionRecallChart({ split }: { split: AnalysisSplit }) {
  const curve = split.precision_recall_curve;
  return <Chart option={{ ...base, xAxis: { type: "value", name: "Recall", min: 0, max: 1 }, yAxis: { type: "value", name: "Precision", min: 0, max: 1 }, series: [{ type: "line", showSymbol: false, data: curve?.recall.map((x, index) => [x, curve.precision[index]]) ?? [], lineStyle: { color: "#c86b32", width: 3 } }] }} />;
}

export function ConfusionChart({ split }: { split: AnalysisSplit }) {
  const matrix = split.confusion_matrix;
  const data = matrix?.values.flatMap((row, y) => row.map((value, x) => [x, y, value])) ?? [];
  return <Chart option={{ ...base, tooltip: { position: "top" }, xAxis: { type: "category", name: "Predicted", data: matrix?.labels ?? [] }, yAxis: { type: "category", name: "Actual", data: matrix?.labels ?? [] }, visualMap: { min: 0, max: Math.max(1, ...data.map((row) => Number(row[2]))), calculable: false, orient: "horizontal", left: "center", bottom: 0, inRange: { color: ["#edf5f3", "#087f8c"] } }, series: [{ type: "heatmap", data, label: { show: true } }] }} />;
}

export function RegressionChart({ split, mode }: { split: AnalysisSplit; mode: "predictions" | "residuals" }) {
  const data = split.regression_diagnostics;
  const points = mode === "predictions" ? data?.actual.map((actual, index) => [actual, data.predicted[index]]) : data?.predicted.map((predicted, index) => [predicted, data.residuals[index]]);
  return <Chart option={{ ...base, tooltip: { trigger: "item" }, xAxis: { type: "value", name: mode === "predictions" ? "Actual" : "Predicted" }, yAxis: { type: "value", name: mode === "predictions" ? "Predicted" : "Residual" }, series: [{ type: "scatter", data: points ?? [], symbolSize: 7, itemStyle: { color: mode === "predictions" ? "#087f8c" : "#c86b32", opacity: .7 } }] }} />;
}

export function ImportanceChart({ items }: { items: Array<{ feature: string; importance: number }> }) {
  const selected = [...items].slice(0, 12).reverse();
  return <Chart option={{ ...base, grid: { left: 140, right: 30, top: 12, bottom: 30 }, xAxis: { type: "value" }, yAxis: { type: "category", data: selected.map((item) => item.feature) }, series: [{ type: "bar", data: selected.map((item) => item.importance), itemStyle: { color: "#087f8c", borderRadius: [0, 3, 3, 0] } }] }} />;
}

function Chart({ option }: { option: Record<string, unknown> }) { return <ReactEChartsCore echarts={echarts} option={option} style={{ height: 320, width: "100%" }} lazyUpdate />; }

function sameConvergenceEvents(previous: { events: RunEvent[] }, next: { events: RunEvent[] }) {
  const before = previous.events.filter((event) => event.type === "trial_completed");
  const after = next.events.filter((event) => event.type === "trial_completed");
  const beforePlanned = [...previous.events].reverse().find((event) => event.type === "run_started")?.payload.planned_trials;
  const afterPlanned = [...next.events].reverse().find((event) => event.type === "run_started")?.payload.planned_trials;
  return beforePlanned === afterPlanned && before.length === after.length && before.every((event, index) =>
    event.sequence_number === after[index].sequence_number
    && event.payload.cost === after[index].payload.cost
    && event.payload.trial_id === after[index].payload.trial_id
  );
}
