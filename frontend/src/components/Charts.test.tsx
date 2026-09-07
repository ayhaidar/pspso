import { render } from "@testing-library/react";
import { beforeEach, expect, it, vi } from "vitest";
import type { RunEvent } from "../types";
import { ConvergenceChart } from "./Charts";

const chartRender = vi.hoisted(() => vi.fn());

vi.mock("echarts-for-react/lib/core", () => ({
  default: (props: unknown) => {
    chartRender(props);
    return null;
  },
}));

beforeEach(() => chartRender.mockClear());

it("keeps the convergence chart mounted until a completed trial changes", () => {
  const completed = event(1, "trial_completed", { cost: 0.4 });
  const { rerender } = render(<ConvergenceChart events={[completed]} />);

  rerender(<ConvergenceChart events={[completed, event(2, "run_log", {})]} />);
  expect(chartRender).toHaveBeenCalledTimes(1);

  rerender(<ConvergenceChart events={[
    completed,
    event(2, "run_log", {}),
    event(3, "trial_completed", { cost: 0.3 }),
  ]} />);
  expect(chartRender).toHaveBeenCalledTimes(2);
});

it("plots best cost by candidate completion rather than by individual fold fit", () => {
  render(<ConvergenceChart events={[
    event(1, "run_started", { planned_trials: 3, planned_candidate_fits: 15 }),
    event(2, "trial_completed", { trial_id: 2, cost: 0.4 }),
    event(3, "model_fit_completed", { fit_id: 11 }),
    event(4, "trial_completed", { trial_id: 1, cost: 0.25 }),
    event(5, "trial_completed", { trial_id: 3, cost: 0.3 }),
  ]}/>)

  const props = chartRender.mock.calls[chartRender.mock.calls.length - 1]?.[0] as {
    option: {
      xAxis: { name: string; data: number[] };
      yAxis: { name: string; scale: boolean };
      series: Array<{ data: number[]; step: string; smooth: boolean }>;
      tooltip: { formatter: (items: Array<{ dataIndex: number }>) => string };
    };
  };
  expect(props.option.xAxis).toMatchObject({
    name: "Candidates completed (of 3)",
    data: [1, 2, 3],
  });
  expect(props.option.yAxis).toMatchObject({ name: "Best cost so far", scale: true });
  expect(props.option.series[0]).toMatchObject({
    data: [0.4, 0.25, 0.25],
    step: "end",
    smooth: false,
  });
  expect(props.option.tooltip.formatter([{ dataIndex: 0 }])).toContain("PSO trial ID: 2");
});

function event(sequenceNumber: number, type: string, payload: Record<string, unknown>): RunEvent {
  return {
    sequence_number: sequenceNumber,
    type,
    timestamp: "2026-09-07T00:00:00Z",
    payload,
  };
}
