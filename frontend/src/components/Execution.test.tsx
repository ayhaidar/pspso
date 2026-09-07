import { render, screen } from "@testing-library/react";
import { describe, expect, it } from "vitest";
import type { RunEvent, RunSnapshot } from "../types";
import { EventTimeline, StrategyCanvas } from "./Execution";

describe("StrategyCanvas", () => {
  it("shows the configured PSO concurrency without claiming queued particles are active", () => {
    const run = {
      run_id: "run-1",
      experiment_id: "experiment-1",
      experiment_name: "Parallel check",
      status: "running",
      event_count: 2,
      strategy: "pso",
      request: {
        strategy: "pso",
        pso: { particles: 5, iterations: 2, topology: "global" },
        runtime: { max_trials: 10, trial_workers: 2 },
      },
    } as RunSnapshot;
    const events = [
      event(1, "run_started", { planned_trials: 10 }),
      event(2, "trial_started", { trial_id: 1, iteration: 0, particle_index: 0 }),
    ];

    render(<StrategyCanvas run={run} events={events} />);

    expect(screen.getByText(/up to 2 particles are evaluated concurrently/i)).toBeInTheDocument();
    expect(screen.getByText(/1 active/i)).toBeInTheDocument();
  });
});

it("separates candidate fits from the additional winner refit", () => {
  render(<EventTimeline events={[
    event(1, "run_started", {
      strategy: "pso",
      planned_trials: 200,
      planned_candidate_fits: 200,
      planned_fits: 201,
    }),
  ]} />);

  expect(screen.getByText(/200 candidates · 200 candidate fits \+ 1 winner refit/)).toBeVisible();
});

function event(sequence_number: number, type: string, payload: Record<string, unknown>): RunEvent {
  return { sequence_number, type, payload, timestamp: new Date().toISOString() };
}
