import { act, fireEvent, render, screen, waitFor } from "@testing-library/react";
import { afterAll, afterEach, beforeAll, beforeEach, describe, expect, it, vi } from "vitest";
import { HttpResponse, http } from "msw";
import { setupServer } from "msw/node";
import { WorkflowProvider, mergeEvents, reconcileRunSnapshot, useWorkflow } from "./workflow";
import type { RunEvent, RunSnapshot } from "./types";

const events: RunEvent[] = [];
let status = "running";
let valid = true;
let cancelRequested = false;
const historyCursors: number[] = [];
const metadata = {
  estimators: { svm: { label: "SVM", tasks: ["binary_classification"], dependency: { installed: true },
    defaults: { binary_classification: { fixed_params: {}, search_space: { C: { type: "choice", values: [1] } } } } } },
  tasks: { binary_classification: { metrics: ["roc_auc"] } }, metrics: {},
};
const server = setupServer(
  http.get("*/api/v1/estimators", () => HttpResponse.json(metadata)),
  http.get("*/api/v1/datasets/examples", () => HttpResponse.json([])),
  http.get("*/api/v1/datasets", () => HttpResponse.json([])),
  http.get("*/api/v1/experiments", () => HttpResponse.json([])),
  http.post("*/api/v1/workflow/validate", () => HttpResponse.json({ valid, errors: valid ? {} : { dataset: ["Select a target"] } })),
  http.get("*/api/v1/runs/run-1", () => HttpResponse.json({ run_id: "run-1", experiment_id: "experiment-1", experiment_name: "Test", status, cancel_requested: cancelRequested })),
  http.get("*/api/v1/runs/run-1/history", ({ request }) => {
    const after = Number(new URL(request.url).searchParams.get("after") ?? 0);
    historyCursors.push(after);
    return HttpResponse.json({ run_id: "run-1", events: events.filter((event) => (event.sequence_number ?? 0) > after) });
  }),
  http.get("*/api/v1/runs/run-1/analysis", () => HttpResponse.json({
    task: "binary_classification", metric: "roc_auc", train: { metrics: {} },
    validation: { metrics: { roc_auc: 0.9 } }, feature_importance: { available: false, items: [] },
  })),
  http.post("*/api/v1/runs/run-1/cancel", () => {
    cancelRequested = true;
    return HttpResponse.json({ run_id: "run-1", cancel_requested: true });
  }),
);

class EventSourceMock extends EventTarget {
  static instances: EventSourceMock[] = [];
  onerror: (() => void) | null = null;
  onopen: (() => void) | null = null;
  closed = false;
  constructor(public url: string) { super(); EventSourceMock.instances.push(this); }
  close() { this.closed = true; }
  send(event: RunEvent) { this.dispatchEvent(new MessageEvent(event.type, { data: JSON.stringify(event) })); }
}

function Harness() {
  const workflow = useWorkflow();
  return <>
    <button onClick={() => void workflow.loadRun("run-1")}>Load run</button>
    <button onClick={() => void workflow.validateStage("data")}>Validate data</button>
    <button onClick={() => workflow.updateDraft({ metric: "accuracy" })}>Change metric</button>
    <button onClick={() => void workflow.startRun()}>Start run</button>
    <button onClick={() => void workflow.cancelRun()}>Cancel run</button>
    <output data-testid="events">{workflow.events.map((event) => event.sequence_number).join(",")}</output>
    <output data-testid="status">{workflow.run?.status}</output>
    <output data-testid="validation">{String(workflow.validations.data?.valid ?? "unchecked")}</output>
    <output data-testid="full-validation">{String(workflow.validations.full?.valid ?? "unchecked")}</output>
    <output data-testid="connection">{workflow.connection}</output>
    <output data-testid="cancel-requested">{String(workflow.run?.cancel_requested ?? false)}</output>
    <output data-testid="message">{workflow.message}</output>
  </>;
}

beforeAll(() => server.listen({ onUnhandledRequest: "error" }));
beforeEach(() => {
  localStorage.clear(); events.length = 0; historyCursors.length = 0; status = "running"; valid = true; cancelRequested = false;
  EventSourceMock.instances = [];
  vi.stubGlobal("EventSource", EventSourceMock);
});
afterEach(() => { server.resetHandlers(); vi.unstubAllGlobals(); });
afterAll(() => server.close());

describe("workflow", () => {
  it("invalidates stage approval after the draft changes", async () => {
    render(<WorkflowProvider><Harness/></WorkflowProvider>);
    fireEvent.click(screen.getByText("Validate data"));
    await waitFor(() => expect(screen.getByTestId("validation")).toHaveTextContent("true"));
    fireEvent.click(screen.getByText("Change metric"));
    expect(screen.getByTestId("validation")).toHaveTextContent("unchecked");
  });

  it("does not submit a run when validation fails", async () => {
    valid = false;
    let submissions = 0;
    server.use(http.post("*/api/v1/runs", () => { submissions++; return HttpResponse.json({}); }));
    render(<WorkflowProvider><Harness/></WorkflowProvider>);
    fireEvent.click(screen.getByText("Start run"));
    await waitFor(() => expect(screen.getByTestId("full-validation")).toHaveTextContent("false"));
    expect(submissions).toBe(0);
  });

  it("resumes after saved history, deduplicates replay, and refreshes terminal results", async () => {
    events.push({ sequence_number: 1, type: "run_started", timestamp: "2026-09-06T00:00:00Z", payload: {} });
    const rendered = render(<WorkflowProvider><Harness/></WorkflowProvider>);
    fireEvent.click(screen.getByText("Load run"));
    await waitFor(() => expect(EventSourceMock.instances).toHaveLength(1));
    const stream = EventSourceMock.instances[0];
    expect(stream.url).toContain("after=1");
    act(() => {
      stream.send(events[0]);
      stream.send({ ...events[0], sequence_number: 2, type: "model_fit_started" });
      stream.send({ ...events[0], sequence_number: 2, type: "model_fit_started" });
    });
    expect(screen.getByTestId("events")).toHaveTextContent("1,2");
    status = "completed";
    const completed = { ...events[0], sequence_number: 3, type: "run_completed" };
    events.push(completed);
    act(() => stream.send(completed));
    await waitFor(() => expect(screen.getByTestId("status")).toHaveTextContent("completed"));
    expect(historyCursors).toContain(3);
    expect(stream.closed).toBe(true);
    rendered.unmount();
  });

  it("recovers missing events through incremental polling after a connection failure", async () => {
    render(<WorkflowProvider><Harness/></WorkflowProvider>);
    fireEvent.click(screen.getByText("Load run"));
    await waitFor(() => expect(EventSourceMock.instances).toHaveLength(1));
    act(() => EventSourceMock.instances[0].onerror?.());
    events.push({ sequence_number: 1, type: "trial_started", timestamp: "2026-09-06T00:00:00Z", payload: {} });
    await waitFor(() => expect(screen.getByTestId("events")).toHaveTextContent("1"), { timeout: 3000 });
    expect(screen.getByTestId("connection")).toHaveTextContent("polling");
  });

  it("persists immediate cancellation feedback while the worker is stopping", async () => {
    render(<WorkflowProvider><Harness/></WorkflowProvider>);
    fireEvent.click(screen.getByText("Load run"));
    await waitFor(() => expect(screen.getByTestId("status")).toHaveTextContent("running"));

    fireEvent.click(screen.getByText("Cancel run"));

    await waitFor(() => expect(screen.getByTestId("cancel-requested")).toHaveTextContent("true"));
  });

  it("displays a service failure", async () => {
    server.use(http.get("*/api/v1/estimators", () => HttpResponse.json({ detail: "Service unavailable" }, { status: 503 })));
    render(<WorkflowProvider><Harness/></WorkflowProvider>);
    await waitFor(() => expect(screen.getByTestId("message")).toHaveTextContent("Service unavailable"));
  });
});

it("merges an out-of-order event replay into one ordered history", () => {
  const base = { type: "run_log", timestamp: "", payload: {} };
  const current = [{ ...base, sequence_number: 3 }];
  expect(mergeEvents(current, [
    { ...base, sequence_number: 1 }, { ...base, sequence_number: 3 }, { ...base, sequence_number: 2 },
  ]).map((event) => event.sequence_number)).toEqual([1, 2, 3]);
  expect(mergeEvents(current, [])).toBe(current);
  expect(mergeEvents(current, [{ ...base, sequence_number: 3 }])).toBe(current);
});

it("preserves a run snapshot while a background refresh has no changes", () => {
  const current = {
    run_id: "run-1",
    status: "running",
    updated_at: "2026-09-07T00:00:00Z",
    attempts: [{ heartbeat_at: "2026-09-07T00:00:00Z" }],
  } as RunSnapshot;
  const unchanged = JSON.parse(JSON.stringify(current)) as RunSnapshot;
  const updated = { ...unchanged, updated_at: "2026-09-07T00:00:01Z" };

  expect(reconcileRunSnapshot(current, unchanged)).toBe(current);
  expect(reconcileRunSnapshot(current, updated)).toBe(updated);
});
