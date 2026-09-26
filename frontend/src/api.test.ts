import { afterEach, describe, expect, it, vi } from "vitest";
import { api } from "./api";

afterEach(() => vi.unstubAllGlobals());

describe("versioned API client", () => {
  it("presents schema validation messages without exposing the transport structure", async () => {
    vi.stubGlobal("fetch", vi.fn().mockResolvedValue({ ok: false, json: async () => ({ detail: [{ loc: ["body", "preprocessing"], msg: "Value error, Ordinal features need an ordered category list." }] }) }));
    await expect(api.snapshot("invalid")).rejects.toThrow("Ordinal features need an ordered category list.");
  });
  it("loads only incremental run history from the v1 service", async () => {
    const fetchMock = vi.fn().mockResolvedValue({
      ok: true,
      json: async () => ({ run_id: "run-1", events: [] }),
    });
    vi.stubGlobal("fetch", fetchMock);

    await api.history("run-1", 12);

    expect(fetchMock).toHaveBeenCalledWith("/api/v1/runs/run-1/history?after=12", undefined);
  });

  it("uses v1 export URLs", () => {
    expect(api.exportUrl("run-1", "predictions")).toBe(
      "/api/v1/runs/run-1/exports/predictions",
    );
  });

  it("surfaces structured backend failures", async () => {
    vi.stubGlobal(
      "fetch",
      vi.fn().mockResolvedValue({
        ok: false,
        json: async () => ({ detail: "Metric and task are incompatible." }),
      }),
    );

    await expect(api.snapshot("bad-run")).rejects.toThrow("Metric and task are incompatible.");
  });
});
