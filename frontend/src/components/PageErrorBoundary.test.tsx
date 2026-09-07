import { render, screen } from "@testing-library/react";
import { expect, it, vi } from "vitest";
import { PageErrorBoundary } from "./PageErrorBoundary";

it("offers recovery when a page cannot load after a dashboard update", () => {
  const errors = vi.spyOn(console, "error").mockImplementation(() => undefined);
  const expectedError = (event: ErrorEvent) => {
    if (event.message.includes("Failed to fetch dynamically imported module")) event.preventDefault();
  };
  window.addEventListener("error", expectedError);
  function FailedPage(): never { throw new Error("Failed to fetch dynamically imported module"); }
  try {
    render(<PageErrorBoundary><FailedPage/></PageErrorBoundary>);
    expect(screen.getByRole("alert")).toHaveTextContent("This page could not be loaded");
    expect(screen.getByRole("button", { name: "Reload dashboard" })).toBeVisible();
  } finally { errors.mockRestore(); window.removeEventListener("error", expectedError); }
});
