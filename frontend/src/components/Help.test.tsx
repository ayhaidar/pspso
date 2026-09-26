import { fireEvent, render, screen } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";
import { StageGuide } from "./Help";

describe("StageGuide", () => {
  it("explains the live execution stage and can be dismissed", () => {
    const onClose = vi.fn();
    render(<StageGuide stage={4} onClose={onClose} />);

    expect(screen.getByText("Follow real execution")).toBeInTheDocument();
    expect(screen.getByText(/finished fits with the planned total/i)).toBeInTheDocument();
    fireEvent.click(screen.getByRole("button", { name: /close stage guide/i }));
    expect(onClose).toHaveBeenCalledOnce();
  });
});
