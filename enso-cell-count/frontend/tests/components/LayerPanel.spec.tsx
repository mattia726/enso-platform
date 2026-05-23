// Component test: layer panel toggles + opacity slider + smoothing tier +
// threshold profile dropdown fire the right onChange callbacks.

import { describe, expect, it } from "vitest";
import { render, screen, fireEvent } from "@testing-library/react";

import LayerPanel, {
  type LayerPanelState,
} from "@/app/macrodissection/components/LayerPanel";

function initial(): LayerPanelState {
  return {
    layer: "adequacy",
    opacity: 0.65,
    smoothing: "balanced",
    showRawTiles: false,
    profileName: "humanitas_ngs",
    roiToolEnabled: true,
  };
}

describe("<LayerPanel />", () => {
  it("activates the adequacy layer by default", () => {
    const calls: LayerPanelState[] = [];
    render(<LayerPanel state={initial()} onChange={(s) => calls.push(s)} />);
    const adequacy = screen.getByRole("button", { name: /^Adequacy/ });
    expect(adequacy.getAttribute("aria-pressed")).toBe("true");
  });

  it("switches the active layer when a button is clicked", () => {
    const calls: LayerPanelState[] = [];
    render(<LayerPanel state={initial()} onChange={(s) => calls.push(s)} />);
    fireEvent.click(screen.getByRole("button", { name: /^Purity/ }));
    expect(calls).toHaveLength(1);
    expect(calls[0].layer).toBe("purity");
  });

  it("emits the new opacity on slider change", () => {
    const calls: LayerPanelState[] = [];
    render(<LayerPanel state={initial()} onChange={(s) => calls.push(s)} />);
    const slider = document.querySelector("[data-opacity-slider]") as HTMLInputElement;
    fireEvent.change(slider, { target: { value: "0.4" } });
    expect(calls).toHaveLength(1);
    expect(calls[0].opacity).toBeCloseTo(0.4, 5);
  });

  it("changes the smoothing tier", () => {
    const calls: LayerPanelState[] = [];
    render(<LayerPanel state={initial()} onChange={(s) => calls.push(s)} />);
    fireEvent.click(document.querySelector("[data-smoothing-button='detail']")!);
    expect(calls[0].smoothing).toBe("detail");
  });

  it("changes the threshold profile via the dropdown", () => {
    const calls: LayerPanelState[] = [];
    render(<LayerPanel state={initial()} onChange={(s) => calls.push(s)} />);
    const select = document.querySelector("[data-profile-select]") as HTMLSelectElement;
    fireEvent.change(select, { target: { value: "research" } });
    expect(calls[0].profileName).toBe("research");
  });

  it("toggles the ROI drawing tool", () => {
    const calls: LayerPanelState[] = [];
    render(<LayerPanel state={initial()} onChange={(s) => calls.push(s)} />);
    fireEvent.click(document.querySelector("[data-roi-tool-toggle]")!);
    expect(calls[0].roiToolEnabled).toBe(false);
  });
});
