// Visual assessment for the macrodissection workbench.
//
// Runs Chromium against a live `next start` server and captures
// the eight named screenshots described in the plan. Run via:
//
//   npm run start &  # serves the built site on http://localhost:3000
//   npx playwright test tests/visual/macrodissection.spec.ts
//
// The output directory is read from env var ENSO_SCREENSHOT_DIR;
// falls back to ./test-results/screenshots if unset.

import { expect, test } from "@playwright/test";
import { mkdirSync } from "node:fs";
import path from "node:path";

const OUT_DIR =
  process.env.ENSO_SCREENSHOT_DIR ?? path.join(process.cwd(), "test-results", "screenshots");

const TARGET = process.env.ENSO_WORKBENCH_URL ?? "http://localhost:3000/macrodissection";

mkdirSync(OUT_DIR, { recursive: true });

function shot(name: string): string {
  return path.join(OUT_DIR, `${name}.png`);
}

test.describe("Macrodissection workbench visual assessment", () => {
  test.setTimeout(120_000);

  test("captures the 8 reference screenshots", async ({ page }) => {
    await page.setViewportSize({ width: 1600, height: 950 });
    await page.goto(TARGET, { waitUntil: "networkidle" });

    // Wait for the workbench to be ready: OpenSeadragon canvas + an active
    // layer button.
    await page.waitForSelector("[data-layer-button='adequacy']", { timeout: 30000 });
    await page.waitForFunction(
      () => Boolean(document.querySelector("canvas#enso-heatmap-overlay")),
      undefined,
      { timeout: 30000 },
    );
    await page.waitForTimeout(700); // let OSD settle on the home zoom

    // 1) Workbench overview (default: adequacy overlay at low zoom)
    await page.screenshot({ path: shot("01_workbench_overview_adequacy_low_zoom"), fullPage: false });

    // 2) Purity overlay
    await page.click("[data-layer-button='purity']");
    await page.waitForTimeout(400);
    await page.screenshot({ path: shot("02_purity_overlay_low_zoom") });

    // 3) Cellularity overlay
    await page.click("[data-layer-button='cellularity']");
    await page.waitForTimeout(400);
    await page.screenshot({ path: shot("03_cellularity_overlay_low_zoom") });

    // 4) Adequacy back, with tile detail (high zoom-look) — use detail
    //    smoothing setting so the raw tile lattice becomes visible at low
    //    zoom too, demonstrating that smoothing is decoupled.
    await page.click("[data-layer-button='adequacy']");
    await page.click("[data-smoothing-button='detail']");
    await page.waitForTimeout(400);
    await page.screenshot({ path: shot("04_adequacy_detail_smoothing") });

    // Reset smoothing for the ROI walk-through.
    await page.click("[data-smoothing-button='balanced']");
    await page.waitForTimeout(300);

    // 5) Auto-suggested candidates panel populated.
    await page.click("[data-auto-suggest]");
    await page.waitForTimeout(800);
    await page.screenshot({ path: shot("05_candidate_areas_list") });

    // Click the first candidate to load an editable polygon.
    const firstCand = page.locator("[data-candidate-rank='1']").first();
    if (await firstCand.count()) {
      await firstCand.click();
      await page.waitForTimeout(800);
    } else {
      // Fallback: draw a polygon manually around the centre.
      const viewer = page.locator("[data-wsi-viewer]").first();
      const box = await viewer.boundingBox();
      if (box) {
        const cx = box.x + box.width / 2;
        const cy = box.y + box.height / 2;
        const radius = Math.min(box.width, box.height) / 6;
        const corners = [
          [cx - radius, cy - radius],
          [cx + radius, cy - radius],
          [cx + radius, cy + radius],
          [cx - radius, cy + radius],
        ];
        for (const [x, y] of corners) {
          await page.mouse.click(x, y);
          await page.waitForTimeout(150);
        }
        // Double-click last vertex to close.
        await page.mouse.dblclick(corners[3][0], corners[3][1]);
      }
      await page.waitForTimeout(600);
    }

    // 6) ROI drawn + adequacy card showing pass/borderline/fail.
    await page.screenshot({ path: shot("06_roi_drawn_with_metrics_card") });

    // 7) Save the ROI then lock it; show the saved-ROI sidebar entry.
    await page.click("[data-save-roi]");
    await page.waitForTimeout(400);
    const lockBtn = page.locator("[data-roi-lock]").first();
    if (await lockBtn.count()) {
      await lockBtn.click();
      await page.waitForTimeout(400);
    }
    await page.screenshot({ path: shot("07_locked_roi_in_history") });

    // 8) Open the printable macrodissection sheet.
    await page.click("[data-open-report]");
    await page.waitForTimeout(800);
    await page.screenshot({ path: shot("08_macrodissection_sheet") });

    // Sanity assertion — at least the verdict label must appear in DOM.
    await expect(page.locator("[data-report-sheet]")).toBeVisible();
  });
});
