# Plan: Layout, Hero, Viewer, Scatter (implemented)

## What was done

### 1) Automatic adaptation to computer window (layout)

- **Reference:** `third_party/cool_website_only_for_layout_rules/cool_website.html` (breakpoints 810px, 1200px; full-width layout).
- **Changes:**
  - **`frontend/app/globals.css`:** Added `.layout-container` with `width: 100%`, `max-width: 80rem`, `margin: 0 auto`, and responsive padding `clamp(1rem, 4vw, 2rem)` so content adapts to window width.
  - **`frontend/app/page.tsx`:** Header and main use `layout-container` instead of `max-w-6xl mx-auto px-4`. Header outer div has `w-full` so the bar spans the full viewport; inner content is correctly sized and padded. Root div has `w-full`.
- **Result:** Header and main content scale with the window; no “centered but too narrow” effect on large or small screens.

### 2) Larger headers (4/5) and new hero copy

- **`frontend/components/HeroTypewriter.tsx`:**
  - **Copy:**  
    - Line 1: **"Trained on 15,000+ Gigapixel slides."**  
    - Line 2: **"Across 32 Cancer types."**  
    - Line 3: **"2× more accurate than human pathologists."** with **"2× more accurate"** in *italics*.
  - **Size:** Hero lines use `text-3xl md:text-4xl` (up from `text-xl md:text-2xl`). Container `max-w-4xl`.
- **`frontend/components/CaseExplorer.tsx`:** Tumor name (middle header) uses `text-2xl md:text-3xl`.
- **`frontend/components/PerformanceTab.tsx`:** “Predictions vs genomic purity” uses `text-xl md:text-2xl font-semibold`.

### 3) Viewer: centered, solid, scale-to-fit, scroll, default opacity 70%

- **Case Explorer (React):**
  - Viewer wrapper: `flex items-center justify-center`, `min-h-[400px]`, `rounded-xl`, `w-full`. Iframe `w-full h-full min-h-[380px] max-w-full`.
  - Section has `scroll-mt-4` so when scrolling the viewer stays in view.
- **Generated HTML (`ml/enso_purity_mil/interactive_viewer.py`):**
  - Overlay opacity slider default set to **70%** (`value="70"`, `default_opacity_pct=70` in `_write_html`).
  - Body uses flex column and centering so the canvas scales with the container; canvas `max-width: 100%`.

### 4) Viewer HTML: new title, cancer name left, table, hide old block

- **`ml/enso_purity_mil/interactive_viewer.py`:**
  - **Removed:** “Enso Biosciences — Purity Heatmap Viewer”, UUID line, and the four-metric block (Expected, Predicted, |Δ|, Tiles scored).
  - **New content:**
    - **Title:** “Map of Tumor Purity (% of cancer cells in the image)”.
    - **Cancer name:** Left-aligned (`.cancer { align-self: flex-start }`), below title.
    - **Table:** One row, columns “Ground truth (to match)” | “Pathologist” | “Enso” with values `expected`, `ptn`, `predicted`.
  - **CLI:** `--cancer-name` and `--ptn` added; passed from `build_demo_gallery.py` using `TCGA_DISPLAY_NAMES` and `row["ptn"]`.
- **`ml/enso_purity_mil/build_demo_gallery.py`:**
  - Added `TCGA_DISPLAY_NAMES` (same mapping as frontend).
  - Viewer subprocess now gets `--cancer-name` and `--ptn` for each slide.
- **Case Explorer:** Metrics table (Ground truth (to match) | Pathologist | Enso) is **above** the viewer iframe; tumor name is left-aligned above the table.

### 5) Scatter “predictions vs genomic purity” plotted in the website

- **Current state:**  
  - Performance tab loads `/data/scatter_data.json`. If it contains `genomic_purity` (and matching) arrays, it renders **ScatterChart** (interactive, theme-aware).  
  - If the file is missing or a stub (e.g. only `{"note": "..."}`), it falls back to the static PNG `/data/scatter_mil_vs_ptn.png`.
- **Changes:**
  - **`frontend/components/PerformanceTab.tsx`:** Scatter section uses `scatterData?.genomic_purity?.length` so only valid data shows the chart; stub JSON shows the PNG. Section has larger title, `min-h-[320px]`, `overflow-visible`, and clearer structure so the plot is always visible when data or PNG exists.
  - **`docs/vm_generate_artifacts.md`:** “Copy to local” section updated to include `scatter_data.json` and `scatter_mil_vs_ptn.png`; note that replacing the stub with the real file from the VM enables the interactive scatter.
- **To get the interactive scatter:** Run `scripts/run_on_vm_generate_all.sh` on the VM (step 1 writes `scatter_data.json` and the PNG to `ml/runs/fold0/stats/`). Then copy those files into `frontend/public/data/` (see `docs/vm_generate_artifacts.md`). If you have `GCP_SSH_*` set, you can run the script on the VM and `scp` the stats folder into `frontend/public/data/`.

---

## Files touched

| Area        | Files |
|------------|--------|
| Layout     | `frontend/app/globals.css`, `frontend/app/page.tsx` |
| Hero       | `frontend/components/HeroTypewriter.tsx` |
| Case Explorer | `frontend/components/CaseExplorer.tsx` |
| Viewer HTML | `ml/enso_purity_mil/interactive_viewer.py`, `ml/enso_purity_mil/build_demo_gallery.py` |
| Scatter    | `frontend/components/PerformanceTab.tsx`, `docs/vm_generate_artifacts.md` |
| Plan       | `docs/plan_layout_hero_viewer_scatter.md` (this file) |

---

## How to run and verify

- **Frontend:** `cd frontend && npm run dev` then open the app; check header width, hero text (size, copy, italics), Case Explorer (tumor name left, table above viewer, viewer centered), Performance (scatter section visible; chart or PNG).
- **Viewer HTML:** Regenerate gallery on the VM so new `interactive_*.html` files use the new title, cancer name, table, and 70% default opacity. Copy `frontend/gallery/*` from VM to `frontend/public/gallery/`.
- **Scatter data:** On VM run `bash scripts/run_on_vm_generate_all.sh`; copy `ml/runs/fold0/stats/scatter_data.json` and `scatter_mil_vs_ptn.png` to `frontend/public/data/` to see the interactive scatter.

---

## Known limitations / next steps

- **Cool website reference:** The reference HTML is a large Framer export; only breakpoints and full-width + padded container ideas were applied. No pixel-perfect copy.
- **Scatter:** Interactive chart appears only when `scatter_data.json` in `public/data/` contains the three arrays; otherwise the PNG is shown. Replace the stub file with VM-generated data for interactivity.
- **Viewer iframe:** “Scale to fit” is achieved via the wrapper and canvas `max-width: 100%` inside the HTML; the iframe itself fills the React container. For true “image fits viewport” inside the iframe you could add JS that scales the canvas to the iframe’s size (future enhancement).
