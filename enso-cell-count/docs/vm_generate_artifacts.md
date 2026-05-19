# Generating All Demo Artifacts on the VM

Run the following **on the GCP VM** (where the bucket is mounted and the model/cache exist) to generate:

1. **Global stats** — `statistical_tests.json`, `scatter_data.json`, `scatter_mil_vs_ptn.png`
2. **Per-cancer stats** — `per_cancer_stats.json`
3. **Gallery and heatmaps** — one interactive HTML per cancer type (each includes the purity heatmap overlay)
4. **Static cases** — `case_1_base.jpg`, `case_1_mask.png`, … in `frontend/public/cases/` for the native viewer (Cloudflare-ready)

## One-command script

From the repo root on the VM:

```bash
bash scripts/run_on_vm_generate_all.sh
```

Or run the steps manually (see the script for exact paths and env vars).

## Copy artifacts to local

After the script finishes, copy from the VM to your local repo:

- **Stats:** `scp -r $VM:~/enso_workspace/ml/runs/fold0/stats/* frontend/public/data/`
- **Gallery:** `scp -r $VM:~/enso_workspace/frontend/gallery/* frontend/public/gallery/`
- **Static cases (native viewer):** `scp -r $VM:~/enso_workspace/frontend/public/cases/* frontend/public/cases/`

The `frontend/public/cases/` folder contains `case_N_base.jpg` (H&E thumbnail) and `case_N_mask.png` (purity heatmap overlay) for each gallery row; the Case Explorer uses these when the native viewer is enabled.

## After generation: review for pen markers

The gallery is built from one slide per cancer type. Some slides may contain **pen/marker annotations** (e.g. pathologist markings on the glass). These should be excluded from the public demo.

1. Open each generated `interactive_<uuid>.html` (e.g. from `frontend/gallery/` after copying from VM) or view thumbnails.
2. If a slide clearly shows pen/marker ink, add its `file_uuid_original` or barcode to `data/exclude_markers.txt` (one per line).
3. Re-run **only** the gallery step on the VM so a different slide for that cancer type is chosen:

   ```bash
   python -m enso_purity_mil.build_demo_gallery \
     --model-path ml/runs/fold0/best_model.pth \
     --manifest data/processed/wedge_mvp_dataset.xlsx \
     --h5-dir ~/bucket_embeddings/embeddings_fp32 \
     --cache-dir ~/enso_workspace/data/cache \
     --out-dir frontend/gallery \
     --one-per-cancer --err-limit 0.15 \
     --exclude-markers data/exclude_markers.txt
   ```

4. Copy the updated `frontend/gallery/*` to your local `frontend/public/gallery/`. Then re-run **step 4** on the VM (export static cases) so `frontend/public/cases/` matches the updated gallery, and copy `frontend/public/cases/*` to local `frontend/public/cases/`.

No automated pen-marker detection is implemented; review is manual.
