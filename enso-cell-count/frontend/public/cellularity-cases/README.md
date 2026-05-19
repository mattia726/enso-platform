# EnsoCellularity Case Assets

This folder is a frontend-ready cellularity artifact bundle for the same demo
slides used by the current EnsoPurity case explorer.

Use `/cellularity-cases/case_N_base.jpg` as the H&E image and
`/cellularity-cases/case_N_cellularity_mask.png` as the RGBA cellularity overlay.
The preview composites in `/cellularity-cases/previews/` use 70% overlay opacity
and are included only for visual review.

The default palette is aurora-ember: deep blue for low cellularity, teal/green
through the midrange, and a controlled amber-orange accent for dense cellular
regions.

Regenerate from the repository root with:

```bash
python frontend/scripts/build_cellularity_artifacts.py
```
