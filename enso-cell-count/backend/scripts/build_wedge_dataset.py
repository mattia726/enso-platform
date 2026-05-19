"""Build the definitive wedge MVP dataset.

Steps:
  1. Load slides_metadata_report xlsx, filter mpp < 2.
  2. Query GDC /files API to check which file_ids still exist.
  3. Resolve duplicates (same barcode, different file_id): drop missing,
     keep largest if both alive, flag uuid changes.
  4. Remove DX slides.
  5. Tag sample_type=11 as normal (purity=0.0).
  6. Match remaining slides to ABSOLUTE purity via GDC biospecimen API.
  7. Query GDC /files for percent_tumor_nuclei annotations.
  8. Produce the clean excel.
"""
from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests
from tenacity import retry, stop_after_attempt, wait_exponential

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)-8s  %(message)s")
logger = logging.getLogger(__name__)

GDC_FILES_URL = "https://api.gdc.cancer.gov/files"
GDC_CASES_URL = "https://api.gdc.cancer.gov/cases"
_BATCH = 500


# ── GDC helpers ───────────────────────────────────────────────────
@retry(stop=stop_after_attempt(4), wait=wait_exponential(multiplier=2, min=2, max=30))
def _post_gdc(url: str, payload: dict) -> dict:
    resp = requests.post(url, json=payload,
                         headers={"Content-Type": "application/json"}, timeout=120)
    resp.raise_for_status()
    return resp.json()


def check_files_existence(file_ids: list[str]) -> dict[str, dict]:
    """Return {file_id: {exists, file_id_current, file_name, ...}} for each queried id."""
    result: dict[str, dict] = {}
    unique = sorted(set(file_ids))
    n_batches = (len(unique) + _BATCH - 1) // _BATCH

    for i in range(n_batches):
        batch = unique[i * _BATCH : (i + 1) * _BATCH]
        logger.info("GDC /files existence check batch %d/%d (%d ids)", i + 1, n_batches, len(batch))
        payload = {
            "filters": {"op": "in", "content": {"field": "file_id", "value": batch}},
            "fields": "file_id,file_name,file_size,access,state",
            "size": len(batch),
        }
        data = _post_gdc(GDC_FILES_URL, payload)
        for hit in data.get("data", {}).get("hits", []):
            result[hit["file_id"]] = {
                "exists": True,
                "file_name": hit.get("file_name", ""),
                "file_size": hit.get("file_size"),
                "state": hit.get("state", ""),
                "access": hit.get("access", ""),
            }
        if i < n_batches - 1:
            time.sleep(0.2)

    for fid in unique:
        if fid not in result:
            result[fid] = {"exists": False}

    logger.info("GDC file existence: %d found, %d missing",
                sum(1 for v in result.values() if v["exists"]),
                sum(1 for v in result.values() if not v["exists"]))
    return result


def fetch_biospecimen_for_cases(case_ids: list[str], batch_size: int = 250) -> list[dict]:
    """Fetch expanded biospecimen from GDC /cases."""
    all_hits: list[dict] = []
    unique = sorted(set(case_ids))
    n_batches = (len(unique) + batch_size - 1) // batch_size

    for i in range(n_batches):
        batch = unique[i * batch_size : (i + 1) * batch_size]
        if i % 10 == 0:
            logger.info("GDC /cases biospecimen batch %d/%d", i + 1, n_batches)
        payload = {
            "filters": {"op": "in", "content": {"field": "submitter_id", "value": batch}},
            "expand": "samples.portions.slides,samples.portions.analytes.aliquots",
            "fields": "submitter_id,project.project_id",
            "size": batch_size,
        }
        offset = 0
        while True:
            payload["from"] = offset
            data = _post_gdc(GDC_CASES_URL, payload)
            hits = data.get("data", {}).get("hits", [])
            all_hits.extend(hits)
            pag = data.get("data", {}).get("pagination", {})
            if pag.get("page", 1) >= pag.get("pages", 1):
                break
            offset += batch_size
        if i < n_batches - 1:
            time.sleep(0.15)

    logger.info("GDC biospecimen: fetched %d case records", len(all_hits))
    return all_hits


def parse_slide_aliquot_map(hits: list[dict]) -> dict[str, dict]:
    """Build {slide_submitter_id: {aliquots, portion_sub, project_id, ...}}."""
    mapping: dict[str, dict] = {}
    for case in hits:
        case_sub = case.get("submitter_id", "")
        project_id = ""
        proj = case.get("project", {})
        if isinstance(proj, dict):
            project_id = proj.get("project_id", "")
        for sample in case.get("samples", []):
            sample_sub = sample.get("submitter_id", "")
            all_aliquots_by_portion: dict[str, list[str]] = {}
            slide_to_portion: dict[str, str] = {}
            for portion in sample.get("portions", []):
                portion_sub = portion.get("submitter_id", "")
                aliquots: list[str] = []
                for analyte in portion.get("analytes", []):
                    for al in analyte.get("aliquots", []):
                        aliquots.append(al["submitter_id"])
                all_aliquots_by_portion[portion_sub] = aliquots
                for slide in portion.get("slides", []):
                    slide_to_portion[slide["submitter_id"]] = portion_sub
            for slide_sub, portion_sub in slide_to_portion.items():
                same_portion = all_aliquots_by_portion.get(portion_sub, [])
                mapping[slide_sub] = {
                    "case_submitter_id": case_sub,
                    "sample_submitter_id": sample_sub,
                    "portion_submitter_id": portion_sub,
                    "same_portion_aliquots": sorted(same_portion),
                    "project_id": project_id,
                }
    return mapping


def fetch_percent_tumor_nuclei(
    file_id_to_barcode: dict[str, str],
) -> dict[str, float | None]:
    """Query GDC /files for percent_tumor_nuclei, matching by slide submitter_id.

    Parameters
    ----------
    file_id_to_barcode : dict
        ``{file_id: barcode}`` so we can match the correct slide entity
        within the returned biospecimen tree (avoids picking a sibling
        slide's annotation).
    """
    result: dict[str, float | None] = {}
    unique = sorted(set(file_id_to_barcode.keys()))
    n_batches = (len(unique) + _BATCH - 1) // _BATCH

    for i in range(n_batches):
        batch = unique[i * _BATCH : (i + 1) * _BATCH]
        if i % 10 == 0:
            logger.info("GDC /files percent_tumor_nuclei batch %d/%d", i + 1, n_batches)
        payload = {
            "filters": {"op": "in", "content": {"field": "file_id", "value": batch}},
            "fields": "file_id,cases.samples.portions.slides.percent_tumor_nuclei,"
                      "cases.samples.portions.slides.submitter_id",
            "size": len(batch),
        }
        data = _post_gdc(GDC_FILES_URL, payload)
        for hit in data.get("data", {}).get("hits", []):
            fid = hit["file_id"]
            target_barcode = file_id_to_barcode.get(fid)
            ptn = None
            try:
                for case in hit.get("cases", []):
                    for sample in case.get("samples", []):
                        for portion in sample.get("portions", []):
                            for slide in portion.get("slides", []):
                                if slide.get("submitter_id") == target_barcode:
                                    v = slide.get("percent_tumor_nuclei")
                                    if v is not None:
                                        ptn = float(v)
            except Exception:
                pass
            result[fid] = ptn
        if i < n_batches - 1:
            time.sleep(0.15)

    return result


def resolve_missing_uuids(barcodes: list[str]) -> dict[str, dict]:
    """For barcodes whose original file_uuid is gone, find the current SVS file_uuid.

    Queries ``/files`` filtered on
    ``cases.samples.portions.slides.submitter_id == <barcode>``
    and picks the SVS hit.

    Returns ``{barcode: {"file_uuid_new": ..., "file_name": ...}}``
    (empty dict value if not found).
    """
    result: dict[str, dict] = {}
    unique = sorted(set(barcodes))
    # These must be queried one-by-one because the filter is per-barcode
    # but we can batch with OR logic by querying in small groups
    batch_size = 50
    n_batches = (len(unique) + batch_size - 1) // batch_size

    for i in range(n_batches):
        batch = unique[i * batch_size : (i + 1) * batch_size]
        if i % 5 == 0:
            logger.info("GDC /files UUID resolution batch %d/%d (%d barcodes)",
                        i + 1, n_batches, len(batch))
        payload = {
            "filters": {
                "op": "and",
                "content": [
                    {
                        "op": "in",
                        "content": {
                            "field": "cases.samples.portions.slides.submitter_id",
                            "value": batch,
                        },
                    },
                    {
                        "op": "=",
                        "content": {
                            "field": "data_format",
                            "value": "SVS",
                        },
                    },
                ],
            },
            "fields": ("file_id,file_name,file_size,"
                       "cases.samples.portions.slides.submitter_id"),
            "size": 1000,
        }
        data = _post_gdc(GDC_FILES_URL, payload)
        for hit in data.get("data", {}).get("hits", []):
            fid = hit["file_id"]
            fname = hit.get("file_name", "")
            # Determine which barcode(s) this file covers
            slide_subs: set[str] = set()
            try:
                for case in hit.get("cases", []):
                    for sample in case.get("samples", []):
                        for portion in sample.get("portions", []):
                            for slide in portion.get("slides", []):
                                slide_subs.add(slide["submitter_id"])
            except Exception:
                pass
            for bc in batch:
                if bc in slide_subs and bc not in result:
                    result[bc] = {"file_uuid_new": fid, "file_name": fname}
        if i < n_batches - 1:
            time.sleep(0.25)

    for bc in unique:
        if bc not in result:
            result[bc] = {}

    n_found = sum(1 for v in result.values() if v)
    logger.info("UUID resolution: %d / %d missing barcodes resolved", n_found, len(unique))
    return result


# ── Main ──────────────────────────────────────────────────────────
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--slides-xlsx", type=Path,
                    default=Path("data/raw/slides_metadata_report(1).xlsx"))
    ap.add_argument("--abs-tsv", type=Path,
                    default=Path("data/raw/TCGA_mastercalls.abs_tables_JSedit.fixed.txt"))
    ap.add_argument("--out-dir", type=Path, default=Path("data/processed"))
    ap.add_argument("--reports-dir", type=Path, default=Path("data/reports"))
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    args.reports_dir.mkdir(parents=True, exist_ok=True)

    # ── 1. Load & filter ─────────────────────────────────────────
    logger.info("Loading slides metadata")
    raw = pd.read_excel(args.slides_xlsx)
    raw = raw.dropna(subset=["full_path"]).copy()
    raw["barcode"] = raw["full_path"].apply(lambda fp: str(fp).split("/")[-1].split(".")[0])
    processed = raw[raw["base_mpp_x"].notna() & (raw["base_mpp_x"] < 2)].copy()
    logger.info("Post mpp<2: %d rows, %d unique file_ids, %d unique barcodes",
                len(processed), processed["file_id"].nunique(), processed["barcode"].nunique())

    # ── 2. Check GDC existence ───────────────────────────────────
    logger.info("Checking file existence on GDC API...")
    existence = check_files_existence(list(processed["file_id"].unique()))
    processed["gdc_exists"] = processed["file_id"].map(lambda fid: existence[fid]["exists"])

    n_alive = processed["gdc_exists"].sum()
    n_dead = (~processed["gdc_exists"]).sum()
    logger.info("GDC: %d alive, %d missing (of %d)", n_alive, n_dead, len(processed))

    # ── 3. Resolve duplicates ────────────────────────────────────
    logger.info("Resolving barcode duplicates...")
    dup_mask = processed.duplicated(subset=["barcode"], keep=False)
    dups = processed[dup_mask].copy()
    non_dups = processed[~dup_mask].copy()
    logger.info("Duplicates: %d barcodes (%d rows), non-dups: %d",
                dups["barcode"].nunique(), len(dups), len(non_dups))

    resolved: list[pd.DataFrame] = [non_dups]
    dup_stats = {"both_alive": 0, "one_alive": 0, "both_dead": 0}
    for bc, group in dups.groupby("barcode"):
        alive = group[group["gdc_exists"]]
        if len(alive) == 0:
            dup_stats["both_dead"] += 1
            # keep the one with largest area anyway (embedding file exists)
            best = group.sort_values("area", ascending=False).head(1)
            resolved.append(best)
        elif len(alive) == 1:
            dup_stats["one_alive"] += 1
            resolved.append(alive)
        else:
            dup_stats["both_alive"] += 1
            best = alive.sort_values("area", ascending=False).head(1)
            resolved.append(best)

    logger.info("Duplicate resolution: %s", dup_stats)
    deduped = pd.concat(resolved, ignore_index=True)
    assert deduped["barcode"].is_unique, "Barcodes not unique after dedup!"
    logger.info("After dedup: %d unique slides", len(deduped))

    # ── 3b. Resolve missing UUIDs via barcode lookup ────────────
    missing_mask = ~deduped["gdc_exists"]
    n_missing = missing_mask.sum()
    if n_missing > 0:
        logger.info("Resolving %d missing file_uuids by barcode...", n_missing)
        missing_barcodes = list(deduped.loc[missing_mask, "barcode"].unique())
        uuid_map = resolve_missing_uuids(missing_barcodes)

        new_uuids = []
        for _, row in deduped.iterrows():
            if row["gdc_exists"]:
                new_uuids.append(None)
            else:
                info = uuid_map.get(row["barcode"], {})
                new_uuids.append(info.get("file_uuid_new"))
        deduped["file_uuid_new"] = new_uuids

        n_resolved = deduped["file_uuid_new"].notna().sum()
        logger.info("Resolved %d / %d missing UUIDs", n_resolved, n_missing)
    else:
        deduped["file_uuid_new"] = None

    # ── 4. Parse barcode fields ──────────────────────────────────
    deduped["section_type"] = deduped["barcode"].apply(lambda b: b.split("-")[5][:2] if len(b.split("-")) >= 6 else None)
    deduped["sample_type_code"] = deduped["barcode"].apply(lambda b: b.split("-")[3][:2] if len(b.split("-")) >= 4 else None)
    deduped["case_id"] = deduped["barcode"].apply(lambda b: "-".join(b.split("-")[:3]))
    deduped["sample_vial"] = deduped["barcode"].apply(lambda b: "-".join(b.split("-")[:4]))

    # ── 5. Remove DX ─────────────────────────────────────────────
    n_before = len(deduped)
    deduped = deduped[deduped["section_type"].isin(["TS", "MS", "BS"])].copy()
    logger.info("Removed DX: %d → %d frozen-section slides", n_before, len(deduped))

    # ── 6. Separate normals (sample_type 11) ─────────────────────
    is_normal = deduped["sample_type_code"] == "11"
    normals = deduped[is_normal].copy()
    tumours = deduped[~is_normal].copy()
    logger.info("Normals (sample_type=11): %d slides", len(normals))
    logger.info("Tumour slides to match: %d", len(tumours))

    normals["purity"] = 0.0
    normals["ploidy"] = np.nan
    normals["aliquot_barcode"] = None
    normals["gdc_match_type"] = "normal_tissue"
    normals["project_id"] = None  # populated after biospecimen fetch
    if "file_uuid_new" not in normals.columns:
        normals["file_uuid_new"] = None

    # ── 7. GDC biospecimen matching for tumours ──────────────────
    logger.info("Fetching GDC biospecimen for tumour slides...")
    abs_df = pd.read_csv(args.abs_tsv, sep="\t")
    called = abs_df[(abs_df["call status"] == "called") & (abs_df["solution"] == "new")].copy()
    abs_lookup = {row["sample"]: {"purity": row["purity"], "ploidy": row["ploidy"]}
                  for _, row in called.iterrows()}
    abs_set = set(abs_lookup.keys())

    case_ids = sorted(tumours["case_id"].unique())
    hits = fetch_biospecimen_for_cases(case_ids)
    mapping = parse_slide_aliquot_map(hits)

    matched_records: list[dict] = []
    unmatched_count = 0
    for _, row in tumours.iterrows():
        bc = row["barcode"]
        info = mapping.get(bc)
        rec = row.to_dict()
        if info is None:
            rec["purity"] = np.nan
            rec["ploidy"] = np.nan
            rec["aliquot_barcode"] = None
            rec["gdc_match_type"] = "no_gdc_record"
            rec["project_id"] = None
            unmatched_count += 1
            matched_records.append(rec)
            continue

        rec["project_id"] = info.get("project_id", "")

        aliquot_id = None
        for a in info["same_portion_aliquots"]:
            if a in abs_set:
                aliquot_id = a
                break

        if aliquot_id:
            rec["purity"] = abs_lookup[aliquot_id]["purity"]
            rec["ploidy"] = abs_lookup[aliquot_id]["ploidy"]
            rec["aliquot_barcode"] = aliquot_id
            rec["gdc_match_type"] = "same_portion"
        else:
            rec["purity"] = np.nan
            rec["ploidy"] = np.nan
            rec["aliquot_barcode"] = None
            rec["gdc_match_type"] = "no_absolute_match"
            unmatched_count += 1

        matched_records.append(rec)

    tumours_matched = pd.DataFrame(matched_records)
    n_matched_tumour = tumours_matched["aliquot_barcode"].notna().sum()
    logger.info("Tumour matching: %d matched, %d unmatched", n_matched_tumour, unmatched_count)

    # Backfill project_id for normals from the biospecimen mapping
    normal_project_ids = []
    for _, row in normals.iterrows():
        info = mapping.get(row["barcode"])
        normal_project_ids.append(info.get("project_id", "") if info else None)
    normals["project_id"] = normal_project_ids

    # ── 8. Combine ───────────────────────────────────────────────
    all_slides = pd.concat([tumours_matched, normals], ignore_index=True)
    logger.info("Combined dataset: %d slides", len(all_slides))

    # ── 9. Fetch percent_tumor_nuclei ────────────────────────────
    logger.info("Fetching percent_tumor_nuclei from GDC...")
    # Use the best available file_id for each slide (new uuid if original is gone)
    all_slides["_lookup_fid"] = all_slides.apply(
        lambda r: r["file_uuid_new"] if pd.notna(r.get("file_uuid_new")) else r["file_id"],
        axis=1,
    )
    fid_to_bc = dict(zip(all_slides["_lookup_fid"], all_slides["barcode"]))
    ptn_map = fetch_percent_tumor_nuclei(fid_to_bc)
    all_slides["percent_tumor_nuclei"] = all_slides["_lookup_fid"].map(ptn_map)
    all_slides.drop(columns=["_lookup_fid"], inplace=True)
    n_ptn = all_slides["percent_tumor_nuclei"].notna().sum()
    logger.info("percent_tumor_nuclei available for %d / %d slides", n_ptn, len(all_slides))

    # ── 10. Clean excel ──────────────────────────────────────────
    cols = [
        "file_id", "barcode", "aliquot_barcode", "gdc_match_type",
        "purity", "ploidy", "percent_tumor_nuclei",
        "sample_type_code", "section_type", "case_id", "sample_vial",
        "project_id",
        "base_mpp_x", "base_mpp_y", "base_width", "base_height", "area",
        "gdc_exists",
    ]
    if "file_uuid_new" in all_slides.columns:
        cols.append("file_uuid_new")
    clean = all_slides[cols].copy()
    clean.rename(columns={"file_id": "file_uuid_original"}, inplace=True)
    if "file_uuid_new" not in clean.columns:
        clean["file_uuid_new"] = None

    out_path = args.out_dir / "wedge_mvp_dataset.xlsx"
    clean.to_excel(out_path, index=False, engine="openpyxl")
    logger.info("Wrote %s (%d rows)", out_path, len(clean))

    # ── 11. Statistics ───────────────────────────────────────────
    print("\n" + "=" * 70)
    print("WEDGE MVP DATASET STATISTICS")
    print("=" * 70)

    has_purity = clean[clean["purity"].notna()]
    has_aliquot = clean[clean["aliquot_barcode"].notna()]

    print(f"  Total unique slides:              {len(clean)}")
    print(f"  Unique barcodes:                  {clean['barcode'].nunique()}")
    print(f"  Slides with purity (incl. normal):{len(has_purity)}")
    print(f"  Slides matched to aliquot:        {len(has_aliquot)}")
    print(f"  Unique aliquots:                  {has_aliquot['aliquot_barcode'].nunique()}")
    print(f"  Normal tissue slides (purity=0):  {len(normals)}")
    print(f"  Slides still on GDC (orig uuid):  {clean['gdc_exists'].sum()}")
    n_missing_orig = (~clean['gdc_exists']).sum()
    n_resolved_new = clean['file_uuid_new'].notna().sum()
    n_truly_gone = n_missing_orig - n_resolved_new
    print(f"  Slides missing orig uuid:         {n_missing_orig}")
    print(f"  → resolved with new uuid:         {n_resolved_new}")
    print(f"  → truly unresolvable:             {n_truly_gone}")
    print(f"  percent_tumor_nuclei available:   {n_ptn}")

    # Check: any slide linked to multiple aliquots?
    multi_aliquot = has_aliquot.groupby("barcode")["aliquot_barcode"].nunique()
    multi = (multi_aliquot > 1).sum()
    print(f"  Slides linked to >1 aliquot:      {multi}")

    # Match type breakdown
    print(f"\n  Match type breakdown:")
    for mt, cnt in clean["gdc_match_type"].value_counts().items():
        print(f"    {mt:25s} {cnt:>6d}")

    # ── 12. Plots ────────────────────────────────────────────────
    # Slides per aliquot histogram
    if len(has_aliquot) > 0:
        slides_per_aliquot = has_aliquot.groupby("aliquot_barcode").size()
        fig, ax = plt.subplots(figsize=(7, 4))
        bins = np.arange(0.5, slides_per_aliquot.max() + 1.5, 1)
        ax.hist(slides_per_aliquot.values, bins=bins, color="#4c72b0", edgecolor="white", rwidth=0.8)
        for b_val in sorted(slides_per_aliquot.value_counts().index):
            cnt = (slides_per_aliquot == b_val).sum()
            ax.text(b_val, cnt + 20, str(cnt), ha="center", fontsize=9)
        ax.set_title("Number of slides per matched aliquot")
        ax.set_xlabel("Slides per aliquot")
        ax.set_ylabel("Count of aliquots")
        fig.tight_layout()
        hist_path = args.reports_dir / "slides_per_aliquot_wedge.png"
        fig.savefig(hist_path, dpi=150)
        plt.close(fig)
        logger.info("Saved %s", hist_path)

    # Purity distribution
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(has_purity[has_purity["purity"] > 0]["purity"].dropna(), bins=30,
            color="#4c72b0", edgecolor="white", alpha=0.85, label="Tumour")
    ax.axvline(0, color="red", linestyle="--", label=f"Normal (n={len(normals)})")
    ax.set_title("Purity distribution — Wedge MVP dataset")
    ax.set_xlabel("ABSOLUTE purity")
    ax.set_ylabel("Count")
    ax.legend()
    fig.tight_layout()
    pur_path = args.reports_dir / "purity_distribution_wedge.png"
    fig.savefig(pur_path, dpi=150)
    plt.close(fig)
    logger.info("Saved %s", pur_path)

    print("=" * 70)


if __name__ == "__main__":
    main()
