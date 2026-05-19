"""Method 2 — GDC biospecimen-based slide ↔ ABSOLUTE purity matching.

Uses the GDC Portal REST API (``/cases`` with biospecimen expansion) to
resolve the actual portion each slide belongs to, then finds the DNA
aliquot(s) from the **same portion** or **same sample**.

Priorities:
  1. *same_portion* — slide and aliquot share a GDC portion entity.
  2. *same_sample* — slide and aliquot are in different portions of the
     same sample.

All HTTP interactions go through :func:`requests.post` so tests can
easily mock them.
"""
from __future__ import annotations

import logging
import time
from typing import Any

import pandas as pd
import requests
from tenacity import retry, stop_after_attempt, wait_exponential

from enso_purity.data.tcga_barcode import parse_aliquot_barcode

logger = logging.getLogger(__name__)

GDC_CASES_URL = "https://api.gdc.cancer.gov/cases"
_BATCH_SIZE = 250


# ------------------------------------------------------------------
# API helpers
# ------------------------------------------------------------------
def _build_gdc_cases_payload(
    case_submitter_ids: list[str],
    *,
    size: int = _BATCH_SIZE,
    from_: int = 0,
) -> dict[str, Any]:
    return {
        "filters": {
            "op": "in",
            "content": {
                "field": "submitter_id",
                "value": list(case_submitter_ids),
            },
        },
        "expand": (
            "samples.portions.slides,"
            "samples.portions.analytes.aliquots"
        ),
        "fields": "submitter_id",
        "size": size,
        "from": from_,
    }


@retry(stop=stop_after_attempt(4), wait=wait_exponential(multiplier=2, min=2, max=30))
def _post_gdc(payload: dict) -> dict:
    resp = requests.post(
        GDC_CASES_URL,
        json=payload,
        headers={"Content-Type": "application/json"},
        timeout=120,
    )
    resp.raise_for_status()
    return resp.json()


def fetch_biospecimen_for_cases(
    case_ids: list[str],
    *,
    batch_size: int = _BATCH_SIZE,
) -> list[dict]:
    """Fetch expanded biospecimen for a list of case submitter IDs.

    Batches requests to avoid payload-size limits.
    Returns the combined ``hits`` list.
    """
    all_hits: list[dict] = []
    unique_ids = sorted(set(case_ids))
    n_batches = (len(unique_ids) + batch_size - 1) // batch_size

    for i in range(n_batches):
        batch = unique_ids[i * batch_size : (i + 1) * batch_size]
        logger.info("GDC batch %d/%d  (%d case IDs)", i + 1, n_batches, len(batch))

        offset = 0
        while True:
            payload = _build_gdc_cases_payload(batch, size=batch_size, from_=offset)
            data = _post_gdc(payload)
            hits = data.get("data", {}).get("hits", [])
            all_hits.extend(hits)

            pagination = data.get("data", {}).get("pagination", {})
            total_pages = pagination.get("pages", 1)
            current_page = pagination.get("page", 1)
            if current_page >= total_pages:
                break
            offset += batch_size

        if i < n_batches - 1:
            time.sleep(0.3)

    logger.info("GDC: fetched %d case records total", len(all_hits))
    return all_hits


# ------------------------------------------------------------------
# biospecimen parsing
# ------------------------------------------------------------------
def parse_biospecimen_to_slide_aliquot_map(
    hits: list[dict],
) -> dict[str, dict[str, Any]]:
    """Build a per-slide mapping with same-portion and same-sample aliquots.

    Returns ``{slide_submitter_id: {
        "case_submitter_id": ...,
        "sample_submitter_id": ...,
        "portion_submitter_id": ...,
        "same_portion_aliquots": [aliquot_submitter_id, ...],
        "same_sample_aliquots":  [aliquot_submitter_id, ...],
    }}``

    ``same_sample_aliquots`` only contains aliquots from **other** portions
    (i.e. not duplicating what is already in ``same_portion_aliquots``).
    """
    mapping: dict[str, dict[str, Any]] = {}

    for case in hits:
        case_sub = case.get("submitter_id", "")
        for sample in case.get("samples", []):
            sample_sub = sample.get("submitter_id", "")

            all_aliquots_by_portion: dict[str, list[str]] = {}
            slide_to_portion: dict[str, str] = {}

            for portion in sample.get("portions", []):
                portion_sub = portion.get("submitter_id", "")

                aliquots_in_portion: list[str] = []
                for analyte in portion.get("analytes", []):
                    for aliquot in analyte.get("aliquots", []):
                        aliquots_in_portion.append(aliquot["submitter_id"])
                all_aliquots_by_portion[portion_sub] = aliquots_in_portion

                for slide in portion.get("slides", []):
                    slide_sub = slide.get("submitter_id", "")
                    slide_to_portion[slide_sub] = portion_sub

            all_sample_aliquots = {
                a for als in all_aliquots_by_portion.values() for a in als
            }

            for slide_sub, portion_sub in slide_to_portion.items():
                same_portion = all_aliquots_by_portion.get(portion_sub, [])
                same_sample = sorted(all_sample_aliquots - set(same_portion))

                mapping[slide_sub] = {
                    "case_submitter_id": case_sub,
                    "sample_submitter_id": sample_sub,
                    "portion_submitter_id": portion_sub,
                    "same_portion_aliquots": sorted(same_portion),
                    "same_sample_aliquots": same_sample,
                }

    return mapping


# ------------------------------------------------------------------
# matching
# ------------------------------------------------------------------
def _pick_best_aliquot(
    info: dict[str, Any],
    abs_aliquot_set: set[str],
) -> tuple[str | None, str]:
    """Return (aliquot_id, match_type) preferring same-portion matches."""
    for a in info["same_portion_aliquots"]:
        if a in abs_aliquot_set:
            return a, "same_portion"
    for a in info["same_sample_aliquots"]:
        if a in abs_aliquot_set:
            return a, "same_sample"
    return None, "no_match"


def match_slides_to_purity_via_gdc(
    slides: pd.DataFrame,
    abs_raw: pd.DataFrame,
) -> pd.DataFrame:
    """Join slides to ABSOLUTE purity using GDC biospecimen hierarchy.

    Parameters
    ----------
    slides : pd.DataFrame
        Must already have ``barcode`` and ``case_id`` columns
        (output of :func:`slide_purity_matching.load_and_filter_slides`).
    abs_raw : pd.DataFrame
        Raw ABSOLUTE table (unfiltered).
    """
    from enso_purity.data.slide_purity_matching import load_and_filter_absolute

    absolute = load_and_filter_absolute(abs_raw)
    abs_lookup: dict[str, dict] = {}
    for _, row in absolute.iterrows():
        abs_lookup[row["sample"]] = {
            "purity": row["purity"],
            "ploidy": row["ploidy"],
        }
    abs_aliquot_set = set(abs_lookup.keys())

    case_ids = sorted(slides["case_id"].unique())
    hits = fetch_biospecimen_for_cases(case_ids)
    mapping = parse_biospecimen_to_slide_aliquot_map(hits)

    records: list[dict[str, Any]] = []
    for _, slide_row in slides.iterrows():
        barcode = slide_row["barcode"]
        info = mapping.get(barcode)
        if info is None:
            continue

        aliquot_id, match_type = _pick_best_aliquot(info, abs_aliquot_set)
        if aliquot_id is None:
            continue

        rec = slide_row.to_dict()
        rec["aliquot_barcode"] = aliquot_id
        rec["gdc_match_type"] = match_type
        rec["gdc_portion_submitter_id"] = info["portion_submitter_id"]
        rec["purity"] = abs_lookup[aliquot_id]["purity"]
        rec["ploidy"] = abs_lookup[aliquot_id]["ploidy"]
        records.append(rec)

    merged = pd.DataFrame(records)
    logger.info("GDC matching: %d slide-purity rows", len(merged))
    return merged
