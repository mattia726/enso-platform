"""# Getting More Metadata"""

import os
import re
import json
import csv
import time
import pandas as pd
import gcsfs
import tifffile
from concurrent.futures import ThreadPoolExecutor, wait, FIRST_COMPLETED
from google.colab import auth

# =CONFIG=
BUCKET_NAME = "gdc-tcga-phs000178-open"
MASTER_CSV_NAME = "bucket_physical_scan.csv"
OUTPUT_CSV = "slides_metadata_report.csv"

NUM_THREADS = 24              # start here; higher can throttle + increase RAM
MAX_IN_FLIGHT = NUM_THREADS*2 # keep small to cap RAM
BLOCK_SIZE = 2 * 1024 * 1024  # 2MB; bigger = more RAM per open handle
CACHE_TYPE = "none"           # safest for RAM; if too slow, see note below

auth.authenticate_user()

_kv_pat = re.compile(r"\|\s*([^|=]+?)\s*=\s*([^|]+?)\s*(?=\||$)")

def parse_description_kv(desc: str) -> dict:
    out = {}
    if not desc:
        return out
    for m in _kv_pat.finditer(desc):
        out[m.group(1).strip()] = m.group(2).strip()
    return out

def _safe_float(x):
    try:
        x = str(x).strip().replace(",", ".")
        x = re.sub(r"[^\d.+-eE]", "", x)
        return float(x) if x else None
    except Exception:
        return None

def compute_mpp_from_resolution(page0):
    try:
        xres = page0.tags.get("XResolution")
        yres = page0.tags.get("YResolution")
        runit = page0.tags.get("ResolutionUnit")
        if not (xres and yres and runit):
            return (None, None)

        def rat_to_float(tag):
            v = tag.value
            if isinstance(v, tuple) and len(v) == 2:
                num, den = v
                return float(num) / float(den) if den else None
            return float(v)

        x_ppu = rat_to_float(xres)
        y_ppu = rat_to_float(yres)
        unit = int(runit.value)
        if not x_ppu or not y_ppu:
            return (None, None)

        if unit == 2:      # inches
            unit_um = 25400.0
        elif unit == 3:    # centimeters
            unit_um = 10000.0
        else:
            return (None, None)

        return (unit_um / x_ppu, unit_um / y_ppu)
    except Exception:
        return (None, None)

def parse_base_mpp_mag_vendor(page0):
    desc = page0.description or ""
    kv = parse_description_kv(desc)

    vendor = "unknown"
    if "Aperio" in desc:
        vendor = "aperio"
    elif "Hamamatsu" in desc or "NanoZoomer" in desc:
        vendor = "hamamatsu"

    mpp = _safe_float(kv.get("MPP"))
    mppx = _safe_float(kv.get("MPP-x")) or _safe_float(kv.get("MPP X")) or _safe_float(kv.get("mpp_x"))
    mppy = _safe_float(kv.get("MPP-y")) or _safe_float(kv.get("MPP Y")) or _safe_float(kv.get("mpp_y"))

    if mpp is not None:
        if mppx is None: mppx = mpp
        if mppy is None: mppy = mpp

    mag = _safe_float(kv.get("AppMag")) or _safe_float(kv.get("Magnification"))

    if mppx is None or mppy is None:
        rx, ry = compute_mpp_from_resolution(page0)
        if mppx is None: mppx = rx
        if mppy is None: mppy = ry

    return vendor, mppx, mppy, mag

def looks_like_pyramid_level(w0, h0, wi, hi):
    if wi <= 0 or hi <= 0:
        return False
    if wi >= w0 or hi >= h0:
        return False
    ar0 = w0 / h0
    ari = wi / hi
    return abs(ar0 - ari) / ar0 < 0.02

def get_metadata_stream(fs, file_id, full_path):
    gcs_path = f"{BUCKET_NAME}/{full_path}"
    res = {
        "file_id": file_id,
        "full_path": full_path,
        "status": "OK",
        "error": None,
        "vendor": None,
        "base_width": None,
        "base_height": None,
        "base_mpp_x": None,
        "base_mpp_y": None,
        "base_mag": None,
        "level_count_total_pages": 0,
        "level_count_pyramid": 0,
        "levels_json": None,
        "has_0p25": False,
        "has_0p50": False,
    }

    try:
        with fs.open(gcs_path, "rb", block_size=BLOCK_SIZE, cache_type=CACHE_TYPE) as f:
            with tifffile.TiffFile(f) as tif:
                pages = tif.pages
                res["level_count_total_pages"] = len(pages)
                if not pages:
                    res["status"] = "ERROR"
                    res["error"] = "No TIFF pages"
                    return res

                page0 = pages[0]
                w0 = int(page0.imagewidth)
                h0 = int(page0.imagelength)
                res["base_width"] = w0
                res["base_height"] = h0

                vendor, mppx0, mppy0, mag0 = parse_base_mpp_mag_vendor(page0)
                res["vendor"] = vendor
                res["base_mpp_x"] = mppx0
                res["base_mpp_y"] = mppy0
                res["base_mag"] = mag0

                levels = []
                levels.append({
                    "level": 0, "width": w0, "height": h0,
                    "downsample": 1.0, "mpp_x": mppx0, "mpp_y": mppy0
                })

                pyramid_pages = []
                for i in range(1, len(pages)):
                    pi = pages[i]
                    wi = int(pi.imagewidth)
                    hi = int(pi.imagelength)
                    if looks_like_pyramid_level(w0, h0, wi, hi):
                        pyramid_pages.append((i, wi, hi))

                pyramid_pages.sort(key=lambda t: t[1] * t[2], reverse=True)

                for (i, wi, hi) in pyramid_pages:
                    ds_w = w0 / wi
                    ds_h = h0 / hi
                    ds = 0.5 * (ds_w + ds_h)

                    mx = (mppx0 * ds) if (mppx0 is not None) else None
                    my = (mppy0 * ds) if (mppy0 is not None) else None

                    levels.append({
                        "level": i,
                        "width": wi,
                        "height": hi,
                        "downsample": float(ds),
                        "mpp_x": mx,
                        "mpp_y": my
                    })

                res["level_count_pyramid"] = len(levels)

                def has_target(levels, target, tol=0.03):
                    for L in levels:
                        mx, my = L.get("mpp_x"), L.get("mpp_y")
                        if mx is None or my is None:
                            continue
                        if abs(mx-target)/target <= tol and abs(my-target)/target <= tol:
                            return True
                    return False

                res["has_0p25"] = has_target(levels, 0.25)
                res["has_0p50"] = has_target(levels, 0.50)
                res["levels_json"] = json.dumps(levels, ensure_ascii=False)

    except FileNotFoundError:
        res["status"] = "MISSING"
    except Exception as e:
        res["status"] = "ERROR"
        res["error"] = str(e)

    return res

def iter_rows_csv(path):
    # Low RAM: read only needed columns (and no full DataFrame if you want)
    df = pd.read_csv(path, usecols=["file_id", "full_path"])
    for file_id, full_path in df.itertuples(index=False, name=None):
        yield file_id, full_path

def main():
    fs = gcsfs.GCSFileSystem(anon=False)

    if not os.path.exists(MASTER_CSV_NAME):
        raise FileNotFoundError(f"Missing {MASTER_CSV_NAME} locally (download it first).")

    rows_iter = iter(iter_rows_csv(MASTER_CSV_NAME))

    fieldnames = [
        "file_id","full_path","status","error",
        "vendor","base_width","base_height","base_mpp_x","base_mpp_y","base_mag",
        "level_count_total_pages","level_count_pyramid",
        "has_0p25","has_0p50","levels_json"
    ]

    t0 = time.time()
    completed = 0

    with ThreadPoolExecutor(max_workers=NUM_THREADS) as ex, open(OUTPUT_CSV, "w", newline="") as out:
        writer = csv.DictWriter(out, fieldnames=fieldnames)
        writer.writeheader()

        in_flight = set()

        # prime the queue
        for _ in range(MAX_IN_FLIGHT):
            try:
                file_id, full_path = next(rows_iter)
            except StopIteration:
                break
            in_flight.add(ex.submit(get_metadata_stream, fs, file_id, full_path))

        while in_flight:
            done, in_flight = wait(in_flight, return_when=FIRST_COMPLETED)
            for fut in done:
                r = fut.result()
                writer.writerow(r)
                completed += 1

                if completed % 200 == 0:
                    rate = completed / max(1e-9, (time.time() - t0))
                    print(f"[{completed}] Rate: {rate:.1f} slides/s")

                # push one more task to keep pipeline full
                try:
                    file_id, full_path = next(rows_iter)
                    in_flight.add(ex.submit(get_metadata_stream, fs, file_id, full_path))
                except StopIteration:
                    pass

    print(f"✅ Done. Wrote {completed} rows to {OUTPUT_CSV}")

if __name__ == "__main__":
    main()