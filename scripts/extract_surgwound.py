#!/usr/bin/env python3
"""
Extract SurgWound HuggingFace dataset into clean per-image JPEG files + pivoted CSV.

Source: xuxuxuxuxu/SurgWound (public VQA dataset, ~697 unique surgical wound images)

Each image appears multiple times (once per clinical attribute). This script:
  1. Downloads all splits via HuggingFace datasets
  2. Filters to VQA rows (task_type == "multi_choice")
  3. Groups by image identity (image_name) and pivots fields into columns
  4. Saves each unique image once as JPEG
  5. Outputs labels.csv + manifest.json

Usage:
    python scripts/extract_surgwound.py --output_dir data/surgwound
    python scripts/extract_surgwound.py --output_dir data/surgwound --limit 50  # smoke test
"""

import argparse
import base64
import hashlib
import json
import logging
import os
import sys
import time
from collections import defaultdict
from io import BytesIO
from pathlib import Path

import pandas as pd
from datasets import load_dataset
from PIL import Image
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Field name → CSV column mapping
# ---------------------------------------------------------------------------
FIELD_TO_COL = {
    "Wound Location": "wound_location",
    "Healing Status": "healing_status",
    "Closure Method": "closure_method",
    "Exudate Type": "exudate_type",
    "Erythema": "erythema",
    "Edema": "edema",
    "Urgency Level": "urgency_level",
    "Infection Risk Assessment": "infection_risk",
}

LABEL_COLS = list(FIELD_TO_COL.values())

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def decode_image(b64_string: str) -> tuple[Image.Image, bytes]:
    """Decode a base64-encoded image string → (PIL Image, raw_bytes).

    The dataset stores images as raw base64 JPEG strings (no data-URI prefix).
    Some entries may have a leading quote character from JSON artefacts — strip it.
    """
    cleaned = b64_string.strip().strip('"').strip("'")
    raw = base64.b64decode(cleaned)
    img = Image.open(BytesIO(raw)).convert("RGB")
    return img, raw


def image_hash(raw_bytes: bytes) -> str:
    """Stable short hash for image identity verification."""
    return hashlib.md5(raw_bytes[:4096]).hexdigest()[:12]


def slugify_field(field_name: str) -> str:
    """Convert field name to snake_case column name."""
    return FIELD_TO_COL.get(field_name, field_name.lower().replace(" ", "_"))


# ---------------------------------------------------------------------------
# Main extraction
# ---------------------------------------------------------------------------

def extract(output_dir: str, limit: int | None = None, hf_token: str | None = None):
    output_path = Path(output_dir)
    images_dir = output_path / "images"
    images_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # 1. Load dataset
    # ------------------------------------------------------------------
    log.info("Loading dataset xuxuxuxuxu/SurgWound ...")
    t0 = time.time()
    kwargs = {}
    if hf_token:
        kwargs["token"] = hf_token
    ds = load_dataset("xuxuxuxuxu/SurgWound", **kwargs)
    log.info("Loaded in %.1fs — splits: %s", time.time() - t0, list(ds.keys()))

    # ------------------------------------------------------------------
    # 2. Iterate rows, collect per-image data
    # ------------------------------------------------------------------
    # image_name → {split, fields: {field: answer}, b64_image (first seen)}
    image_data: dict[str, dict] = {}
    field_audit: dict[str, set] = defaultdict(set)  # field → set of answer values
    task_type_counts: dict[str, int] = defaultdict(int)
    skipped_non_vqa = 0
    decode_errors: list[dict] = []
    total_rows = 0
    rows_processed = 0

    for split_name in ds:
        split_ds = ds[split_name]
        log.info("Processing split '%s' (%d rows) ...", split_name, len(split_ds))

        for idx, row in enumerate(tqdm(split_ds, desc=f"  {split_name}")):
            total_rows += 1

            task_type = row.get("task_type", "unknown")
            task_type_counts[task_type] += 1

            # Skip report generation rows
            if task_type != "multi_choice":
                skipped_non_vqa += 1
                continue

            img_name = row["image_name"]
            field = row["field"]
            answer = row["answer"]

            # Track field audit
            field_audit[field].add(answer)

            # Initialize image entry if first time
            if img_name not in image_data:
                image_data[img_name] = {
                    "split": split_name,
                    "fields": {},
                    "b64_image": row["image"],
                }

            image_data[img_name]["fields"][field] = answer
            rows_processed += 1

            # Early stop for smoke test
            if limit and len(image_data) >= limit:
                # Still need to finish collecting all fields for images we've seen
                # But stop accepting new images
                pass

        # After processing split, if limit reached, stop
        if limit and len(image_data) >= limit:
            # Continue to process remaining splits to get all fields for existing images
            pass

    # If limit is set, trim to only first N images
    if limit:
        all_names = list(image_data.keys())[:limit]
        image_data = {k: image_data[k] for k in all_names}

    log.info(
        "Collected %d unique images from %d VQA rows (%d non-VQA skipped)",
        len(image_data), rows_processed, skipped_non_vqa,
    )

    # ------------------------------------------------------------------
    # 3. Field audit
    # ------------------------------------------------------------------
    log.info("=" * 60)
    log.info("FIELD AUDIT")
    log.info("=" * 60)
    unknown_fields = set()
    for field_name, values in sorted(field_audit.items()):
        col = FIELD_TO_COL.get(field_name)
        status = f"→ {col}" if col else "⚠ UNMAPPED"
        if not col:
            unknown_fields.add(field_name)
        log.info("  %-30s %s", field_name, status)
        for v in sorted(values):
            log.info("      • %s", v)
    if unknown_fields:
        log.warning("Unmapped fields found: %s — add to FIELD_TO_COL!", unknown_fields)
    log.info("=" * 60)

    # ------------------------------------------------------------------
    # 4. Decode images, save as JPEG, build label rows
    # ------------------------------------------------------------------
    label_rows = []
    split_counts = defaultdict(int)
    hash_set = set()

    log.info("Saving images and building label table ...")
    for img_name, info in tqdm(image_data.items(), desc="  saving"):
        img_stem = Path(img_name).stem  # e.g. "606" from "606.jpg"
        img_id = img_stem
        split = info["split"]
        split_counts[split] += 1

        # Decode and save image
        try:
            img, raw_bytes = decode_image(info["b64_image"])
            h = image_hash(raw_bytes)
            hash_set.add(h)
            save_path = images_dir / f"{img_id}.jpg"
            img.save(str(save_path), "JPEG", quality=95)
        except Exception as e:
            decode_errors.append({
                "image_name": img_name,
                "error": str(e),
            })
            log.warning("Decode error for %s: %s", img_name, e)
            continue

        # Build label row
        row_dict = {
            "img_id": img_id,
            "image_path": f"images/{img_id}.jpg",
            "split": split,
        }
        for field_name, col_name in FIELD_TO_COL.items():
            row_dict[col_name] = info["fields"].get(field_name, "MISSING")

        label_rows.append(row_dict)

    # ------------------------------------------------------------------
    # 5. Save labels.csv
    # ------------------------------------------------------------------
    df = pd.DataFrame(label_rows)
    # Ensure column order
    cols = ["img_id", "image_path", "split"] + LABEL_COLS
    for c in cols:
        if c not in df.columns:
            df[c] = "MISSING"
    df = df[cols]
    df = df.sort_values(["split", "img_id"]).reset_index(drop=True)

    csv_path = output_path / "labels.csv"
    df.to_csv(csv_path, index=False)
    log.info("Saved %s (%d rows)", csv_path, len(df))

    # ------------------------------------------------------------------
    # 6. Save manifest.json
    # ------------------------------------------------------------------
    # Label distributions
    label_distributions = {}
    for col in LABEL_COLS:
        if col in df.columns:
            label_distributions[col] = df[col].value_counts().to_dict()

    # Missing counts
    missing_counts = {}
    for col in LABEL_COLS:
        n_missing = (df[col] == "MISSING").sum()
        if n_missing > 0:
            missing_counts[col] = int(n_missing)

    manifest = {
        "dataset": "xuxuxuxuxu/SurgWound",
        "extraction_date": time.strftime("%Y-%m-%d %H:%M:%S"),
        "total_unique_images": len(df),
        "split_counts": {k: int(v) for k, v in split_counts.items()},
        "total_vqa_rows_processed": rows_processed,
        "total_non_vqa_skipped": skipped_non_vqa,
        "task_type_counts": dict(task_type_counts),
        "unique_image_hashes": len(hash_set),
        "fields_found": sorted(field_audit.keys()),
        "field_to_column": FIELD_TO_COL,
        "label_distributions": label_distributions,
        "missing_field_counts": missing_counts,
        "decode_errors": decode_errors,
        "decode_error_count": len(decode_errors),
        "limit_applied": limit,
    }

    manifest_path = output_path / "manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2, default=str)
    log.info("Saved %s", manifest_path)

    # ------------------------------------------------------------------
    # 7. Summary
    # ------------------------------------------------------------------
    log.info("=" * 60)
    log.info("EXTRACTION COMPLETE")
    log.info("=" * 60)
    log.info("  Output dir:      %s", output_path)
    log.info("  Unique images:   %d", len(df))
    log.info("  Split counts:    %s", dict(split_counts))
    log.info("  Decode errors:   %d", len(decode_errors))
    for col in LABEL_COLS:
        n_missing = (df[col] == "MISSING").sum()
        if n_missing > 0:
            log.info("  Missing %-20s %d", col, n_missing)
    log.info("=" * 60)

    return df, manifest


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Extract SurgWound dataset into images + pivoted CSV"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="data/surgwound",
        help="Output directory (default: data/surgwound)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Max number of unique images to extract (for smoke testing)",
    )
    parser.add_argument(
        "--hf_token",
        type=str,
        default=None,
        help="HuggingFace token (optional, dataset is public). "
             "Also reads HF_TOKEN from environment / .env file.",
    )
    args = parser.parse_args()

    # Try to load .env if python-dotenv is available
    try:
        from dotenv import load_dotenv
        load_dotenv()
    except ImportError:
        pass

    hf_token = args.hf_token or os.environ.get("HF_TOKEN")

    extract(
        output_dir=args.output_dir,
        limit=args.limit,
        hf_token=hf_token,
    )


if __name__ == "__main__":
    main()
