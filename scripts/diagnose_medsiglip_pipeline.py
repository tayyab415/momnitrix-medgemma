#!/usr/bin/env python3
"""Diagnostics for MedSigLIP SurgWound fine-tuning pipeline.

Checks:
1) Data integrity (CSV rows, split counts, image existence)
2) Label encoding correctness and MISSING handling
3) pos_weight recomputation from train split
4) Batch-level masking behavior sanity
5) T4-oriented training config suitability (effective batch, steps/epoch)
6) Rough memory sanity estimate for selective unfreezing
"""

from __future__ import annotations

import csv
import math
import random
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple


EXPECTED_SPLITS = {"train": 480, "validation": 69, "test": 137}
EXPECTED_TOTAL = 686

LABEL_NAMES = [
    "healing_status",
    "erythema",
    "edema",
    "infection_risk",
    "urgency",
    "exudate",
]

# Notebook config assumptions for T4
BATCH_SIZE = 4
GRAD_ACCUM = 16
EPOCHS = 5
N_UNFREEZE = 4


@dataclass
class Sample:
    split: str
    image_path: str
    labels: List[float]


def encode_labels(row: Dict[str, str]) -> List[float]:
    labels: List[float] = []

    # 0: healing_status (Not Healed -> 1, Healed -> 0)
    labels.append(1.0 if row["healing_status"] == "Not Healed" else 0.0)

    # 1: erythema (MISSING -> -1)
    if row["erythema"] == "MISSING":
        labels.append(-1.0)
    else:
        labels.append(1.0 if row["erythema"] == "Existent" else 0.0)

    # 2: edema (MISSING -> -1)
    if row["edema"] == "MISSING":
        labels.append(-1.0)
    else:
        labels.append(1.0 if row["edema"] == "Existent" else 0.0)

    # 3: infection_risk (Medium/High -> 1)
    labels.append(1.0 if row["infection_risk"] in ("Medium", "High") else 0.0)

    # 4: urgency (home care -> 0, yellow/red -> 1)
    labels.append(0.0 if row["urgency_level"].startswith("Home Care") else 1.0)

    # 5: exudate (MISSING -> -1, Non-existent -> 0, else -> 1)
    if row["exudate_type"] == "MISSING":
        labels.append(-1.0)
    elif row["exudate_type"] == "Non-existent":
        labels.append(0.0)
    else:
        labels.append(1.0)

    return labels


def load_samples(base_path: Path) -> List[Sample]:
    labels_csv = base_path / "labels.csv"
    rows: List[Sample] = []
    with labels_csv.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(
                Sample(
                    split=row["split"],
                    image_path=row["image_path"],
                    labels=encode_labels(row),
                )
            )
    return rows


def compute_label_stats(samples: List[Sample]) -> Dict[str, Dict[str, int]]:
    stats: Dict[str, Dict[str, int]] = {}
    for idx, name in enumerate(LABEL_NAMES):
        neg = pos = missing = 0
        for sample in samples:
            value = sample.labels[idx]
            if value == -1.0:
                missing += 1
            elif value == 1.0:
                pos += 1
            else:
                neg += 1
        stats[name] = {"neg": neg, "pos": pos, "missing": missing}
    return stats


def print_stats_table(title: str, stats: Dict[str, Dict[str, int]]) -> None:
    print(f"\n{title}")
    print("-" * len(title))
    print(f"{'label':20s} {'neg':>6s} {'pos':>6s} {'miss':>6s} {'pos_weight':>10s}")
    for name in LABEL_NAMES:
        neg = stats[name]["neg"]
        pos = stats[name]["pos"]
        miss = stats[name]["missing"]
        pw = (neg / pos) if pos > 0 else float("inf")
        print(f"{name:20s} {neg:6d} {pos:6d} {miss:6d} {pw:10.2f}")


def mask_batch_valid_entries(batch: List[Sample]) -> int:
    valid = 0
    for sample in batch:
        for value in sample.labels:
            if value >= 0:
                valid += 1
    return valid


def run_masking_sanity(train_samples: List[Sample], trials: int = 1000) -> Tuple[int, int]:
    min_valid = 10**9
    max_valid = -1
    batch_size = BATCH_SIZE

    for _ in range(trials):
        batch = random.sample(train_samples, k=batch_size)
        valid = mask_batch_valid_entries(batch)
        min_valid = min(min_valid, valid)
        max_valid = max(max_valid, valid)

    return min_valid, max_valid


def check_images_exist(base_path: Path, samples: List[Sample]) -> int:
    missing = 0
    for sample in samples:
        if not (base_path / sample.image_path).exists():
            missing += 1
    return missing


def t4_fit_summary() -> None:
    print("\nT4 Config Fit Summary")
    print("---------------------")
    effective_batch = BATCH_SIZE * GRAD_ACCUM
    forward_passes_per_epoch = math.ceil(EXPECTED_SPLITS["train"] / BATCH_SIZE)
    optimizer_steps_per_epoch = math.ceil(
        forward_passes_per_epoch / GRAD_ACCUM)
    total_optimizer_steps = optimizer_steps_per_epoch * EPOCHS

    print(
        f"batch_size={BATCH_SIZE}, grad_accum={GRAD_ACCUM}, effective_batch={effective_batch}")
    print(f"forward_passes/epoch={forward_passes_per_epoch}")
    print(f"optimizer_steps/epoch={optimizer_steps_per_epoch}")
    print(f"total_optimizer_steps (epochs={EPOCHS})={total_optimizer_steps}")
    print(f"unfreeze_last_n_blocks={N_UNFREEZE}")

    # Coarse memory budget guideline for this setup
    print("\nRough VRAM expectation (from selective-unfreeze setup):")
    print("- expected range ~5-7 GB on T4 with fp16 and batch_size=4")
    print("- if OOM: reduce N_UNFREEZE to 2 or batch_size to 2")


def main() -> int:
    base_path = Path(__file__).resolve().parents[1] / "data" / "surgwound"
    print(f"Using dataset path: {base_path}")

    if not base_path.exists():
        print("[ERROR] Dataset path not found")
        return 1

    samples = load_samples(base_path)
    print(f"Loaded rows: {len(samples)}")

    if len(samples) != EXPECTED_TOTAL:
        print(f"[ERROR] Expected {EXPECTED_TOTAL} rows, got {len(samples)}")
        return 1

    split_counts = Counter(sample.split for sample in samples)
    print(f"Split counts: {dict(split_counts)}")

    for split_name, expected in EXPECTED_SPLITS.items():
        got = split_counts.get(split_name, 0)
        if got != expected:
            print(
                f"[ERROR] Split mismatch for {split_name}: expected {expected}, got {got}")
            return 1

    missing_images = check_images_exist(base_path, samples)
    if missing_images > 0:
        print(f"[ERROR] Missing images: {missing_images}")
        return 1
    print("Image existence: OK (all referenced files found)")

    # Split samples
    train_samples = [s for s in samples if s.split == "train"]
    val_samples = [s for s in samples if s.split == "validation"]
    test_samples = [s for s in samples if s.split == "test"]

    # Stats
    train_stats = compute_label_stats(train_samples)
    all_stats = compute_label_stats(samples)
    print_stats_table("Train Label Stats", train_stats)
    print_stats_table("All-Split Label Stats", all_stats)

    # Expected train counts from plan/notebook
    expected_train = {
        "healing_status": {"neg": 282, "pos": 198, "missing": 0},
        "erythema": {"neg": 334, "pos": 129, "missing": 17},
        "edema": {"neg": 328, "pos": 50, "missing": 102},
        "infection_risk": {"neg": 402, "pos": 78, "missing": 0},
        "urgency": {"neg": 423, "pos": 57, "missing": 0},
        "exudate": {"neg": 367, "pos": 70, "missing": 43},
    }

    mismatch = False
    for label in LABEL_NAMES:
        if train_stats[label] != expected_train[label]:
            print(
                f"[ERROR] Train stat mismatch for {label}: got={train_stats[label]}, expected={expected_train[label]}")
            mismatch = True

    if mismatch:
        return 1

    print("\nTrain label counts: OK (match notebook assumptions)")

    # Masking sanity
    min_valid, max_valid = run_masking_sanity(train_samples, trials=1000)
    print("\nMasking sanity over random train batches (size=4, trials=1000)")
    print(
        f"valid label entries per batch: min={min_valid}, max={max_valid}, full_batch=24")
    if min_valid <= 0:
        print("[ERROR] Found batch with zero valid entries")
        return 1

    # Validation/test positive support sanity
    def positive_support(split_name: str, split_samples: List[Sample]) -> None:
        stats = compute_label_stats(split_samples)
        print(f"\n{split_name} positive support:")
        for label in LABEL_NAMES:
            print(
                f"- {label}: pos={stats[label]['pos']}, valid={stats[label]['pos'] + stats[label]['neg']}")

    positive_support("Validation", val_samples)
    positive_support("Test", test_samples)

    t4_fit_summary()

    print("\n[PASS] Diagnostics complete. Pipeline assumptions are internally consistent for T4 config.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
