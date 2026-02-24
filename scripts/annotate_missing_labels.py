"""
Re-annotate MISSING wound labels using Gemini 3 Pro Preview vision.

Finds all images in labels.csv where erythema, edema, or exudate_type is
"MISSING" and asks Gemini to assess each one from the wound photograph.

Usage:
    python scripts/annotate_missing_labels.py \
        --labels annotated_data/surgwound/labels.csv \
        --images annotated_data/surgwound/images \
        --output annotated_data/surgwound/labels.csv \
        --checkpoint annotated_data/surgwound/annotation_checkpoint.json

The script is fully resumable: checkpoint is saved after each image and
re-loaded on restart. Safe to kill mid-run.

Output: a new CSV with MISSING values filled in where Gemini is confident,
and a separate column per label indicating whether the value was AI-annotated.

Requirements:
    pip install google-genai pillow pandas tqdm python-dotenv
    GEMINI_API_KEY in .env or environment
"""

from __future__ import annotations

import argparse
import json
import os
import time
from collections.abc import Mapping
from dataclasses import dataclass, field, asdict
from io import BytesIO
from pathlib import Path
import re

import pandas as pd
from PIL import Image as PILImage
from tqdm import tqdm

# ── Optional dotenv ────────────────────────────────────────────────────────────
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

# ── Gemini client ──────────────────────────────────────────────────────────────
try:
    from google import genai
    from google.genai import types as genai_types
except ImportError:
    raise ImportError("Install google-genai: pip install google-genai")


# ═══════════════════════════════════════════════════════════════════════════════
# Constants
# ═══════════════════════════════════════════════════════════════════════════════

DEFAULT_GEMINI_MODEL = "gemini-3-pro-preview"
DEFAULT_THINKING_LEVEL = "HIGH"
ACCEPTED_CONFIDENCE = {"HIGH", "MEDIUM"}

# The three labels that contain MISSING values in SurgWound
ANNOTATABLE_LABELS = ["erythema", "edema", "exudate_type"]

# Per-label prompt definitions: what Gemini should look for and valid responses
LABEL_PROMPTS: dict[str, dict] = {
    "erythema": {
        "question": (
            "Is erythema (redness, hyperaemia) present in the wound or periwound skin? "
            "Erythema appears as reddened or pink skin around or within the wound, "
            "distinct from normal skin tone."
        ),
        "valid_values": ["Existent", "Non-existent"],
        "positive": "Existent",
        "negative": "Non-existent",
    },
    "edema": {
        "question": (
            "Is edema (swelling, puffiness) present in the wound or surrounding tissue? "
            "Edema appears as raised, puffy, or swollen tissue that may feel tense. "
            "Look for the wound edges being raised above the skin level."
        ),
        "valid_values": ["Existent", "Non-existent"],
        "positive": "Existent",
        "negative": "Non-existent",
    },
    "exudate_type": {
        "question": (
            "What type of wound exudate (drainage/discharge) is present? "
            "Options:\n"
            "  - Non-existent: wound is dry, no discharge visible\n"
            "  - Serous: thin, clear or pale yellow fluid\n"
            "  - Sanguineous: bloody or blood-tinged drainage\n"
            "  - Seropurulent: cloudy, slightly purulent\n"
            "  - Purulent: thick, opaque, yellow/green/brown pus"
        ),
        "valid_values": ["Non-existent", "Serous", "Sanguineous", "Seropurulent", "Purulent"],
        "positive": None,  # Multi-class — no single positive value
        "negative": "Non-existent",
    },
}

# Shared system prompt context for all assessments
SYSTEM_PROMPT = """You are an expert wound care clinician assessing one surgical wound image.

Critical output contract:
- Return output that matches the provided JSON schema exactly.
- Do NOT add extra keys.
- Do NOT include any prose outside the schema fields.
- Do NOT start with phrases like "Here is the JSON".

Clinical behavior:
- Be conservative: only mark a finding present when visual evidence is clear.
- Use only the allowed enum values provided in the prompt.
"""


# ═══════════════════════════════════════════════════════════════════════════════
# Data structures
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class AnnotationResult:
    img_id: int
    label: str
    original_value: str
    new_value: str
    confidence: str          # "HIGH", "MEDIUM", "LOW", "UNCERTAIN"
    reasoning: str
    is_ai_annotated: bool


@dataclass
class Checkpoint:
    """Tracks completed annotations for resumability."""
    completed: dict[str, AnnotationResult] = field(default_factory=dict)
    # key: f"{img_id}_{label}"

    def key(self, img_id: int, label: str) -> str:
        return f"{img_id}_{label}"

    def has(self, img_id: int, label: str) -> bool:
        return self.key(img_id, label) in self.completed

    def add(self, result: AnnotationResult) -> None:
        self.completed[self.key(result.img_id, result.label)] = result

    def save(self, path: Path) -> None:
        data = {k: asdict(v) for k, v in self.completed.items()}
        path.write_text(json.dumps(data, indent=2))

    @classmethod
    def load(cls, path: Path) -> "Checkpoint":
        if not path.exists():
            return cls()
        data = json.loads(path.read_text())
        completed = {k: AnnotationResult(**v) for k, v in data.items()}
        return cls(completed=completed)


# ═══════════════════════════════════════════════════════════════════════════════
# Gemini annotation
# ═══════════════════════════════════════════════════════════════════════════════

def image_to_jpeg_bytes(img_path: Path, max_size: int = 800) -> bytes:
    """Resize image and encode as JPEG bytes for Gemini API."""
    img = PILImage.open(img_path).convert("RGB")

    # Resize to keep API calls fast and cheap
    w, h = img.size
    if max(w, h) > max_size:
        scale = max_size / max(w, h)
        img = img.resize((int(w * scale), int(h * scale)), PILImage.Resampling.LANCZOS)

    buf = BytesIO()
    img.save(buf, format="JPEG", quality=85)
    return buf.getvalue()


def build_annotation_prompt(label: str, missing_labels_for_image: list[str]) -> str:
    """Build the user-facing prompt for a single label assessment."""
    info = LABEL_PROMPTS[label]
    valid_str = ", ".join(f'"{v}"' for v in info["valid_values"])

    # Mention other missing labels for context (so model knows what else is unknown)
    other_missing = [l for l in missing_labels_for_image if l != label]
    context_note = ""
    if other_missing:
        context_note = f"\n\nNote: The following other assessments are also unknown for this wound: {', '.join(other_missing)}."

    return f"""Assess this wound image.

QUESTION: {info['question']}

Valid response values for "value": {valid_str} (or "UNCERTAIN" if image quality is insufficient){context_note}

Rules:
- If visual evidence is weak, choose value="UNCERTAIN" and confidence="LOW".
- Keep reasoning to one short sentence grounded only in visible image evidence.
- Follow the JSON schema from the API config; do not invent additional fields."""


def _build_generate_config(
    valid_values: list[str],
    include_thinking: bool = True,
) -> genai_types.GenerateContentConfig:
    """Build generation config with high-thinking and strict JSON schema."""
    response_schema = {
        "type": "OBJECT",
        "properties": {
            "value": {
                "type": "STRING",
                "enum": valid_values + ["UNCERTAIN"],
                "description": "Final label decision. Use UNCERTAIN when visual evidence is insufficient.",
            },
            "confidence": {
                "type": "STRING",
                "enum": ["HIGH", "MEDIUM", "LOW"],
                "description": "Confidence for the chosen value.",
            },
            "reasoning": {
                "type": "STRING",
                "description": "One short sentence (max 100 chars) based only on visible image findings.",
                "max_length": 200,
            },
        },
        "required": ["value", "confidence", "reasoning"],
    }

    config_kwargs = {
        "system_instruction": SYSTEM_PROMPT,
        # 2048 tokens: gemini-2.5-pro thinking tokens consume from this budget;
        # 256 was too small — model would exhaust the budget on thinking alone.
        "max_output_tokens": 2048,
        "response_mime_type": "application/json",
        # Use only response_schema (Gemini-native, uppercase types).
        # Do NOT also set response_json_schema — they use different type formats
        # and having both causes conflicting/empty responses.
        "response_schema": response_schema,
    }

    if include_thinking:
        # Best-effort thinking config: different SDK versions expose different fields.
        try:
            config_kwargs["thinking_config"] = genai_types.ThinkingConfig(
                thinking_level=DEFAULT_THINKING_LEVEL
            )
        except Exception:
            pass

    return genai_types.GenerateContentConfig(**config_kwargs)


def call_gemini_with_retry(
    client: genai.Client,
    model_name: str,
    image_bytes: bytes,
    prompt: str,
    valid_values: list[str],
    max_retries: int = 3,
) -> dict | None:
    """Call Gemini with exponential backoff retries."""
    use_thinking = True

    for attempt in range(max_retries):
        try:
            config = _build_generate_config(valid_values, include_thinking=use_thinking)
            response = client.models.generate_content(
                model=model_name,
                contents=genai_types.Content(
                    role="user",
                    parts=[
                        genai_types.Part.from_bytes(
                            data=image_bytes,
                            mime_type="image/jpeg",
                        ),
                        genai_types.Part.from_text(text=prompt),
                    ],
                ),
                config=config,
            )

            # Preferred path: structured parsed output from response schema.
            parsed_output = getattr(response, "parsed", None)
            if parsed_output is not None:
                if isinstance(parsed_output, Mapping):
                    return dict(parsed_output)
                if hasattr(parsed_output, "model_dump"):
                    dumped = parsed_output.model_dump()
                    if isinstance(dumped, Mapping):
                        return dict(dumped)
                if isinstance(parsed_output, str):
                    parsed_candidate = json.loads(parsed_output)
                    if isinstance(parsed_candidate, Mapping):
                        return dict(parsed_candidate)

            candidate_texts: list[str] = []
            if getattr(response, "text", None):
                candidate_texts.append(str(response.text).strip())

            for candidate in getattr(response, "candidates", []) or []:
                content = getattr(candidate, "content", None)
                for part in getattr(content, "parts", []) or []:
                    text_part = getattr(part, "text", None)
                    if text_part:
                        candidate_texts.append(str(text_part).strip())

            raw = "\n".join(t for t in candidate_texts if t).strip()

            # Accept plain JSON, fenced JSON, and prefix/suffix wrappers.
            candidate = raw
            if candidate.startswith("```"):
                lines = candidate.split("\n")
                candidate = "\n".join(
                    line for line in lines
                    if not line.strip().startswith("```")
                ).strip()

            if not candidate.startswith("{"):
                match = re.search(r"\{.*\}", candidate, flags=re.DOTALL)
                if match:
                    candidate = match.group(0)

            parsed = json.loads(candidate)
            return parsed

        except json.JSONDecodeError as e:
            print(f"  [WARN] JSON parse error on attempt {attempt+1}: {e}")
            print(f"  Raw response: {raw[:500]}")
        except Exception as e:
            print(f"  [WARN] API error on attempt {attempt+1}: {e}")
            if "Thinking level is not supported" in str(e) and use_thinking:
                use_thinking = False
                print("  [INFO] Retrying without thinking_config for this model")
                # Don't wait — this is a config mismatch, not a rate limit
                continue

        if attempt < max_retries - 1:
            wait = 2 ** (attempt + 1)
            print(f"  Retrying in {wait}s...")
            time.sleep(wait)

    return None


def annotate_image(
    client: genai.Client,
    model_name: str,
    img_id: int,
    img_path: Path,
    labels_to_annotate: list[str],
) -> list[AnnotationResult]:
    """
    Annotate a single image for all its MISSING labels.

    Makes one Gemini call per label (separate calls = clearer focused prompts).
    """
    results = []

    # Encode image once, reuse for all labels
    image_bytes = image_to_jpeg_bytes(img_path)

    for label in labels_to_annotate:
        info = LABEL_PROMPTS[label]
        prompt = build_annotation_prompt(label, labels_to_annotate)

        parsed = call_gemini_with_retry(
            client,
            model_name,
            image_bytes,
            prompt,
            info["valid_values"],
        )

        if parsed is None:
            print(f"  [ERROR] Failed to annotate img_id={img_id} label={label} — keeping MISSING")
            results.append(AnnotationResult(
                img_id=img_id,
                label=label,
                original_value="MISSING",
                new_value="MISSING",
                confidence="UNCERTAIN",
                reasoning="API call failed",
                is_ai_annotated=False,
            ))
            continue

        raw_value = str(parsed.get("value", "UNCERTAIN")).strip()
        confidence = str(parsed.get("confidence", "LOW")).strip().upper()
        reasoning = str(parsed.get("reasoning", "")).strip()

        # Validate value against allowed values
        valid_values = info["valid_values"] + ["UNCERTAIN"]
        if raw_value not in valid_values:
            # Try case-insensitive match
            matched = next(
                (v for v in valid_values if v.lower() == raw_value.lower()),
                None
            )
            if matched:
                raw_value = matched
            else:
                print(f"  [WARN] img_id={img_id} label={label}: invalid value '{raw_value}', keeping MISSING")
                raw_value = "UNCERTAIN"

        # If uncertain/low-confidence, keep as MISSING (don't inject bad data)
        if raw_value == "UNCERTAIN" or confidence not in ACCEPTED_CONFIDENCE:
            new_value = "MISSING"
        else:
            new_value = raw_value
        is_ai_annotated = new_value != "MISSING"

        results.append(AnnotationResult(
            img_id=img_id,
            label=label,
            original_value="MISSING",
            new_value=new_value,
            confidence=confidence,
            reasoning=reasoning,
            is_ai_annotated=is_ai_annotated,
        ))

    return results


# ═══════════════════════════════════════════════════════════════════════════════
# Main pipeline
# ═══════════════════════════════════════════════════════════════════════════════

def run_annotation(
    labels_csv: Path,
    images_dir: Path,
    output_csv: Path,
    checkpoint_path: Path,
    model_name: str,
    dry_run: bool = False,
    limit: int | None = None,
) -> None:
    """Full annotation pipeline."""

    # ── Load data ──────────────────────────────────────────────────────────────
    df = pd.read_csv(labels_csv)
    print(f"Loaded {len(df)} rows from {labels_csv}")

    # ── Find all MISSING entries ───────────────────────────────────────────────
    tasks: list[tuple[int, list[str]]] = []  # (img_id, [labels_to_annotate])

    for _, row in df.iterrows():
        img_id = int(row["img_id"])
        missing = [
            label for label in ANNOTATABLE_LABELS
            if str(row[label]) == "MISSING"
        ]
        if missing:
            tasks.append((img_id, missing))

    # Deduplicate and sort
    tasks = sorted(tasks, key=lambda x: x[0])

    total_annotations = sum(len(labels) for _, labels in tasks)
    original_missing_counts = {
        label: int((df[label] == "MISSING").sum())
        for label in ANNOTATABLE_LABELS
    }
    print(f"\nImages with at least one MISSING label: {len(tasks)}")
    print(f"Total individual label annotations needed: {total_annotations}")
    print(f"\nBreakdown by label:")
    for label in ANNOTATABLE_LABELS:
        count = sum(1 for _, labels in tasks if label in labels)
        print(f"  {label}: {count}")

    if dry_run:
        print("\n[DRY RUN] Not calling Gemini. Exiting.")
        return

    if limit:
        tasks = tasks[:limit]
        print(f"\nLimited to first {limit} images for this run.")

    active_total_annotations = sum(len(labels) for _, labels in tasks)

    # ── Load checkpoint ────────────────────────────────────────────────────────
    checkpoint = Checkpoint.load(checkpoint_path)
    already_done = sum(
        1 for img_id, labels in tasks
        for label in labels
        if checkpoint.has(img_id, label)
    )
    print(f"\nCheckpoint: {already_done} annotations already completed, resuming...")

    # ── Set up Gemini client ───────────────────────────────────────────────────
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise ValueError(
            "GEMINI_API_KEY not set. Add it to your .env file or environment."
        )
    client = genai.Client(api_key=api_key)
    print(f"✓ Gemini client initialized (model: {model_name}, thinking={DEFAULT_THINKING_LEVEL})")

    # ── Run annotations ────────────────────────────────────────────────────────
    newly_annotated = 0
    failed = 0

    with tqdm(total=max(active_total_annotations - already_done, 0), desc="Annotating") as pbar:
        for img_id, labels_to_annotate in tasks:
            # Filter to only labels not yet done
            remaining = [
                l for l in labels_to_annotate
                if not checkpoint.has(img_id, l)
            ]
            if not remaining:
                continue

            img_path = images_dir / f"{img_id}.jpg"
            if not img_path.exists():
                # Try .png
                img_path = images_dir / f"{img_id}.png"
            if not img_path.exists():
                print(f"  [WARN] Image not found for img_id={img_id}, skipping")
                continue

            results = annotate_image(client, model_name, img_id, img_path, remaining)

            for result in results:
                checkpoint.add(result)
                if result.is_ai_annotated:
                    newly_annotated += 1
                elif result.new_value == "MISSING":
                    failed += 1
                pbar.update(1)

            # Save checkpoint after each image
            checkpoint.save(checkpoint_path)

            # Polite delay to avoid rate limits
            time.sleep(0.3)

    print(f"\n{'='*60}")
    print(f"Annotation complete:")
    print(f"  Successfully annotated: {newly_annotated}")
    print(f"  Kept as MISSING (uncertain/failed): {failed}")
    print(f"{'='*60}")

    # ── Apply annotations to DataFrame ────────────────────────────────────────
    # Add provenance columns
    for label in ANNOTATABLE_LABELS:
        df[f"{label}_source"] = "original"
        df[f"{label}_confidence"] = ""
        df[f"{label}_reason"] = ""

    for key, result in checkpoint.completed.items():
        img_id = result.img_id
        label = result.label
        new_value = result.new_value

        # Strict guard: only update cells that are currently MISSING
        mask = (df["img_id"] == img_id) & (df[label] == "MISSING")
        if mask.sum() == 0:
            continue

        df.loc[mask, label] = new_value
        if result.is_ai_annotated:
            df.loc[mask, f"{label}_source"] = model_name
            df.loc[mask, f"{label}_confidence"] = result.confidence
            df.loc[mask, f"{label}_reason"] = result.reasoning[:500]
        else:
            df.loc[mask, f"{label}_source"] = "kept_missing"
            df.loc[mask, f"{label}_confidence"] = result.confidence
            df.loc[mask, f"{label}_reason"] = result.reasoning[:500]

    # ── Save output ────────────────────────────────────────────────────────────
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_csv, index=False)
    print(f"\n✓ Annotated labels saved to {output_csv}")

    # ── Print summary ──────────────────────────────────────────────────────────
    print(f"\nRemaining MISSING after annotation:")
    for label in ANNOTATABLE_LABELS:
        remaining_missing = (df[label] == "MISSING").sum()
        originally_missing = original_missing_counts[label]
        filled = originally_missing - remaining_missing
        print(f"  {label}: {remaining_missing} still MISSING "
              f"({filled}/{originally_missing} filled = {100*filled/max(originally_missing,1):.0f}%)")

    # Print confidence distribution
    print(f"\nConfidence distribution of AI annotations:")
    all_results = list(checkpoint.completed.values())
    ai_results = [r for r in all_results if r.is_ai_annotated]
    for conf in ["HIGH", "MEDIUM", "LOW"]:
        count = sum(1 for r in ai_results if r.confidence == conf)
        print(f"  {conf}: {count}")

    # ── Print sample annotations for spot-checking ───────────────────────────
    print(f"\nSample annotations (first 5 per label for spot-check):")
    for label in ANNOTATABLE_LABELS:
        label_results = [r for r in all_results if r.label == label and r.is_ai_annotated][:5]
        if not label_results:
            continue
        print(f"\n  {label}:")
        for r in label_results:
            print(f"    img_id={r.img_id}: {r.new_value} ({r.confidence}) — {r.reasoning[:80]}")


# ═══════════════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════════════

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Re-annotate MISSING wound labels using Gemini vision"
    )
    parser.add_argument(
        "--labels",
        type=Path,
        default=Path("annotated_data/surgwound/labels.csv"),
        help="Path to input labels.csv",
    )
    parser.add_argument(
        "--images",
        type=Path,
        default=Path("annotated_data/surgwound/images"),
        help="Directory containing wound images (named {img_id}.jpg)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("annotated_data/surgwound/labels.csv"),
        help="Where to write the annotated CSV",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path("annotated_data/surgwound/annotation_checkpoint.json"),
        help="Checkpoint file for resumability",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=DEFAULT_GEMINI_MODEL,
        help="Gemini model name (default: gemini-3-pro-preview)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print what would be annotated without calling Gemini",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit to first N images (for testing)",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_annotation(
        labels_csv=args.labels,
        images_dir=args.images,
        output_csv=args.output,
        checkpoint_path=args.checkpoint,
        model_name=args.model,
        dry_run=args.dry_run,
        limit=args.limit,
    )
