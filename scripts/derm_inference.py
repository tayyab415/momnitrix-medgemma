from __future__ import annotations

import json
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


@dataclass
class DermArtifacts:
    classifier: Any
    scaler: Any  # StandardScaler | None
    labels: list[str]
    threshold: float
    top_k: int
    embedding_dim: int


def load_artifacts(artifact_dir: str | Path = "artifacts/derm") -> DermArtifacts:
    base = Path(artifact_dir)
    classifier = pickle.loads((base / "derm_classifier.pkl").read_bytes())
    labels = json.loads((base / "derm_labels.json").read_text(encoding="utf-8"))["labels"]
    config = json.loads((base / "derm_config.json").read_text(encoding="utf-8"))

    scaler = None
    scaler_path = base / "derm_scaler.pkl"
    if config.get("scaled") and scaler_path.exists():
        scaler = pickle.loads(scaler_path.read_bytes())

    return DermArtifacts(
        classifier=classifier,
        scaler=scaler,
        labels=labels,
        threshold=float(config.get("threshold", 0.5)),
        top_k=int(config.get("top_k", 5)),
        embedding_dim=int(config.get("embedding_dim", 6144)),
    )


def predict_embedding(
    embedding: np.ndarray,
    artifacts: DermArtifacts,
) -> dict[str, Any]:
    if embedding.ndim == 1:
        embedding = np.expand_dims(embedding, axis=0)

    if embedding.shape[1] != artifacts.embedding_dim:
        raise ValueError(
            f"Expected embedding dim {artifacts.embedding_dim}, got {embedding.shape[1]}"
        )

    if artifacts.scaler is not None:
        embedding = artifacts.scaler.transform(embedding)

    probs = [
        float(artifacts.classifier.estimators_[i].predict_proba(embedding)[:, 1][0])
        for i in range(len(artifacts.labels))
    ]

    scores = {
        label: prob for label, prob in sorted(
            zip(artifacts.labels, probs), key=lambda kv: kv[1], reverse=True
        )
    }
    top_k_items = list(scores.items())[: artifacts.top_k]
    max_confidence = top_k_items[0][1] if top_k_items else 0.0

    return {
        "scores": scores,
        "top_k": [
            {"condition": condition, "score": round(score, 4)}
            for condition, score in top_k_items
        ],
        "max_confidence": round(float(max_confidence), 4),
        "low_confidence": bool(max_confidence < artifacts.threshold),
        "threshold": artifacts.threshold,
    }


if __name__ == "__main__":
    artifacts = load_artifacts("artifacts/derm")
    print(f"Scaler loaded: {artifacts.scaler is not None}")
    sample_embedding = np.zeros((artifacts.embedding_dim,), dtype=np.float32)
    output = predict_embedding(sample_embedding, artifacts)
    print(json.dumps(output, indent=2))
