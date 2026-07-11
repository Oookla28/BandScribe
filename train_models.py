from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from bandscribe_ml import (
    DEFAULT_MODEL_PATH,
    KEY_LABELS,
    PROGRESSION_LABELS,
    HarmonyModel,
    extract_features,
    generate_labeled_samples,
)


SEED = 20260711


def _softmax(logits: np.ndarray) -> np.ndarray:
    logits = logits - np.max(logits, axis=1, keepdims=True)
    exp = np.exp(logits)
    return exp / np.sum(exp, axis=1, keepdims=True)


def train_softmax(
    features: np.ndarray,
    labels: np.ndarray,
    classes: int,
    epochs: int = 700,
    learning_rate: float = 0.045,
    l2: float = 2e-4,
) -> tuple[np.ndarray, np.ndarray]:
    """Train multiclass logistic regression with deterministic full-batch Adam."""
    rows, columns = features.shape
    weights = np.zeros((columns, classes), dtype=np.float64)
    bias = np.zeros(classes, dtype=np.float64)
    one_hot = np.eye(classes, dtype=np.float64)[labels]
    mw = np.zeros_like(weights)
    vw = np.zeros_like(weights)
    mb = np.zeros_like(bias)
    vb = np.zeros_like(bias)
    beta1, beta2 = 0.9, 0.999
    for epoch in range(1, epochs + 1):
        probabilities = _softmax(features @ weights + bias)
        error = (probabilities - one_hot) / rows
        grad_w = features.T @ error + l2 * weights
        grad_b = error.sum(axis=0)
        mw = beta1 * mw + (1.0 - beta1) * grad_w
        vw = beta2 * vw + (1.0 - beta2) * grad_w * grad_w
        mb = beta1 * mb + (1.0 - beta1) * grad_b
        vb = beta2 * vb + (1.0 - beta2) * grad_b * grad_b
        mw_hat = mw / (1.0 - beta1**epoch)
        vw_hat = vw / (1.0 - beta2**epoch)
        mb_hat = mb / (1.0 - beta1**epoch)
        vb_hat = vb / (1.0 - beta2**epoch)
        weights -= learning_rate * mw_hat / (np.sqrt(vw_hat) + 1e-8)
        bias -= learning_rate * mb_hat / (np.sqrt(vb_hat) + 1e-8)
    return weights.astype(np.float32), bias.astype(np.float32)


def _accuracy(model: HarmonyModel, features: np.ndarray, key_labels: np.ndarray, progression_labels: np.ndarray) -> dict[str, float]:
    predictions = model.predict_features(features)
    assert isinstance(predictions, list)
    predicted_keys = np.array([item["key_index"] for item in predictions])
    predicted_progressions = np.array([item["progression_index"] for item in predictions])
    return {
        "key_accuracy": float(np.mean(predicted_keys == key_labels)),
        "progression_accuracy": float(np.mean(predicted_progressions == progression_labels)),
        "joint_accuracy": float(np.mean((predicted_keys == key_labels) & (predicted_progressions == progression_labels))),
    }


def train(
    output_path: str | Path = DEFAULT_MODEL_PATH,
    seed: int = SEED,
    test_variations: tuple[int, ...] = (8, 9),
    wav_dir: str | Path | None = None,
) -> dict[str, Any]:
    """Train, evaluate by held-out variation, save the model, and return JSON-safe metrics."""
    np.random.seed(seed)
    examples = generate_labeled_samples(range(10), seed=seed, wav_dir=wav_dir)
    features = np.stack([extract_features(item.samples, item.sample_rate) for item in examples]).astype(np.float64)
    keys = np.array([item.key_index for item in examples], dtype=np.int64)
    progressions = np.array([item.progression_index for item in examples], dtype=np.int64)
    variations = np.array([item.variation for item in examples], dtype=np.int64)
    test_mask = np.isin(variations, np.asarray(test_variations))
    train_mask = ~test_mask
    if not np.any(train_mask) or not np.any(test_mask):
        raise ValueError("variation split must leave at least one train and one test example")

    mean = features[train_mask].mean(axis=0)
    scale = features[train_mask].std(axis=0)
    scale[scale < 1e-6] = 1.0
    normalized = (features - mean) / scale
    key_weights, key_bias = train_softmax(normalized[train_mask], keys[train_mask], len(KEY_LABELS))
    progression_weights, progression_bias = train_softmax(
        normalized[train_mask], progressions[train_mask], len(PROGRESSION_LABELS)
    )
    model = HarmonyModel(
        mean,
        scale,
        key_weights,
        key_bias,
        progression_weights,
        progression_bias,
    )

    target = Path(output_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        target,
        format_version=np.array(1, dtype=np.int64),
        seed=np.array(seed, dtype=np.int64),
        test_variations=np.asarray(test_variations, dtype=np.int64),
        feature_mean=model.feature_mean,
        feature_scale=model.feature_scale,
        key_weights=model.key_weights,
        key_bias=model.key_bias,
        progression_weights=model.progression_weights,
        progression_bias=model.progression_bias,
        key_labels=np.asarray(KEY_LABELS),
        progression_labels=np.asarray(PROGRESSION_LABELS),
    )
    metrics = {
        "model_path": str(target.resolve()),
        "dataset": "synthetic_chord_progressions_v1",
        "scope": "12 major keys and 3 progression classes",
        "seed": seed,
        "train_samples": int(np.sum(train_mask)),
        "test_samples": int(np.sum(test_mask)),
        "train_variations": sorted(set(variations[train_mask].tolist())),
        "test_variations": sorted(set(variations[test_mask].tolist())),
        "train": _accuracy(model, features[train_mask], keys[train_mask], progressions[train_mask]),
        "test": _accuracy(model, features[test_mask], keys[test_mask], progressions[test_mask]),
        "caveat": "Synthetic holdout accuracy is not real-song accuracy.",
    }
    metrics_path = target.with_suffix(".metrics.json")
    metrics_path.write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    metrics["metrics_path"] = str(metrics_path.resolve())
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="Train BandScribe's pure NumPy harmony model")
    parser.add_argument("--output", type=Path, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--wav-dir", type=Path, help="optionally save the labeled synthetic WAV dataset")
    args = parser.parse_args()
    print(json.dumps(train(args.output, args.seed, wav_dir=args.wav_dir), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

