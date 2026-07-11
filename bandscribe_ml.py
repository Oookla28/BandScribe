from __future__ import annotations

import wave
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np


SAMPLE_RATE = 8000
KEY_LABELS = ("C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B")
PROGRESSION_LABELS = ("I-V-vi-IV", "vi-IV-I-V", "I-bVII-IV-I")
PROGRESSION_CHORDS = (
    ((0, "major"), (7, "major"), (9, "minor"), (5, "major")),
    ((9, "minor"), (5, "major"), (0, "major"), (7, "major")),
    ((0, "major"), (10, "major"), (5, "major"), (0, "major")),
)
DEFAULT_MODEL_PATH = Path(__file__).resolve().parent / "models" / "harmony_model.npz"


@dataclass(frozen=True)
class LabeledAudio:
    samples: np.ndarray
    sample_rate: int
    key_index: int
    progression_index: int
    variation: int

    @property
    def key(self) -> str:
        return KEY_LABELS[self.key_index]

    @property
    def progression(self) -> str:
        return PROGRESSION_LABELS[self.progression_index]


def _variation_rng(seed: int, key_index: int, progression_index: int, variation: int) -> np.random.Generator:
    stream = seed + variation * 1009 + key_index * 9176 + progression_index * 65537
    return np.random.default_rng(stream)


def synthesize_progression(
    key_index: int,
    progression_index: int,
    variation: int = 0,
    sample_rate: int = SAMPLE_RATE,
    seed: int = 20260711,
) -> np.ndarray:
    """Synthesize one four-chord labeled example as a mono float32 array."""
    if not 0 <= key_index < len(KEY_LABELS):
        raise ValueError("key_index must be in [0, 11]")
    if not 0 <= progression_index < len(PROGRESSION_LABELS):
        raise ValueError("progression_index must be in [0, 2]")
    rng = _variation_rng(seed, key_index, progression_index, variation)
    chord_seconds = 0.42 + 0.025 * (variation % 5)
    count = int(round(chord_seconds * sample_rate))
    t = np.arange(count, dtype=np.float64) / sample_rate
    audio_parts: list[np.ndarray] = []
    inversion = variation % 3
    detune = rng.uniform(-5.0, 5.0)
    harmonics = 2 + variation % 3

    for chord_position, (root_offset, quality) in enumerate(PROGRESSION_CHORDS[progression_index]):
        root_pc = (key_index + root_offset) % 12
        intervals = (0, 4, 7) if quality == "major" else (0, 3, 7)
        midi_notes = [48 + root_pc + interval for interval in intervals]
        midi_notes = [note + (12 if i < inversion else 0) for i, note in enumerate(midi_notes)]
        chord = np.zeros(count, dtype=np.float64)
        for note_index, midi_note in enumerate(midi_notes):
            cents = detune + rng.normal(0.0, 1.2)
            frequency = 440.0 * 2.0 ** ((midi_note - 69 + cents / 100.0) / 12.0)
            phase = rng.uniform(0.0, 2.0 * np.pi)
            note_gain = rng.uniform(0.75, 1.15) / len(midi_notes)
            for harmonic in range(1, harmonics + 1):
                chord += note_gain * np.sin(2.0 * np.pi * frequency * harmonic * t + phase) / harmonic
        attack = np.minimum(1.0, t / 0.018)
        release = np.minimum(1.0, (chord_seconds - t) / 0.035)
        envelope = attack * np.maximum(0.0, release) * np.exp(-0.25 * t)
        pulse = 0.93 + 0.07 * np.sin(2.0 * np.pi * (2.0 + variation % 2) * t + chord_position)
        chord = chord * envelope * pulse
        chord += rng.normal(0.0, 0.004 + 0.001 * (variation % 4), count)
        audio_parts.append(chord)

    audio = np.concatenate(audio_parts)
    peak = float(np.max(np.abs(audio)))
    if peak > 0.0:
        audio = 0.82 * audio / peak
    return audio.astype(np.float32)


def write_wav(path: str | Path, samples: np.ndarray, sample_rate: int = SAMPLE_RATE) -> Path:
    """Write a mono array as a standard PCM16 WAV file."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    mono = _as_mono(samples)
    pcm = (np.clip(mono, -1.0, 1.0) * 32767.0).astype("<i2")
    with wave.open(str(target), "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(int(sample_rate))
        wav_file.writeframes(pcm.tobytes())
    return target


def load_wav(path: str | Path, max_seconds: float = 120.0) -> tuple[np.ndarray, int]:
    """Load 8/16/32-bit PCM WAV audio and mix channels to mono."""
    with wave.open(str(path), "rb") as wav_file:
        channels = wav_file.getnchannels()
        width = wav_file.getsampwidth()
        sample_rate = wav_file.getframerate()
        frames = min(wav_file.getnframes(), int(max_seconds * sample_rate))
        raw = wav_file.readframes(frames)
    if width == 1:
        data = (np.frombuffer(raw, dtype=np.uint8).astype(np.float32) - 128.0) / 128.0
    elif width == 2:
        data = np.frombuffer(raw, dtype="<i2").astype(np.float32) / 32768.0
    elif width == 4:
        data = np.frombuffer(raw, dtype="<i4").astype(np.float32) / 2147483648.0
    else:
        raise ValueError(f"unsupported WAV sample width: {width * 8} bits")
    if channels > 1:
        data = data[: len(data) - len(data) % channels].reshape(-1, channels).mean(axis=1)
    return data.astype(np.float32), sample_rate


def generate_labeled_samples(
    variations: Iterable[int] = range(10),
    sample_rate: int = SAMPLE_RATE,
    seed: int = 20260711,
    wav_dir: str | Path | None = None,
) -> list[LabeledAudio]:
    """Build the balanced synthetic dataset, optionally mirroring every array to WAV."""
    examples: list[LabeledAudio] = []
    output = Path(wav_dir) if wav_dir is not None else None
    for variation in variations:
        for key_index in range(len(KEY_LABELS)):
            for progression_index in range(len(PROGRESSION_LABELS)):
                samples = synthesize_progression(key_index, progression_index, variation, sample_rate, seed)
                example = LabeledAudio(samples, sample_rate, key_index, progression_index, variation)
                examples.append(example)
                if output is not None:
                    filename = f"v{variation:02d}_{KEY_LABELS[key_index].replace('#', 's')}_p{progression_index}.wav"
                    write_wav(output / filename, samples, sample_rate)
    return examples


def _as_mono(samples: np.ndarray) -> np.ndarray:
    audio = np.asarray(samples, dtype=np.float32)
    if audio.ndim == 2:
        audio = audio.mean(axis=1)
    if audio.ndim != 1:
        raise ValueError("samples must be a mono vector or a [frames, channels] array")
    return np.nan_to_num(audio, copy=False)


def segmented_chroma(samples: np.ndarray, sample_rate: int, segments: int = 4) -> np.ndarray:
    """Return L1-normalized FFT chroma for equal-length temporal segments."""
    audio = _as_mono(samples)
    if len(audio) < segments * 128:
        raise ValueError("audio is too short for segmented chroma extraction")
    boundaries = np.linspace(0, len(audio), segments + 1, dtype=int)
    result = np.zeros((segments, 12), dtype=np.float64)
    for segment_index in range(segments):
        part = audio[boundaries[segment_index] : boundaries[segment_index + 1]].astype(np.float64)
        part -= np.mean(part)
        spectrum = np.abs(np.fft.rfft(part * np.hanning(len(part))))
        frequencies = np.fft.rfftfreq(len(part), 1.0 / sample_rate)
        useful = (frequencies >= 55.0) & (frequencies <= min(2400.0, sample_rate * 0.48))
        frequencies = frequencies[useful]
        magnitudes = np.sqrt(spectrum[useful])
        midi = np.rint(69.0 + 12.0 * np.log2(frequencies / 440.0)).astype(int)
        np.add.at(result[segment_index], midi % 12, magnitudes)
        total = result[segment_index].sum()
        if total > 0.0:
            result[segment_index] /= total
    return result.astype(np.float32)


def extract_features(samples: np.ndarray, sample_rate: int) -> np.ndarray:
    """Create absolute chroma plus key-invariant ordered transition features."""
    chroma = segmented_chroma(samples, sample_rate, segments=4).astype(np.float64)
    global_chroma = chroma.mean(axis=0)
    transitions = []
    for index in range(4):
        left = chroma[index]
        right = chroma[(index + 1) % 4]
        correlation = np.array([np.dot(left, np.roll(right, shift)) for shift in range(12)])
        correlation /= correlation.sum() + 1e-12
        transitions.append(correlation)
    return np.concatenate((chroma.ravel(), global_chroma, np.concatenate(transitions))).astype(np.float32)


def _softmax(logits: np.ndarray) -> np.ndarray:
    shifted = logits - np.max(logits, axis=-1, keepdims=True)
    exp = np.exp(shifted)
    return exp / np.sum(exp, axis=-1, keepdims=True)


class HarmonyModel:
    """Small two-head linear softmax model with bundled preprocessing metadata."""

    def __init__(
        self,
        feature_mean: np.ndarray,
        feature_scale: np.ndarray,
        key_weights: np.ndarray,
        key_bias: np.ndarray,
        progression_weights: np.ndarray,
        progression_bias: np.ndarray,
        key_labels: tuple[str, ...] = KEY_LABELS,
        progression_labels: tuple[str, ...] = PROGRESSION_LABELS,
    ) -> None:
        self.feature_mean = np.asarray(feature_mean, dtype=np.float32)
        self.feature_scale = np.asarray(feature_scale, dtype=np.float32)
        self.key_weights = np.asarray(key_weights, dtype=np.float32)
        self.key_bias = np.asarray(key_bias, dtype=np.float32)
        self.progression_weights = np.asarray(progression_weights, dtype=np.float32)
        self.progression_bias = np.asarray(progression_bias, dtype=np.float32)
        self.key_labels = tuple(key_labels)
        self.progression_labels = tuple(progression_labels)

    def predict_features(self, features: np.ndarray) -> dict[str, Any] | list[dict[str, Any]]:
        matrix = np.asarray(features, dtype=np.float32)
        single = matrix.ndim == 1
        if single:
            matrix = matrix[None, :]
        normalized = (matrix - self.feature_mean) / self.feature_scale
        key_probabilities = _softmax(normalized @ self.key_weights + self.key_bias)
        progression_probabilities = _softmax(normalized @ self.progression_weights + self.progression_bias)
        results = []
        for key_probs, progression_probs in zip(key_probabilities, progression_probabilities):
            key_index = int(np.argmax(key_probs))
            progression_index = int(np.argmax(progression_probs))
            results.append(
                {
                    "key": self.key_labels[key_index],
                    "key_index": key_index,
                    "key_confidence": float(key_probs[key_index]),
                    "progression": self.progression_labels[progression_index],
                    "progression_index": progression_index,
                    "progression_confidence": float(progression_probs[progression_index]),
                    "key_probabilities": {label: float(value) for label, value in zip(self.key_labels, key_probs)},
                    "progression_probabilities": {
                        label: float(value) for label, value in zip(self.progression_labels, progression_probs)
                    },
                }
            )
        return results[0] if single else results

    def predict(self, samples: np.ndarray, sample_rate: int) -> dict[str, Any]:
        return self.predict_features(extract_features(samples, sample_rate))  # type: ignore[return-value]


def load_model(path: str | Path = DEFAULT_MODEL_PATH) -> HarmonyModel:
    """Load a saved model without pickle or third-party ML dependencies."""
    with np.load(Path(path), allow_pickle=False) as archive:
        return HarmonyModel(
            archive["feature_mean"],
            archive["feature_scale"],
            archive["key_weights"],
            archive["key_bias"],
            archive["progression_weights"],
            archive["progression_bias"],
            tuple(str(value) for value in archive["key_labels"].tolist()),
            tuple(str(value) for value in archive["progression_labels"].tolist()),
        )


def predict_harmony(
    audio: str | Path | np.ndarray,
    sample_rate: int | None = None,
    model: HarmonyModel | None = None,
    model_path: str | Path = DEFAULT_MODEL_PATH,
) -> dict[str, Any]:
    """Convenience API for bandscribe_core: predict from a WAV path or audio array."""
    if isinstance(audio, (str, Path)):
        samples, actual_rate = load_wav(audio)
    else:
        if sample_rate is None:
            raise ValueError("sample_rate is required when audio is an array")
        samples, actual_rate = audio, sample_rate
    predictor = model if model is not None else load_model(model_path)
    return predictor.predict(samples, actual_rate)


predict = predict_harmony


