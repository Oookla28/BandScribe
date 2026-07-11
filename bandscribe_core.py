from __future__ import annotations

import hashlib
import importlib.util
import json
import math
import re
import shutil
import struct
import subprocess
import sys
import wave
from datetime import datetime, timezone
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from bandscribe_ml import DEFAULT_MODEL_PATH, predict_harmony


TPQ = 480
SAMPLE_RATE = 22050
CHROMATIC = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
MAJOR_SCALE = [0, 2, 4, 5, 7, 9, 11]
PENTATONIC = [0, 2, 4, 7, 9, 12]
ROMAN_DEGREES = {
    "I": (0, "major"),
    "ii": (1, "minor"),
    "iii": (2, "minor"),
    "IV": (3, "major"),
    "V": (4, "major"),
    "vi": (5, "minor"),
    "bVII": (6, "major"),
}


@dataclass
class AudioMetrics:
    duration_seconds: float
    bpm: int
    key: str
    energy: float
    density: float
    source: str
    warnings: list[str] = field(default_factory=list)


@dataclass
class SeparationResult:
    mode: str
    stems: dict[str, str]
    log: str


def ensure_dirs(root: Path) -> dict[str, Path]:
    dirs = {
        "uploads": root / "outputs" / "uploads",
        "jobs": root / "outputs" / "jobs",
        "previews": root / "outputs" / "previews",
    }
    for folder in dirs.values():
        folder.mkdir(parents=True, exist_ok=True)
    return dirs


def safe_name(name: str) -> str:
    stem = Path(name).stem or "audio"
    stem = re.sub(r"[^0-9A-Za-z\u4e00-\u9fff._-]+", "_", stem).strip("._-")
    return stem[:36] or "audio"


def make_job_id(filename: str, data: bytes) -> str:
    digest = hashlib.sha1(data).hexdigest()[:10]
    return f"{safe_name(filename)}-{digest}"


def save_upload(root: Path, filename: str, data: bytes) -> tuple[str, str]:
    dirs = ensure_dirs(root)
    job_id = make_job_id(filename, data)
    suffix = Path(filename).suffix.lower() or ".audio"
    upload_dir = dirs["uploads"] / job_id
    upload_dir.mkdir(parents=True, exist_ok=True)
    upload_path = upload_dir / f"original{suffix}"
    upload_path.write_bytes(data)
    return job_id, str(upload_path)


def demucs_is_available() -> bool:
    return bool(shutil.which("demucs") or importlib.util.find_spec("demucs"))


def run_demucs(upload_path: Path, job_dir: Path, use_demucs: bool = True) -> SeparationResult:
    fallback = {
        "drums": str(upload_path),
        "rhythm_guitar": str(upload_path),
        "lead_or_keys": str(upload_path),
        "full_mix": str(upload_path),
    }
    if not use_demucs:
        return SeparationResult("demo_fallback", fallback, "已跳过 demucs，使用原始音频作为所有试听源。")

    if not demucs_is_available():
        return SeparationResult(
            "missing_demucs",
            fallback,
            "没有检测到 demucs。先用原始音频占位；安装 demucs 后会自动调用真实分离。",
        )

    out_dir = job_dir / "separated"
    out_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable,
        "-m",
        "demucs",
        "-n",
        "htdemucs_6s",
        "--out",
        str(out_dir),
        str(upload_path),
    ]
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=20 * 60, check=False)
    except Exception as exc:
        return SeparationResult("demucs_error", fallback, f"demucs 启动失败：{exc}")

    log = (proc.stdout + "\n" + proc.stderr).strip()
    if proc.returncode != 0:
        return SeparationResult(
            "demucs_failed",
            fallback,
            f"demucs 返回码 {proc.returncode}。已降级为原始音频占位。\n{log[-1800:]}",
        )

    model_dirs = list(out_dir.glob(f"*/{upload_path.stem}"))
    if not model_dirs:
        return SeparationResult("demucs_no_output", fallback, f"demucs 完成但未找到输出目录。\n{log[-1800:]}")

    wavs = {path.stem.lower(): str(path) for path in model_dirs[0].glob("*.wav")}
    stems = {
        "drums": wavs.get("drums", str(upload_path)),
        "rhythm_guitar": wavs.get("guitar") or wavs.get("other") or str(upload_path),
        "lead_or_keys": wavs.get("piano") or wavs.get("other") or wavs.get("guitar") or str(upload_path),
        "bass": wavs.get("bass", str(upload_path)),
        "vocals": wavs.get("vocals", str(upload_path)),
        "full_mix": str(upload_path),
    }
    return SeparationResult("demucs_htdemucs_6s", stems, log[-1800:] or "demucs 分离完成。")


def load_wav_mono(path: Path, max_seconds: int = 120) -> tuple[np.ndarray, int, float]:
    with wave.open(str(path), "rb") as wf:
        channels = wf.getnchannels()
        sample_width = wf.getsampwidth()
        sample_rate = wf.getframerate()
        total_frames = wf.getnframes()
        frames_to_read = min(total_frames, max_seconds * sample_rate)
        raw = wf.readframes(frames_to_read)

    if sample_width == 1:
        data = np.frombuffer(raw, dtype=np.uint8).astype(np.float32)
        data = (data - 128.0) / 128.0
    elif sample_width == 2:
        data = np.frombuffer(raw, dtype="<i2").astype(np.float32) / 32768.0
    elif sample_width == 4:
        data = np.frombuffer(raw, dtype="<i4").astype(np.float32) / 2147483648.0
    else:
        raise ValueError(f"暂不支持 {sample_width * 8}-bit WAV")

    if channels > 1:
        data = data.reshape(-1, channels).mean(axis=1)
    duration = total_frames / float(sample_rate)
    return data, sample_rate, duration


def estimate_metrics(path: Path) -> AudioMetrics:
    warnings: list[str] = []
    try:
        samples, sample_rate, duration = load_wav_mono(path)
        if len(samples) < sample_rate:
            raise ValueError("音频太短，启用默认估计")

        frame = 1024
        usable = samples[: len(samples) - (len(samples) % frame)]
        frames = usable.reshape(-1, frame)
        rms = np.sqrt(np.mean(frames * frames, axis=1))
        energy = float(np.clip(np.mean(np.abs(samples)) * 3.0, 0.02, 1.0))
        threshold = float(np.mean(rms) + np.std(rms) * 0.8)
        peak_idx = np.where(rms > threshold)[0]
        peak_times = peak_idx * frame / float(sample_rate)
        peak_times = _thin_peaks(peak_times, min_gap=0.18)
        density = float(len(peak_times) / max(duration, 1.0))
        bpm = _estimate_bpm(peak_times)
        zcr = float(np.mean(samples[1:] * samples[:-1] < 0.0))
        key = _pick_key(path.name, energy, zcr)
        return AudioMetrics(duration, bpm, key, energy, density, "WAV 包络估计", warnings)
    except Exception as exc:
        size_mb = path.stat().st_size / 1_000_000 if path.exists() else 3.0
        duration = float(np.clip(size_mb * 14.0, 32.0, 210.0))
        warnings.append(f"无法直接解析音频波形：{exc}；启用文件大小/文件名启发式。")
        return AudioMetrics(duration, 120, _pick_key(path.name, 0.35, 0.08), 0.35, 1.2, "启发式默认值", warnings)


def _thin_peaks(times: np.ndarray, min_gap: float) -> np.ndarray:
    if len(times) == 0:
        return times
    kept = [float(times[0])]
    for item in times[1:]:
        if float(item) - kept[-1] >= min_gap:
            kept.append(float(item))
    return np.array(kept, dtype=np.float32)


def _estimate_bpm(peak_times: np.ndarray) -> int:
    if len(peak_times) < 4:
        return 120
    intervals = np.diff(peak_times)
    intervals = intervals[(intervals > 0.25) & (intervals < 1.2)]
    if len(intervals) == 0:
        return 120
    bpm = 60.0 / float(np.median(intervals))
    while bpm < 75:
        bpm *= 2
    while bpm > 180:
        bpm /= 2
    return int(np.clip(round(bpm), 76, 176))


def _pick_key(filename: str, energy: float, zcr: float) -> str:
    lower = filename.lower()
    for key in ["c#", "f#", "bb", "eb", "ab", "c", "d", "e", "f", "g", "a", "b"]:
        if re.search(rf"(^|[^a-z]){re.escape(key)}([^a-z]|$)", lower):
            return key.upper().replace("BB", "A#").replace("EB", "D#").replace("AB", "G#")
    index = int((hashlib.sha1(filename.encode("utf-8")).digest()[0] + energy * 10 + zcr * 100) % 6)
    return ["C", "G", "D", "A", "E", "F"][index]


def generate_drum_part(metrics: AudioMetrics, bars: int = 8) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    notes: list[tuple[float, float, int, int]] = []
    busy = metrics.density > 1.8 or metrics.energy > 0.5
    for bar in range(1, bars + 1):
        grid = ["."] * 16
        beat0 = (bar - 1) * 4.0
        for step in range(0, 16, 2):
            grid[step] = "x"
            notes.append((beat0 + step / 4.0, 0.12, 42, 58))
        for step in [0, 8]:
            grid[step] = "K"
            notes.append((beat0 + step / 4.0, 0.18, 36, 100))
        if busy and bar % 2 == 0:
            grid[10] = "k"
            notes.append((beat0 + 2.5, 0.14, 36, 76))
        for step in [4, 12]:
            grid[step] = "S"
            notes.append((beat0 + step / 4.0, 0.16, 38, 96))
        if busy and bar % 4 == 0:
            grid[15] = "s"
            notes.append((beat0 + 3.75, 0.10, 38, 60))
        rows.append(
            {
                "小节": bar,
                "计数": "1e&a 2e&a 3e&a 4e&a",
                "鼓谱": "".join(grid),
                "说明": "K/k=底鼓, S/s=军鼓, x=踩镲",
            }
        )
    return {"rows": rows, "notes": notes}


def generate_chord_part(metrics: AudioMetrics, bars: int = 8, progression: str | None = None) -> dict[str, Any]:
    if progression == "I-bVII-IV-I":
        romans = ["I", "bVII", "IV", "I"]
        label = "model I-bVII-IV-I"
    elif progression == "vi-IV-I-V":
        romans = ["vi", "IV", "I", "V"]
        label = "model vi-IV-I-V"
    elif progression == "I-V-vi-IV":
        romans = ["I", "V", "vi", "IV"]
        label = "model I-V-vi-IV"
    elif metrics.energy > 0.62:
        romans = ["I", "bVII", "IV", "I"]
        label = "rock I-bVII-IV-I"
    elif metrics.density > 2.2:
        romans = ["vi", "IV", "I", "V"]
        label = "dense vi-IV-I-V"
    else:
        romans = ["I", "V", "vi", "IV"]
        label = "default I-V-vi-IV"

    rows: list[dict[str, Any]] = []
    notes: list[tuple[float, float, int, int]] = []
    for bar in range(1, bars + 1):
        roman = romans[(bar - 1) % len(romans)]
        chord_name, chord_pitches = chord_from_roman(metrics.key, roman)
        beat0 = (bar - 1) * 4.0
        for offset, duration, velocity in [(0.0, 1.15, 86), (1.5, 0.45, 72), (2.0, 0.80, 78), (3.0, 0.65, 76)]:
            for pitch in chord_pitches:
                notes.append((beat0 + offset, duration, pitch, velocity))
        rows.append(
            {
                "小节": bar,
                "和弦": chord_name,
                "级数": roman,
                "扫弦": "D - D U - U D U",
                "用途": "节奏吉他/键盘铺底",
            }
        )
    return {"rows": rows, "notes": notes, "label": label}


def predict_harmony_with_fallback(path: Path, metrics: AudioMetrics, root: Path) -> dict[str, Any]:
    result: dict[str, Any] = {
        "used": False,
        "source": "heuristic",
        "key": metrics.key,
        "key_confidence": 0.0,
        "progression": None,
        "progression_confidence": 0.0,
        "model_path": str(DEFAULT_MODEL_PATH),
        "warning": "",
    }
    model_path = root / "models" / "harmony_model.npz"
    result["model_path"] = str(model_path)
    if not model_path.exists():
        result["warning"] = "和声模型尚未训练，继续使用启发式。运行 train_models.py 可生成模型。"
        return result
    try:
        prediction = predict_harmony(path, model_path=model_path)
        result.update(prediction)
        result["source"] = "synthetic_harmony_model_v1"
        key_ok = float(prediction["key_confidence"]) >= 0.35
        progression_ok = float(prediction["progression_confidence"]) >= 0.45
        if key_ok:
            metrics.key = str(prediction["key"])
        else:
            result["warning"] = "模型调性置信度偏低，保留启发式调性。"
        if not progression_ok:
            result["progression"] = None
            result["warning"] = (result["warning"] + " 模型和弦进行置信度偏低，使用规则模板。").strip()
        result["used"] = key_ok or progression_ok
    except Exception as exc:
        result["warning"] = f"和声模型推理失败，已回退启发式：{exc}"
    return result


def record_harmony_feedback(
    root: Path,
    result: dict[str, Any],
    corrected_key: str,
    corrected_progression: str,
) -> Path:
    feedback_dir = root / "outputs" / "feedback"
    feedback_dir.mkdir(parents=True, exist_ok=True)
    target = feedback_dir / "harmony_feedback.jsonl"
    record = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "job_id": result["job_id"],
        "upload_path": result["upload_path"],
        "predicted_key": result["metrics"]["key"],
        "predicted_progression": result["harmony_model"].get("progression"),
        "corrected_key": corrected_key,
        "corrected_progression": corrected_progression,
        "model_source": result["harmony_model"].get("source"),
    }
    with target.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    return target

def generate_melody_part(metrics: AudioMetrics, bars: int = 8) -> dict[str, Any]:
    key_pc = note_to_pc(metrics.key)
    scale = [60 + key_pc + interval for interval in PENTATONIC]
    while scale[0] > 64:
        scale = [pitch - 12 for pitch in scale]
    while scale[0] < 55:
        scale = [pitch + 12 for pitch in scale]

    rows: list[dict[str, Any]] = []
    notes: list[tuple[float, float, int, int]] = []
    pattern = [0, 2, 4, 2, 1, 0, 3, 4, 5, 4, 2, 1]
    offsets = [0.0, 1.5, 2.5]
    durations = [0.9, 0.65, 1.0]
    item = 0
    for bar in range(1, bars + 1):
        for slot, offset in enumerate(offsets):
            degree = pattern[item % len(pattern)]
            pitch = scale[degree % len(scale)]
            beat = (bar - 1) * 4.0 + offset
            duration = durations[slot]
            notes.append((beat, duration, pitch, 92 if slot == 0 else 78))
            rows.append(
                {
                    "小节": bar,
                    "拍点": f"{1 + offset:.1f}",
                    "音符": midi_note_name(pitch),
                    "MIDI": pitch,
                    "时值": f"{duration:.2f} 拍",
                    "建议": "主音吉他/键盘右手",
                }
            )
            item += 1
    return {"rows": rows, "notes": notes}


def chord_from_roman(key: str, roman: str) -> tuple[str, list[int]]:
    key_pc = note_to_pc(key)
    degree, quality = ROMAN_DEGREES[roman]
    if roman == "bVII":
        root_pc = (key_pc + 10) % 12
    else:
        root_pc = (key_pc + MAJOR_SCALE[degree]) % 12
    root_name = CHROMATIC[root_pc]
    intervals = [0, 4, 7] if quality == "major" else [0, 3, 7]
    chord_name = f"{root_name}{'' if quality == 'major' else 'm'}"
    root_midi = 48 + root_pc
    if root_midi > 59:
        root_midi -= 12
    pitches = [root_midi + interval for interval in intervals]
    return chord_name, pitches


def note_to_pc(name: str) -> int:
    normalized = name.strip().upper().replace("DB", "C#").replace("EB", "D#").replace("GB", "F#").replace("AB", "G#").replace("BB", "A#")
    return CHROMATIC.index(normalized) if normalized in CHROMATIC else 0


def midi_note_name(pitch: int) -> str:
    return f"{CHROMATIC[pitch % 12]}{pitch // 12 - 1}"


def generate_arrangement_ideas(metrics: AudioMetrics, chord_label: str) -> list[dict[str, str]]:
    tempo_hint = "偏快" if metrics.bpm >= 132 else "中速" if metrics.bpm >= 100 else "偏慢"
    density_hint = "密集" if metrics.density > 2.0 else "留白"
    return [
        {
            "编号": "A",
            "改编方向": "主歌减法",
            "执行方法": f"{tempo_hint} {metrics.bpm} BPM 下，主歌只留底鼓 1/3 拍、军鼓 2/4 拍，吉他用闷音八分音符。",
            "试听": "听鼓组预览 + 和弦预览",
        },
        {
            "编号": "B",
            "改编方向": "副歌放大",
            "执行方法": f"沿用 {chord_label}，副歌改成全开放扫弦，键盘在每小节第一拍加根音八度。",
            "试听": "听全曲草稿",
        },
        {
            "编号": "C",
            "改编方向": "间奏 Hook",
            "执行方法": f"主音吉他/键盘使用 {metrics.key} 大调五声音阶，把旋律预览的前 4 小节重复两次。",
            "试听": "听旋律预览",
        },
        {
            "编号": "D",
            "改编方向": "动态反差",
            "执行方法": f"如果原曲能量是 {metrics.energy:.2f}，Bridge 降到半拍感，最后一遍副歌恢复完整鼓组和开放和弦。",
            "试听": "听全曲草稿后对照原始分离轨",
        },
    ]


def render_drum_score(notes: list[tuple[float, float, int, int]], bars: int = 8) -> str:
    rows = [("HH", 42, "x"), ("T1", 48, "o"), ("SD", 38, "o"), ("T2", 45, "o"), ("BD", 36, "o")]
    lines = ["BandScribe Drum Staff | 4/4 | each cell = 1/16 note"]
    for first_bar in range(0, bars, 2):
        block_bars = list(range(first_bar, min(first_bar + 2, bars)))
        lines.append("    " + " 1e&a 2e&a 3e&a 4e&a " * len(block_bars))
        for label, pitch, mark in rows:
            measures = []
            for bar_index in block_bars:
                cells = ["-"] * 16
                for start, _, note_pitch, _ in notes:
                    if int(note_pitch) != pitch:
                        continue
                    step = int(round((float(start) - bar_index * 4.0) * 4.0))
                    if 0 <= step < 16:
                        cells[step] = mark
                measures.append("".join(cells))
            lines.append(f"{label:>2} |" + "|".join(measures) + "|")
        lines.append("")
    lines.append("HH=closed hi-hat, SD=snare, BD=bass drum, T1/T2=toms")
    return "\n".join(lines)


def render_guitar_tab(notes: list[tuple[float, float, int, int]], bars: int = 8) -> str:
    strings = [("e", 64), ("B", 59), ("G", 55), ("D", 50), ("A", 45), ("E", 40)]
    assigned: dict[tuple[int, int], tuple[int, int]] = {}
    for start, _, pitch, _ in notes:
        candidates = []
        for string_index, (_, open_pitch) in enumerate(strings):
            fret = int(pitch) - open_pitch
            if 0 <= fret <= 20:
                candidates.append((fret + string_index * 0.25, string_index, fret))
        if not candidates:
            continue
        _, string_index, fret = min(candidates)
        bar_index = int(float(start) // 4.0)
        step = int(round((float(start) - bar_index * 4.0) * 4.0))
        if 0 <= bar_index < bars and 0 <= step < 16:
            assigned[(bar_index, step)] = (string_index, fret)

    lines = ["BandScribe Lead Guitar TAB | standard tuning E A D G B e | 4/4"]
    for first_bar in range(0, bars, 2):
        block_bars = list(range(first_bar, min(first_bar + 2, bars)))
        lines.append("    " + " 1e&a 2e&a 3e&a 4e&a " * len(block_bars))
        for string_index, (label, _) in enumerate(strings):
            measures = []
            for bar_index in block_bars:
                cells = ["--"] * 16
                for step in range(16):
                    placement = assigned.get((bar_index, step))
                    if placement and placement[0] == string_index:
                        cells[step] = f"{placement[1]:02d}"
                measures.append("".join(cells))
            lines.append(f"{label} |" + "|".join(measures) + "|")
        lines.append("")
    lines.append("00=open string; 01-20=fret number; each cell=1/16 note")
    return "\n".join(lines)


def _staff_position(pitch: int) -> tuple[int, str]:
    name = midi_note_name(int(pitch))
    letter = name[0]
    octave = int(name[-1])
    return octave * 7 + "CDEFGAB".index(letter), name


def _render_staff(title: str, pitches: list[int], top: int, bottom: int) -> str:
    lines = [title]
    for offset in range(0, len(pitches), 12):
        block = pitches[offset : offset + 12]
        positions = [_staff_position(pitch) for pitch in block]
        lines.append("notes " + " ".join(f"{name:>3}" for _, name in positions))
        for position in range(top, bottom - 1, -1):
            is_line = (top - position) % 2 == 0
            cells = []
            for note_position, name in positions:
                if int(np.clip(note_position, bottom, top)) == position:
                    accidental = "#" if "#" in name else "o"
                    cells.append(f"{accidental:^3}")
                else:
                    cells.append("---" if is_line else "   ")
            edge = "|" if is_line else " "
            lines.append(f"{edge}{''.join(cells)}{edge}")
        lines.append("")
    return "\n".join(lines)


def render_keyboard_score(
    melody_notes: list[tuple[float, float, int, int]],
    chord_notes: list[tuple[float, float, int, int]],
) -> str:
    melody_pitches = [int(note[2]) for note in melody_notes]
    roots_by_start: dict[float, int] = {}
    for start, _, pitch, _ in chord_notes:
        roots_by_start[float(start)] = min(int(pitch), roots_by_start.get(float(start), 127))
    bass_pitches = [pitch for _, pitch in sorted(roots_by_start.items())]
    treble = _render_staff("TREBLE STAFF (right hand / melody)", melody_pitches, 38, 30)
    bass = _render_staff("BASS STAFF (left hand / chord roots)", bass_pitches, 26, 18)
    return treble + "\n" + bass + "\n# means sharp; notes outside the staff are clamped and named above."


def write_notation_artifacts(
    job_dir: Path,
    drums: dict[str, Any],
    chords: dict[str, Any],
    melody: dict[str, Any],
) -> dict[str, str]:
    artifact_dir = job_dir / "artifacts"
    artifact_dir.mkdir(parents=True, exist_ok=True)
    scores = {
        "drum_score": render_drum_score(drums["notes"]),
        "guitar_tab": render_guitar_tab(melody["notes"]),
        "keyboard_score": render_keyboard_score(melody["notes"], chords["notes"]),
    }
    result = dict(scores)
    for key, score_text in scores.items():
        path = artifact_dir / f"{key}.txt"
        path.write_text(score_text, encoding="utf-8")
        result[f"{key}_path"] = str(path)
    return result

def write_all_artifacts(job_dir: Path, metrics: AudioMetrics, drums: dict[str, Any], chords: dict[str, Any], melody: dict[str, Any]) -> dict[str, str]:
    artifact_dir = job_dir / "artifacts"
    artifact_dir.mkdir(parents=True, exist_ok=True)
    total_beats = 32.0

    drum_midi = artifact_dir / "drum_rhythm.mid"
    chord_midi = artifact_dir / "rhythm_chords.mid"
    melody_midi = artifact_dir / "lead_melody.mid"
    full_midi = artifact_dir / "full_band_sketch.mid"
    drum_wav = artifact_dir / "drum_rhythm_preview.wav"
    chord_wav = artifact_dir / "rhythm_chords_preview.wav"
    melody_wav = artifact_dir / "lead_melody_preview.wav"
    full_wav = artifact_dir / "full_band_sketch_preview.wav"

    write_midi(
        drum_midi,
        metrics.bpm,
        [{"name": "Drum rhythm", "channel": 9, "program": None, "notes": drums["notes"]}],
    )
    write_midi(
        chord_midi,
        metrics.bpm,
        [{"name": "Rhythm chords", "channel": 0, "program": 27, "notes": chords["notes"]}],
    )
    write_midi(
        melody_midi,
        metrics.bpm,
        [{"name": "Lead melody", "channel": 1, "program": 30, "notes": melody["notes"]}],
    )
    write_midi(
        full_midi,
        metrics.bpm,
        [
            {"name": "Drums", "channel": 9, "program": None, "notes": drums["notes"]},
            {"name": "Rhythm guitar", "channel": 0, "program": 27, "notes": chords["notes"]},
            {"name": "Lead", "channel": 1, "program": 30, "notes": melody["notes"]},
        ],
    )

    synth_preview(drum_wav, metrics.bpm, total_beats, drum_notes=drums["notes"])
    synth_preview(chord_wav, metrics.bpm, total_beats, tonal_notes=chords["notes"], tone_amp=0.18)
    synth_preview(melody_wav, metrics.bpm, total_beats, tonal_notes=melody["notes"], tone_amp=0.26)
    synth_preview(
        full_wav,
        metrics.bpm,
        total_beats,
        drum_notes=drums["notes"],
        tonal_notes=chords["notes"] + melody["notes"],
        tone_amp=0.14,
    )

    return {
        "drum_midi": str(drum_midi),
        "chord_midi": str(chord_midi),
        "melody_midi": str(melody_midi),
        "full_midi": str(full_midi),
        "drum_wav": str(drum_wav),
        "chord_wav": str(chord_wav),
        "melody_wav": str(melody_wav),
        "full_wav": str(full_wav),
    }


def write_midi(path: Path, bpm: int, tracks: list[dict[str, Any]]) -> None:
    tempo = int(60_000_000 / max(bpm, 1))
    chunks = [_midi_track("Tempo", [(0, 0, b"\xff\x51\x03" + tempo.to_bytes(3, "big")), (0, 1, b"\xff\x58\x04\x04\x02\x18\x08")])]
    for track in tracks:
        events: list[tuple[int, int, bytes]] = []
        channel = int(track["channel"])
        program = track.get("program")
        if program is not None:
            events.append((0, 0, bytes([0xC0 | channel, int(program)])))
        for start, duration, pitch, velocity in track["notes"]:
            start_tick = int(round(float(start) * TPQ))
            end_tick = int(round((float(start) + float(duration)) * TPQ))
            pitch = int(np.clip(pitch, 0, 127))
            velocity = int(np.clip(velocity, 1, 127))
            events.append((start_tick, 2, bytes([0x90 | channel, pitch, velocity])))
            events.append((max(end_tick, start_tick + 1), 1, bytes([0x80 | channel, pitch, 0])))
        chunks.append(_midi_track(str(track["name"]), events))

    header = b"MThd" + struct.pack(">LHHH", 6, 1, len(chunks), TPQ)
    path.write_bytes(header + b"".join(chunks))


def _midi_track(name: str, events: list[tuple[int, int, bytes]]) -> bytes:
    data = bytearray()
    name_bytes = name.encode("utf-8")
    data += _var_len(0) + b"\xff\x03" + _var_len(len(name_bytes)) + name_bytes
    last_tick = 0
    for tick, _, payload in sorted(events, key=lambda item: (item[0], item[1])):
        tick = max(0, int(tick))
        data += _var_len(tick - last_tick) + payload
        last_tick = tick
    data += _var_len(0) + b"\xff\x2f\x00"
    return b"MTrk" + struct.pack(">L", len(data)) + bytes(data)


def _var_len(value: int) -> bytes:
    value = max(0, int(value))
    buffer = [value & 0x7F]
    value >>= 7
    while value:
        buffer.insert(0, (value & 0x7F) | 0x80)
        value >>= 7
    return bytes(buffer)


def synth_preview(
    path: Path,
    bpm: int,
    total_beats: float,
    drum_notes: list[tuple[float, float, int, int]] | None = None,
    tonal_notes: list[tuple[float, float, int, int]] | None = None,
    tone_amp: float = 0.18,
) -> None:
    seconds_per_beat = 60.0 / max(float(bpm), 1.0)
    total_seconds = total_beats * seconds_per_beat + 1.0
    audio = np.zeros(int(total_seconds * SAMPLE_RATE), dtype=np.float32)
    rng = np.random.default_rng(20260710)

    for start, duration, pitch, velocity in drum_notes or []:
        start_s = float(start) * seconds_per_beat
        if int(pitch) == 36:
            _add_kick(audio, start_s, velocity / 127.0)
        elif int(pitch) == 38:
            _add_noise_hit(audio, start_s, 0.18, velocity / 127.0, rng, lowpass=True)
        else:
            _add_noise_hit(audio, start_s, 0.07, velocity / 127.0, rng, lowpass=False)

    for start, duration, pitch, velocity in tonal_notes or []:
        _add_tone(
            audio,
            float(start) * seconds_per_beat,
            max(float(duration) * seconds_per_beat, 0.08),
            midi_to_hz(int(pitch)),
            tone_amp * (velocity / 127.0),
        )

    _write_wav_float(path, audio)


def _add_tone(audio: np.ndarray, start_s: float, duration_s: float, frequency: float, amp: float) -> None:
    start = int(max(start_s, 0.0) * SAMPLE_RATE)
    count = int(max(duration_s, 0.03) * SAMPLE_RATE)
    end = min(len(audio), start + count)
    if start >= end:
        return
    t = np.arange(end - start, dtype=np.float32) / SAMPLE_RATE
    envelope = np.minimum(1.0, t / 0.025) * np.exp(-1.7 * t / max(duration_s, 0.1))
    wave_data = np.sin(2 * np.pi * frequency * t) + 0.35 * np.sin(2 * np.pi * frequency * 2.0 * t)
    audio[start:end] += (amp * envelope * wave_data).astype(np.float32)


def _add_kick(audio: np.ndarray, start_s: float, amp: float) -> None:
    start = int(start_s * SAMPLE_RATE)
    count = int(0.22 * SAMPLE_RATE)
    end = min(len(audio), start + count)
    if start >= end:
        return
    t = np.arange(end - start, dtype=np.float32) / SAMPLE_RATE
    freq = 70.0 - 35.0 * np.clip(t / 0.22, 0, 1)
    envelope = np.exp(-18.0 * t)
    audio[start:end] += (amp * 0.9 * envelope * np.sin(2 * np.pi * freq * t)).astype(np.float32)


def _add_noise_hit(audio: np.ndarray, start_s: float, duration_s: float, amp: float, rng: np.random.Generator, lowpass: bool) -> None:
    start = int(start_s * SAMPLE_RATE)
    count = int(duration_s * SAMPLE_RATE)
    end = min(len(audio), start + count)
    if start >= end:
        return
    t = np.arange(end - start, dtype=np.float32) / SAMPLE_RATE
    noise = rng.normal(0, 1, end - start).astype(np.float32)
    if lowpass and len(noise) > 5:
        noise = np.convolve(noise, np.ones(5, dtype=np.float32) / 5.0, mode="same")
    envelope = np.exp(-26.0 * t)
    audio[start:end] += amp * 0.45 * envelope * noise


def _write_wav_float(path: Path, audio: np.ndarray) -> None:
    peak = float(np.max(np.abs(audio))) if len(audio) else 0.0
    if peak > 0:
        audio = audio / peak * 0.85
    pcm = np.clip(audio, -1.0, 1.0)
    pcm16 = (pcm * 32767.0).astype("<i2")
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(SAMPLE_RATE)
        wf.writeframes(pcm16.tobytes())


def midi_to_hz(pitch: int) -> float:
    return 440.0 * (2.0 ** ((pitch - 69) / 12.0))


def build_analysis(root: Path, job_id: str, upload_path: Path, use_demucs: bool = True) -> dict[str, Any]:
    dirs = ensure_dirs(root)
    job_dir = dirs["jobs"] / job_id
    job_dir.mkdir(parents=True, exist_ok=True)

    separation = run_demucs(upload_path, job_dir, use_demucs=use_demucs)
    metrics_source = Path(separation.stems.get("drums") or upload_path)
    metrics = estimate_metrics(metrics_source)
    harmony_source = Path(
        separation.stems.get("rhythm_guitar")
        or separation.stems.get("full_mix")
        or upload_path
    )
    harmony_model = predict_harmony_with_fallback(harmony_source, metrics, root)
    if harmony_model["warning"]:
        metrics.warnings.append(harmony_model["warning"])
    drums = generate_drum_part(metrics)
    chords = generate_chord_part(metrics, progression=harmony_model.get("progression"))
    melody = generate_melody_part(metrics)
    ideas = generate_arrangement_ideas(metrics, chords["label"])
    artifacts = write_all_artifacts(job_dir, metrics, drums, chords, melody)
    notation = write_notation_artifacts(job_dir, drums, chords, melody)

    return {
        "job_id": job_id,
        "upload_path": str(upload_path),
        "separation": asdict(separation),
        "metrics": asdict(metrics),
        "harmony_model": harmony_model,
        "drums": {"rows": drums["rows"]},
        "chords": {"rows": chords["rows"], "label": chords["label"]},
        "melody": {"rows": melody["rows"]},
        "ideas": ideas,
        "artifacts": artifacts,
        "notation": notation,
    }






