from __future__ import annotations

from pathlib import Path
import wave

import numpy as np

from bandscribe_core import build_analysis, save_upload


ROOT = Path(__file__).resolve().parent
SAMPLE_RATE = 22050


def make_demo_wav() -> bytes:
    seconds = 6
    t = np.arange(seconds * SAMPLE_RATE, dtype=np.float32) / SAMPLE_RATE
    audio = 0.18 * np.sin(2 * np.pi * 110 * t)
    for beat in np.arange(0, seconds, 0.5):
        start = int(beat * SAMPLE_RATE)
        end = min(len(audio), start + int(0.05 * SAMPLE_RATE))
        audio[start:end] += np.hanning(end - start) * 0.65
    pcm = np.clip(audio, -1, 1)
    pcm16 = (pcm * 32767).astype("<i2")
    path = ROOT / "outputs" / "smoke_input.wav"
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(SAMPLE_RATE)
        wf.writeframes(pcm16.tobytes())
    return path.read_bytes()


def main() -> None:
    data = make_demo_wav()
    job_id, upload_path = save_upload(ROOT, "smoke_C.wav", data)
    result = build_analysis(ROOT, job_id, Path(upload_path), use_demucs=False)
    assert result["separation"]["mode"] == "demo_fallback"
    assert result["harmony_model"]["source"] in {"synthetic_harmony_model_v1", "heuristic"}
    assert len(result["drums"]["rows"]) == 8
    assert len(result["chords"]["rows"]) == 8
    assert len(result["melody"]["rows"]) == 24
    assert len(result["ideas"]) >= 3
    assert "HH" in result["notation"]["drum_score"]
    assert "standard tuning" in result["notation"]["guitar_tab"]
    assert "TREBLE STAFF" in result["notation"]["keyboard_score"]
    for notation_key in ("drum_score_path", "guitar_tab_path", "keyboard_score_path"):
        score_path = Path(result["notation"][notation_key])
        assert score_path.exists() and score_path.stat().st_size > 0, score_path
    for artifact_path in result["artifacts"].values():
        artifact = Path(artifact_path)
        assert artifact.exists() and artifact.stat().st_size > 0, artifact
    print("job_id:", result["job_id"])
    print("mode:", result["separation"]["mode"])
    print("bpm/key:", result["metrics"]["bpm"], result["metrics"]["key"])
    print("drum rows:", len(result["drums"]["rows"]))
    print("chord rows:", len(result["chords"]["rows"]))
    print("melody rows:", len(result["melody"]["rows"]))
    print("ideas:", len(result["ideas"]))
    print("full preview:", result["artifacts"]["full_wav"])


if __name__ == "__main__":
    main()



