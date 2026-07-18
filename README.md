# BandScribe

BandScribe is a Streamlit app for quick band transcription sketches. The first target is not perfect accuracy; it is a complete pipeline that runs:

1. Upload audio.
2. Try demucs source separation.
3. Generate a drum rhythm chart.
4. Generate a rhythm guitar chord progression.
5. Generate lead guitar / keyboard melody notes.
6. Generate at least 3 arrangement ideas.
7. Provide audition files as separated audio, WAV previews, and MIDI downloads.

## Run

```powershell
python -m pip install -r requirements.txt
streamlit run streamlit_app.py
```

On this Windows machine, `python` may not be on `PATH`. Use the bundled runtime directly:

```powershell
& 'C:\Users\lenovo\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe' -m pip install -r requirements.txt
.\run_app.ps1
```

For real source separation:

```powershell
python -m pip install -r requirements-demucs.txt
streamlit run streamlit_app.py
```

With the bundled runtime:

```powershell
& 'C:\Users\lenovo\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe' -m pip install -r requirements-demucs.txt
.\run_app.ps1
```

If demucs is not installed or fails, BandScribe keeps going by using the original audio as a placeholder track.

## Deploy to Streamlit Community Cloud

1. Push this repository to GitHub.
2. Open https://share.streamlit.io and choose Create app.
3. Select Oookla28/BandScribe, branch main, and entrypoint streamlit_app.py.
4. Choose a public custom subdomain such as bandscribe-oookla28, then deploy.

Only source files are committed. Runtime uploads, Demucs stems, previews, and feedback stay under outputs/ and are intentionally excluded from GitHub.
## Checkpoints

1. Stage 0: Upload an audio file. Verify that the original player appears and a file is written under `outputs/uploads/`.
2. Stage 1: Enable "try real demucs separation". If demucs is installed, verify separated WAV files under `outputs/jobs/<job_id>/separated/`; otherwise verify the app reports fallback mode and every track still plays.
3. Stage 2: Verify drum staff, lead-guitar TAB, keyboard treble/bass staff, chord table, and note-event table all render.
4. Stage 3: Verify each table has a WAV preview player and a MIDI download.
5. Stage 4: Verify at least 3 arrangement ideas appear and the full-band sketch preview plays.
6. Stage 5: Verify the sidebar reports a trained harmony model, results show confidence, and member corrections are appended under outputs/feedback/.

## Train The Harmony Model

The first trainable checkpoint predicts 12 major keys and three common chord progressions. It uses a deterministic synthetic dataset and a pure NumPy softmax model:

~~~powershell
python train_models.py
~~~

On this Windows machine:

~~~powershell
& 'C:\Users\lenovo\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe' train_models.py
~~~

The model is saved at models/harmony_model.npz. The app can also retrain it from the sidebar. Synthetic holdout accuracy is only a pipeline sanity check; use the correction controls in the chord tab to collect real-song labels under outputs/feedback/harmony_feedback.jsonl.

Verification: training must report non-empty train/test splits, the model file must exist, and a new analysis must show synthetic_harmony_model_v1 or a clear confidence-based fallback reason.
## Fast Smoke Test

This does not require Streamlit:

```powershell
python smoke_test.py
```

If `python` is not on `PATH`:

```powershell
& 'C:\Users\lenovo\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe' smoke_test.py
```

It creates a small synthetic WAV, runs the fallback pipeline, and prints artifact paths.
