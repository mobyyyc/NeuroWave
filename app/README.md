# NeuroWave App Prototype

Dependency-free frontend prototype for the Windows-first desktop app workflow.

Start the Python backend from the repo root:

```powershell
python scripts\app_backend.py --host 127.0.0.1 --port 8765
```

Serve the static app from the repo root:

```powershell
python -m http.server 5173 --directory app
```

Open:

```text
http://127.0.0.1:5173
```

Current prototype notes:

- Drag/drop loads audio into the browser for waveform, crop, and crop playback.
- The backend still needs a filesystem `audio_path`, so use the Audio Path field for prediction.
- A desktop wrapper can later provide the real dropped file path, or the backend can gain upload support.
