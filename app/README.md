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

Run the Electron desktop shell from the repo root:

```powershell
npm install
npm run desktop
```

The shell starts the Python backend automatically in development. It uses
`.venv\Scripts\python.exe` on Windows unless `NEUROWAVE_PYTHON` is set.
You can also copy `desktop/settings.example.json` to `desktop/settings.local.json`
to set the development Python path, backend port, default model, and output folder.

Create the first Windows development package:

```powershell
npm run package:win
```

This writes an ignored portable `.exe` under `dist\`. The current package is a desktop
wrapper for the app and backend source; it still expects Python dependencies and the
model checkpoint to be available on the user's machine. A fully bundled consumer
installer with embedded runtime/model belongs to the later product-hardening phase.
For packaged testing, place a `settings.local.json` beside the `.exe` if you need to
override the Python executable, model path, or output folder.

Packaged development builds search common repo/package locations for
`.venv\Scripts\python.exe` and `models\v3.5_noise_detune_loss.pt`. Backend startup
logs are written to `neurowave-backend.log` beside the packaged executable unless
`backend.logPath` overrides that location.

Current prototype notes:

- Drag/drop loads audio into the browser for waveform, crop, and crop playback.
- The default UI is producer-facing: drag, crop, confirm pitch, predict, view params, save.
- Advanced contains developer/runtime fields such as backend URL, model path, raw audio path, and output directory.
- Advanced backend/model/output values persist locally after editing.
- The backend still needs a filesystem `audio_path`; Electron can fill this from a dropped file path when available.
- A desktop wrapper can later provide the real dropped file path, or the backend can gain upload support.
- After prediction, the frontend loads predicted JSON, WAV, and spectrogram artifacts
  through the backend's current-process artifact allowlist.
- Crop zoom is frontend-only and does not change the crop seconds sent to the backend.
- Export buttons download the registered predicted JSON/WAV artifacts through the backend.
- The Folder button asks the backend to open the registered run directory on Windows.
- In Electron, dropped files can provide a filesystem path for the backend Audio Path field.
