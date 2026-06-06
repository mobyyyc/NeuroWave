# NeuroWave App

Dependency-free frontend for the Windows-first desktop app workflow.

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

Create the Windows development package:

```powershell
npm run package:win
```

This writes an ignored portable `.exe` under `dist\`. The current package is a desktop
wrapper for the app, backend source, and selected model checkpoint when
`models\v3.5_noise_detune_loss.pt` exists at build time. It still expects Python/Torch
dependencies to be available on the user's machine. A fully bundled consumer installer
with embedded Python/Torch runtime belongs to the product-hardening phase.
For packaged testing, place a `settings.local.json` beside the `.exe` if you need to
override the Python executable, model path, or output folder.

Packaged development builds first use the bundled checkpoint under
`resources\models\v3.5_noise_detune_loss.pt` when present, then search common
repo/package locations for `models\v3.5_noise_detune_loss.pt`. For portable builds,
optional `settings.local.json` is read beside the outer portable `.exe`, not the
temporary extraction folder. Runtime app data is stored under `%LOCALAPPDATA%\NeuroWave\`
by default:

- imported audio: `Inputs\`
- prediction runs: `Runs\`
- backend logs: `Logs\`

Use `backend.logPath`, `app.inputDir`, or `app.outputDir` in `settings.local.json`
only when you need to override those defaults during development.

Current prototype notes:

- Drag/drop loads audio into the browser for waveform, crop, and crop playback.
- The default UI is producer-facing: drag, crop, confirm pitch, predict, view params, save.
- Runtime, model, audio, and crop readiness are shown before prediction.
- Recent files and recent prediction runs are remembered locally.
- The runtime panel reports CPU/CUDA availability from the local backend.
- The pitch field warns when note and Hz disagree.
- The crop region is capped to the current model window so users cannot select more audio
  than the model input can represent.
- Crop playback draws a vertical playhead over the waveform.
- Advanced contains developer/runtime fields such as backend URL, model path, raw audio path, and output directory.
- Advanced backend/model/output values persist locally after editing.
- The backend still needs a filesystem `audio_path`; Electron imports dropped/selected
  audio into the app data input folder and sends that copied path to Python.
- After prediction, the frontend loads predicted JSON, WAV, and spectrogram artifacts
  through the backend's current-process artifact allowlist.
- Crop zoom is frontend-only and does not change the crop seconds sent to the backend.
- Export buttons download the registered predicted JSON/WAV artifacts through the backend.
- The Folder button asks the backend to open the registered run directory on Windows.
- In Electron, dropped files can provide a filesystem path for the backend Audio Path field.
