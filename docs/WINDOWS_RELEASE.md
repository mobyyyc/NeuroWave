# NeuroWave Windows Release Notes

## Current Package Shape

NeuroWave currently builds as a Windows x64 Electron portable app.

The package includes:

- Electron desktop shell.
- NeuroWave frontend.
- Python backend source under `resources/neurowave-python/`.
- Selected model checkpoint under `resources/models/` when
  `models/v3.5_noise_detune_loss.pt` exists at build time.
- Optional prepared Python runtime under `resources/python-runtime/` when
  `runtime/python/` contains a real runtime at build time.

The package writes user data to `%LOCALAPPDATA%\NeuroWave\`:

- `Inputs\` for imported audio.
- `Runs\` for prediction outputs.
- `Logs\` for backend logs.

## Runtime Status

Development builds can still fall back to a nearby project `.venv`.

Release builds should use a prepared runtime in `runtime/python/` before packaging.
The desktop app looks for these paths first:

- `resources/python-runtime/python.exe`
- `resources/python-runtime/Scripts/python.exe`

If neither exists, the app searches nearby `.venv\Scripts\python.exe` paths, which is
only appropriate for developer machines.

Do not commit the prepared runtime. `runtime/python/` is ignored except for its
placeholder file.

## Build Commands

Build the unpacked app:

```powershell
npm run package:win:dir
```

Build the standalone portable executable:

```powershell
npm run package:win
```

## Release Verification Checklist

- `dist\win-unpacked\resources\models\v3.5_noise_detune_loss.pt` exists.
- `dist\win-unpacked\resources\neurowave-python\scripts\app_backend.py` exists.
- If shipping to non-developer machines, `dist\win-unpacked\resources\python-runtime\python.exe`
  or `dist\win-unpacked\resources\python-runtime\Scripts\python.exe` exists.
- The app opens without a false backend error.
- Backend readiness becomes Ready.
- Runtime readiness shows CPU or CUDA.
- A WAV can be dragged or selected.
- Crop handles work visually.
- Crop preview plays with a moving playhead.
- Predict produces a patch JSON and rendered WAV.
- Original and predicted audio can be compared.
- Target and predicted spectrograms display.
- Save JSON works.
- Save WAV works.
- Folder opens the run directory.
- Backend log is written under `%LOCALAPPDATA%\NeuroWave\Logs\`.

## Known Pre-Website Blockers

- A fully prepared Python/Torch runtime still needs to be produced and tested on a clean
  Windows machine.
- Code signing has not been configured.
- Installer packaging has not been configured; the current target is portable.
- Manual packaged UI verification is still required after every release build.
