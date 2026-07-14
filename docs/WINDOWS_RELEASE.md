# NeuroWave Windows Release Notes

## Current Package Shape

NeuroWave is configured to build as a Windows x64 Electron app delivered by an NSIS installer.

The installer extracts the large bundled Python, CUDA PyTorch, and ML dependencies once
during installation. This avoids the repeated multi-gigabyte Temp-folder extraction that
made the previous portable executable slow to start.

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

Validate the current development virtual environment dependency shape:

```powershell
npm run runtime:check:dev
```

Validate the prepared runtime that will be bundled into the app:

```powershell
npm run runtime:check
```

First-release runtime preparation flow:

```powershell
npm run runtime:prepare
npm run runtime:check
npm run package:win:dir
npm run package:smoke
npm run package:smoke:predict
```

`runtime:prepare` copies the active Python install and the current working
`site-packages` set into ignored `runtime/python/`. This is intentionally a release
candidate staging command, not a source-controlled dependency lock. The package step
excludes venv marker files such as `pyvenv.cfg` from `resources/python-runtime/`.
Re-run `runtime:check` after preparing it.

The bundled runtime has passed clean-machine validation in Windows Sandbox: the packaged
app started its bundled backend and loaded the bundled v3.5 model without the project
repo, project `.venv`, or developer Python installation.

## Build Commands

Build the unpacked app:

```powershell
npm run package:win:dir
```

Build the NSIS installer:

```powershell
npm run package:win
```

Smoke-test the unpacked app backend:

```powershell
npm run package:smoke
```

Smoke-test prediction with the local `playground\testpluck.wav` clip:

```powershell
npm run package:smoke:predict
```

## Release Verification Checklist

- `dist\win-unpacked\resources\models\v3.5_noise_detune_loss.pt` exists.
- `dist\win-unpacked\resources\neurowave-python\scripts\app_backend.py` exists.
- `npm run runtime:check` passes before packaging.
- `npm run package:smoke` passes.
- `npm run package:smoke:predict` passes when `playground\testpluck.wav` exists.
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
- The NSIS installer installs successfully and the installed app completes the same
  startup and prediction checks.

## Known Pre-Website Blockers

- The current 5.35 GB installed runtime payload produces a 2.36 GB compressed archive,
  which the local NSIS compiler cannot memory-map into a single embedded installer.
  `npm run package:win` therefore stops at final NSIS assembly. Resolve this by
  reducing/splitting the runtime payload or by using a staged/download installer before
  treating the NSIS package as releasable.
- Code signing has not been configured.
- Manual packaged UI verification is still required after every release build.
