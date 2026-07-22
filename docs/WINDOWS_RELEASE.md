# NeuroWave Windows Release Notes

## Current Package Shape

NeuroWave is configured to build as a Windows x64 Electron app delivered by an NSIS web
installer and a separately packaged, versioned payload.

The small bootstrap installer downloads the large bundled Python, CPU-only PyTorch, and ML
payload once during installation. This avoids the repeated multi-gigabyte Temp-folder
extraction that made the previous portable executable slow to start and avoids NSIS's
single-embedded-payload size limit. For offline or Windows Sandbox validation, place the
generated payload beside the installer; its checksum is verified before use.

The package includes:

- Electron desktop shell.
- NeuroWave frontend.
- Python backend source under `resources/neurowave-python/`.
- Selected model checkpoint under `resources/models/` when
  `models/v3.5_noise_detune_loss.pt` exists at build time.
- Prepared Python runtime under `resources/python-runtime/`. The public release uses the
  CPU-only staging runtime from `runtime/python-cpu/`; CUDA remains a separate local
  developer-runtime option under `runtime/python/`.

The package writes user data to `%LOCALAPPDATA%\NeuroWave\`:

- `Inputs\` for imported audio.
- `Runs\` for prediction outputs.
- `Logs\` for backend logs.

## Runtime Status

Development builds can still fall back to a nearby project `.venv`.

Public release builds must use a prepared CPU-only runtime in `runtime/python-cpu/` before
packaging. CUDA builds may use `runtime/python/` for local developer validation but are too
large for GitHub Release assets.
The desktop app looks for these paths first:

- `resources/python-runtime/python.exe`
- `resources/python-runtime/Scripts/python.exe`

If neither exists, the app searches nearby `.venv\Scripts\python.exe` paths, which is
only appropriate for developer machines.

Do not commit prepared runtimes. Both runtime staging directories are ignored except for
their placeholder files.

Validate the current development virtual environment dependency shape:

```powershell
npm run runtime:check:dev
```

Validate the prepared runtime that will be bundled into the app:

```powershell
npm run runtime:check
```

Public CPU release preparation flow:

```powershell
npm run runtime:prepare:cpu -- --replace
npm run runtime:check -- --runtime-dir runtime/python-cpu --require-cpu
npm run package:win:cpu:dir
python scripts/smoke_desktop_package.py --app dist/cpu/win-unpacked/NeuroWave.exe
python scripts/smoke_desktop_package.py --app dist/cpu/win-unpacked/NeuroWave.exe --audio playground/testpluck.wav
```

`runtime:prepare:cpu` copies the project dependency set into ignored `runtime/python-cpu/`,
then replaces CUDA PyTorch, TorchVision, and TorchAudio with pinned official CPU wheels.
This is intentionally a release-candidate staging command, not a source-controlled
dependency lock. The package step excludes venv marker files such as `pyvenv.cfg` from
`resources/python-runtime/`. Re-run the CPU runtime check after preparing it.

The bundled runtime has passed clean-machine validation in Windows Sandbox: the packaged
app started its bundled backend and loaded the bundled v3.5 model without the project
repo, project `.venv`, or developer Python installation.

## Build Commands

Build the unpacked app:

```powershell
npm run package:win:dir
```

Build the NSIS web installer and local payload without publishing:

```powershell
npm run package:win
```

Publish the installer and payload to a draft GitHub Release (requires `GH_TOKEN` or
`GITHUB_RELEASE_TOKEN` with repository contents permission):

```powershell
npm run package:win:release
```

For the public CPU release, use these commands instead:

```powershell
npm run package:win:cpu
npm run package:win:cpu:release
```

They write artifacts to `dist\cpu-nsis-web\`. For an offline Windows Sandbox test, copy the
complete `dist\cpu-nsis-web\nsis-web\` directory into
the Sandbox and run `NeuroWave Web Setup 0.1.0.exe` with
`neurowave-0.1.0-x64.nsis.7z` beside it. The bootstrapper validates that payload before
installing it. A published GitHub Release must include the bootstrapper, payload, and
`latest.yml` metadata generated in that directory.

Smoke-test the unpacked app backend:

```powershell
npm run package:smoke
```

Smoke-test prediction with the local `playground\testpluck.wav` clip:

```powershell
npm run package:smoke:predict
```

## Release Verification Checklist

- `dist\cpu\win-unpacked\resources\models\v3.5_noise_detune_loss.pt` exists.
- `dist\cpu\win-unpacked\resources\neurowave-python\scripts\app_backend.py` exists.
- `npm run runtime:check -- --runtime-dir runtime/python-cpu --require-cpu` passes before packaging.
- CPU unpacked-app readiness and prediction smoke tests pass.
- If shipping to non-developer machines, `dist\cpu\win-unpacked\resources\python-runtime\python.exe`
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
- The CPU NSIS web installer installs successfully and the installed app completes the same
  startup and prediction checks.
- Windows Sandbox clean-machine installer validation has passed without the project repo,
  project `.venv`, or developer Python installation.

## Known Pre-Website Blockers

- The release bootstrapper, payload, and `latest.yml` must be uploaded together to the
  configured GitHub Release before an online install can succeed. Keep the generated
  payload beside the bootstrapper when testing offline or in Windows Sandbox.
- The future Vercel website Download page must link only to a verified public GitHub Release;
  Vercel is not the distribution host for the multi-gigabyte installer payload.
- Code signing has not been configured.
- Manual packaged UI verification is still required after every release build.
