const { app, BrowserWindow, ipcMain } = require("electron");
const { spawn } = require("child_process");
const fs = require("fs");
const http = require("http");
const path = require("path");

const APP_ROOT = path.resolve(__dirname, "..");
const PACKAGED_BACKEND_ROOT = path.join(process.resourcesPath || APP_ROOT, "neurowave-python");
const BACKEND_ROOT = app.isPackaged ? PACKAGED_BACKEND_ROOT : APP_ROOT;
const PACKAGE_BASE = process.env.PORTABLE_EXECUTABLE_DIR || path.dirname(process.execPath);
const SETTINGS_BASE = app.isPackaged ? PACKAGE_BASE : APP_ROOT;
const WINDOWS_LOCAL_APP_DATA = process.env.LOCALAPPDATA || app.getPath("appData");
const APP_DATA_ROOT = app.isPackaged ? path.join(WINDOWS_LOCAL_APP_DATA, "NeuroWave") : APP_ROOT;
const LOCAL_SETTINGS_PATH = app.isPackaged
  ? path.join(SETTINGS_BASE, "settings.local.json")
  : path.join(__dirname, "settings.local.json");
const DEFAULT_DEV_USER_DATA = path.join(APP_ROOT, ".electron-user-data");
const DEFAULT_MODEL_NAME = "v3.5_noise_detune_loss.pt";
const localSettings = loadLocalSettings();
const backendSettings = localSettings.backend || {};
const appSettings = localSettings.app || {};
const BACKEND_HOST = backendSettings.host || "127.0.0.1";
const BACKEND_PORT = Number(process.env.NEUROWAVE_BACKEND_PORT || backendSettings.port || "8765");
const BACKEND_URL = `http://${BACKEND_HOST}:${BACKEND_PORT}`;
const HEALTH_URL = `${BACKEND_URL}/health`;
const MAX_IMPORTED_AUDIO_BYTES = 100 * 1024 * 1024;

let mainWindow = null;
let backendProcess = null;
let backendStartedByShell = false;
let backendStartupError = "";

if (!app.isPackaged || process.env.NEUROWAVE_ELECTRON_USER_DATA) {
  app.setPath("userData", process.env.NEUROWAVE_ELECTRON_USER_DATA || DEFAULT_DEV_USER_DATA);
}

function loadLocalSettings() {
  if (!fs.existsSync(LOCAL_SETTINGS_PATH)) {
    return {};
  }
  try {
    return JSON.parse(fs.readFileSync(LOCAL_SETTINGS_PATH, "utf8"));
  } catch (error) {
    console.error(`Could not read ${LOCAL_SETTINGS_PATH}:`, error);
    return {};
  }
}

function defaultPythonPath() {
  if (process.env.NEUROWAVE_PYTHON) {
    return process.env.NEUROWAVE_PYTHON;
  }
  if (backendSettings.pythonPath) {
    return path.resolve(SETTINGS_BASE, backendSettings.pythonPath);
  }
  if (!app.isPackaged && process.platform === "win32") {
    return path.join(APP_ROOT, ".venv", "Scripts", "python.exe");
  }
  if (process.platform === "win32") {
    const venvPython = findFirstExistingPath(
      candidateBaseDirs().map((baseDir) => path.join(baseDir, ".venv", "Scripts", "python.exe")),
    );
    if (venvPython) {
      return venvPython;
    }
  }
  return "python";
}

function candidateBaseDirs() {
  const dirs = [
    SETTINGS_BASE,
    path.resolve(SETTINGS_BASE, ".."),
    path.resolve(SETTINGS_BASE, "..", ".."),
    APP_ROOT,
  ];
  return Array.from(new Set(dirs));
}

function findFirstExistingPath(candidates) {
  return candidates.find((candidate) => fs.existsSync(candidate)) || "";
}

function resolveConfiguredPath(value) {
  if (!value) {
    return "";
  }
  return path.isAbsolute(value) ? value : path.resolve(SETTINGS_BASE, value);
}

function defaultModelPath() {
  const configured = resolveConfiguredPath(appSettings.modelPath);
  if (configured) {
    return configured;
  }
  if (app.isPackaged) {
    const bundled = path.join(process.resourcesPath, "models", DEFAULT_MODEL_NAME);
    if (fs.existsSync(bundled)) {
      return bundled;
    }
  }
  const existing = findFirstExistingPath(
    candidateBaseDirs().map((baseDir) => path.join(baseDir, "models", DEFAULT_MODEL_NAME)),
  );
  if (existing) {
    return existing;
  }
  return path.join(candidateBaseDirs()[0], "models", DEFAULT_MODEL_NAME);
}

function defaultOutputDir() {
  const configured = resolveConfiguredPath(appSettings.outputDir);
  if (configured) {
    return configured;
  }
  if (app.isPackaged) {
    return path.join(APP_DATA_ROOT, "Runs");
  }
  const repoLikeBase = findFirstExistingPath(
    candidateBaseDirs().map((baseDir) => path.join(baseDir, "models")),
  );
  return path.join(repoLikeBase ? path.dirname(repoLikeBase) : SETTINGS_BASE, "runs", "app");
}

function backendLogPath() {
  const configured = resolveConfiguredPath(backendSettings.logPath);
  if (configured) {
    return configured;
  }
  if (app.isPackaged) {
    return path.join(APP_DATA_ROOT, "Logs", "neurowave-backend.log");
  }
  return path.join(SETTINGS_BASE, "neurowave-backend.log");
}

function appendBackendLog(message) {
  const destination = backendLogPath();
  fs.mkdirSync(path.dirname(destination), { recursive: true });
  fs.appendFileSync(destination, `${message}\n`, "utf8");
}

function sanitizeFileName(name) {
  const cleaned = String(name || "audio.wav")
    .replace(/[<>:"/\\|?*\x00-\x1F]/g, "_")
    .replace(/\s+/g, "_")
    .replace(/^_+/, "")
    .slice(0, 80);
  return cleaned || "audio.wav";
}

function appInputDir() {
  const configured = resolveConfiguredPath(appSettings.inputDir);
  if (configured) {
    return configured;
  }
  if (app.isPackaged) {
    return path.join(APP_DATA_ROOT, "Inputs");
  }
  return path.join(SETTINGS_BASE, "app-inputs");
}

function registerIpcHandlers() {
  ipcMain.handle("neurowave:import-audio", async (_event, payload) => {
    const fileName = sanitizeFileName(payload?.name);
    const bytes = payload?.bytes;
    if (!(bytes instanceof ArrayBuffer)) {
      throw new Error("Audio import requires file bytes");
    }
    if (bytes.byteLength <= 0) {
      throw new Error("Audio import is empty");
    }
    if (bytes.byteLength > MAX_IMPORTED_AUDIO_BYTES) {
      throw new Error("Audio import is larger than 100 MB");
    }

    const destinationDir = appInputDir();
    fs.mkdirSync(destinationDir, { recursive: true });
    const stamp = new Date().toISOString().replace(/[:.]/g, "-");
    const destination = path.join(destinationDir, `${stamp}_${fileName}`);
    fs.writeFileSync(destination, Buffer.from(bytes));
    return destination;
  });
}

function requestHealth(timeoutMs = 700) {
  return new Promise((resolve) => {
    const request = http.get(HEALTH_URL, { timeout: timeoutMs }, (response) => {
      response.resume();
      resolve(response.statusCode === 200);
    });
    request.on("timeout", () => {
      request.destroy();
      resolve(false);
    });
    request.on("error", () => resolve(false));
  });
}

async function waitForBackend(timeoutMs = 15000) {
  const start = Date.now();
  while (Date.now() - start < timeoutMs) {
    if (await requestHealth()) {
      return true;
    }
    await new Promise((resolve) => setTimeout(resolve, 250));
  }
  return false;
}

async function startBackendIfNeeded() {
  if (await requestHealth()) {
    return;
  }

  const python = defaultPythonPath();
  const backendLog = backendLogPath();
  appendBackendLog("");
  appendBackendLog(`[${new Date().toISOString()}] Starting backend with ${python}`);
  appendBackendLog(`Backend root: ${BACKEND_ROOT}`);
  const args = [
    "-u",
    path.join(BACKEND_ROOT, "scripts", "app_backend.py"),
    "--host",
    BACKEND_HOST,
    "--port",
    String(BACKEND_PORT),
  ];
  if (backendSettings.quiet !== false) {
    args.push("--quiet");
  }
  if (backendSettings.debugErrors !== false) {
    args.push("--debug-errors");
  }
  backendProcess = spawn(python, args, {
    cwd: BACKEND_ROOT,
    windowsHide: true,
    stdio: process.env.NEUROWAVE_BACKEND_LOGS ? "inherit" : ["ignore", "pipe", "pipe"],
    env: {
      ...process.env,
      PYTHONUNBUFFERED: "1",
    },
  });
  backendStartedByShell = true;

  if (!process.env.NEUROWAVE_BACKEND_LOGS) {
    backendProcess.stdout.on("data", (chunk) => {
      appendBackendLog(String(chunk).trimEnd());
    });
    backendProcess.stderr.on("data", (chunk) => {
      appendBackendLog(String(chunk).trimEnd());
    });
  }

  backendProcess.on("error", (error) => {
    appendBackendLog(`[${new Date().toISOString()}] Backend spawn error: ${error.message}`);
  });

  backendProcess.on("exit", (code, signal) => {
    appendBackendLog(
      `[${new Date().toISOString()}] Backend exited code=${code ?? "null"} signal=${signal ?? "null"}`,
    );
    backendProcess = null;
    backendStartedByShell = false;
  });

  if (!(await waitForBackend())) {
    appendBackendLog(`[${new Date().toISOString()}] Backend health check timed out at ${BACKEND_URL}`);
    throw new Error(`NeuroWave backend did not start at ${BACKEND_URL}. See ${backendLog}`);
  }
  appendBackendLog(`[${new Date().toISOString()}] Backend health check passed at ${BACKEND_URL}`);
}

function stopBackend() {
  if (backendProcess && backendStartedByShell) {
    backendProcess.kill();
  }
}

function createWindow() {
  const modelPath = defaultModelPath();
  const outputDir = defaultOutputDir();
  mainWindow = new BrowserWindow({
    width: 1280,
    height: 860,
    minWidth: 1040,
    minHeight: 720,
    title: "NeuroWave",
    backgroundColor: "#151616",
    webPreferences: {
      preload: path.join(__dirname, "preload.js"),
      contextIsolation: true,
      nodeIntegration: false,
      sandbox: false,
    },
  });

  mainWindow.loadFile(path.join(APP_ROOT, "app", "index.html"), {
    query: {
      backendUrl: BACKEND_URL,
      modelPath,
      outputDir,
      backendLogPath: backendLogPath(),
      backendStartupError,
      modelReady: fs.existsSync(modelPath) ? "1" : "0",
    },
  });

  if (process.env.NEUROWAVE_ELECTRON_SMOKE) {
    mainWindow.webContents.once("did-finish-load", () => {
      setTimeout(() => app.quit(), 500);
    });
  }

  if (process.env.NEUROWAVE_ELECTRON_DEVTOOLS) {
    mainWindow.webContents.openDevTools({ mode: "detach" });
  }
}

app.whenReady().then(async () => {
  registerIpcHandlers();
  try {
    await startBackendIfNeeded();
  } catch (error) {
    backendStartupError = error.message || String(error);
    console.error(error);
  }
  createWindow();

  app.on("activate", () => {
    if (BrowserWindow.getAllWindows().length === 0) {
      createWindow();
    }
  });
});

app.on("window-all-closed", () => {
  if (process.platform !== "darwin") {
    app.quit();
  }
});

app.on("before-quit", () => {
  stopBackend();
});
