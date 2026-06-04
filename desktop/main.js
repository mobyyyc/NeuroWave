const { app, BrowserWindow } = require("electron");
const { spawn } = require("child_process");
const fs = require("fs");
const http = require("http");
const path = require("path");

const ROOT = path.resolve(__dirname, "..");
const LOCAL_SETTINGS_PATH = path.join(__dirname, "settings.local.json");
const DEFAULT_DEV_USER_DATA = path.join(ROOT, ".electron-user-data");
const localSettings = loadLocalSettings();
const backendSettings = localSettings.backend || {};
const appSettings = localSettings.app || {};
const BACKEND_HOST = backendSettings.host || "127.0.0.1";
const BACKEND_PORT = Number(process.env.NEUROWAVE_BACKEND_PORT || backendSettings.port || "8765");
const BACKEND_URL = `http://${BACKEND_HOST}:${BACKEND_PORT}`;
const HEALTH_URL = `${BACKEND_URL}/health`;

let mainWindow = null;
let backendProcess = null;
let backendStartedByShell = false;

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
    return path.resolve(ROOT, backendSettings.pythonPath);
  }
  if (process.platform === "win32") {
    return path.join(ROOT, ".venv", "Scripts", "python.exe");
  }
  return "python";
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
  const args = [
    path.join(ROOT, "scripts", "app_backend.py"),
    "--host",
    BACKEND_HOST,
    "--port",
    String(BACKEND_PORT),
  ];
  if (backendSettings.quiet !== false) {
    args.push("--quiet");
  }
  backendProcess = spawn(python, args, {
    cwd: ROOT,
    windowsHide: true,
    stdio: process.env.NEUROWAVE_BACKEND_LOGS ? "inherit" : "ignore",
  });
  backendStartedByShell = true;

  backendProcess.on("exit", () => {
    backendProcess = null;
    backendStartedByShell = false;
  });

  if (!(await waitForBackend())) {
    throw new Error(`NeuroWave backend did not start at ${BACKEND_URL}`);
  }
}

function stopBackend() {
  if (backendProcess && backendStartedByShell) {
    backendProcess.kill();
  }
}

function createWindow() {
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

  mainWindow.loadFile(path.join(ROOT, "app", "index.html"), {
    query: {
      backendUrl: BACKEND_URL,
      modelPath: appSettings.modelPath || "models/v3.5_noise_detune_loss.pt",
      outputDir: appSettings.outputDir || "runs/app",
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
  try {
    await startBackendIfNeeded();
  } catch (error) {
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
