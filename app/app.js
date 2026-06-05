const state = {
  audioBuffer: null,
  audioFileName: "",
  duration: 0,
  cropStart: 0,
  cropEnd: 0,
  dragMode: null,
  playbackSource: null,
  audioContext: null,
  targetSpectrogram: null,
  predictedSpectrogram: null,
  zoomLevel: 1,
  lastResult: null,
};

const els = {
  backendStatus: document.getElementById("backendStatus"),
  backendUrl: document.getElementById("backendUrl"),
  checkBackend: document.getElementById("checkBackend"),
  modelPath: document.getElementById("modelPath"),
  backendAudioPath: document.getElementById("backendAudioPath"),
  noteName: document.getElementById("noteName"),
  freqHz: document.getElementById("freqHz"),
  cropStart: document.getElementById("cropStart"),
  cropEnd: document.getElementById("cropEnd"),
  zoomLevel: document.getElementById("zoomLevel"),
  zoomValue: document.getElementById("zoomValue"),
  fitCropZoom: document.getElementById("fitCropZoom"),
  resetZoom: document.getElementById("resetZoom"),
  outputDir: document.getElementById("outputDir"),
  playCrop: document.getElementById("playCrop"),
  stopPlayback: document.getElementById("stopPlayback"),
  predictButton: document.getElementById("predictButton"),
  dropZone: document.getElementById("dropZone"),
  fileInput: document.getElementById("fileInput"),
  fileName: document.getElementById("fileName"),
  fileMeta: document.getElementById("fileMeta"),
  canvas: document.getElementById("waveformCanvas"),
  resultSummary: document.getElementById("resultSummary"),
  responseJson: document.getElementById("responseJson"),
  artifactStatus: document.getElementById("artifactStatus"),
  targetAudio: document.getElementById("targetAudio"),
  predictedAudio: document.getElementById("predictedAudio"),
  playTargetResult: document.getElementById("playTargetResult"),
  playPredictedResult: document.getElementById("playPredictedResult"),
  stopResultPlayback: document.getElementById("stopResultPlayback"),
  exportPatch: document.getElementById("exportPatch"),
  exportWav: document.getElementById("exportWav"),
  openRunFolder: document.getElementById("openRunFolder"),
  patchJson: document.getElementById("patchJson"),
  targetSpectrogramCanvas: document.getElementById("targetSpectrogramCanvas"),
  predictedSpectrogramCanvas: document.getElementById("predictedSpectrogramCanvas"),
};

const ctx = els.canvas.getContext("2d");
const SETTINGS_KEY = "neurowave.appSettings.v1";
const SETTINGS_FIELDS = {
  backendUrl: "http://127.0.0.1:8765",
  modelPath: "models/v3.5_noise_detune_loss.pt",
  outputDir: "runs/app",
};

function setStatus(text, kind = "idle") {
  els.backendStatus.textContent = text;
  els.backendStatus.className = `status-pill status-${kind}`;
}

function backendBaseUrl() {
  return els.backendUrl.value.replace(/\/$/, "");
}

function artifactUrl(path) {
  return `${backendBaseUrl()}/artifact?path=${encodeURIComponent(path)}`;
}

function setArtifactStatus(text, kind = "idle") {
  els.artifactStatus.textContent = text;
  els.artifactStatus.className = `artifact-status status-${kind}`;
}

function readStoredSettings() {
  try {
    const raw = window.localStorage.getItem(SETTINGS_KEY);
    return raw ? JSON.parse(raw) : {};
  } catch (_error) {
    return {};
  }
}

function querySetting(name) {
  return new URLSearchParams(window.location.search).get(name);
}

function desktopSetting(name) {
  return window.neurowaveDesktop?.settings?.[name] || null;
}

function desktopBackendDiagnostic() {
  const error = desktopSetting("backendStartupError");
  const logPath = desktopSetting("backendLogPath");
  if (!error && !logPath) {
    return "";
  }
  const parts = [];
  if (error) {
    parts.push(error);
  }
  if (logPath) {
    parts.push(`Log: ${logPath}`);
  }
  return parts.join("\n");
}

function currentSettings() {
  return {
    backendUrl: els.backendUrl.value,
    modelPath: els.modelPath.value,
    outputDir: els.outputDir.value,
  };
}

function saveSettings() {
  window.localStorage.setItem(SETTINGS_KEY, JSON.stringify(currentSettings()));
}

function applySettings() {
  const stored = readStoredSettings();
  for (const [name, fallback] of Object.entries(SETTINGS_FIELDS)) {
    const element = els[name];
    const value = desktopSetting(name) || querySetting(name) || stored[name] || fallback;
    if (element && value) {
      element.value = value;
    }
  }
}

function ensureAudioContext() {
  if (!state.audioContext) {
    state.audioContext = new (window.AudioContext || window.webkitAudioContext)();
  }
  return state.audioContext;
}

function resizeCanvas() {
  const rect = els.canvas.getBoundingClientRect();
  const scale = window.devicePixelRatio || 1;
  els.canvas.width = Math.max(1, Math.floor(rect.width * scale));
  els.canvas.height = Math.max(1, Math.floor(rect.height * scale));
  ctx.setTransform(scale, 0, 0, scale, 0, 0);
  drawWaveform();
  drawStoredSpectrograms();
}

function cropCenterSeconds() {
  if (!state.duration) return 0;
  const start = clamp(state.cropStart, 0, state.duration);
  const end = clamp(state.cropEnd || state.duration, start, state.duration);
  return clamp((start + end) / 2, 0, state.duration);
}

function visibleRange() {
  if (!state.duration) return { start: 0, end: 0, duration: 0 };
  const zoom = Math.max(1, Number(state.zoomLevel) || 1);
  const visibleDuration = state.duration / zoom;
  const center = cropCenterSeconds();
  let start = center - visibleDuration / 2;
  let end = center + visibleDuration / 2;
  if (start < 0) {
    end -= start;
    start = 0;
  }
  if (end > state.duration) {
    start -= end - state.duration;
    end = state.duration;
  }
  start = clamp(start, 0, state.duration);
  end = clamp(end, start, state.duration);
  return { start, end, duration: Math.max(1e-9, end - start) };
}

function secondsToX(seconds) {
  const width = els.canvas.getBoundingClientRect().width;
  const visible = visibleRange();
  if (!state.duration || !visible.duration) return 0;
  return ((seconds - visible.start) / visible.duration) * width;
}

function xToSeconds(x) {
  const width = els.canvas.getBoundingClientRect().width;
  const visible = visibleRange();
  if (!state.duration || width <= 0 || !visible.duration) return 0;
  return clamp(visible.start + (x / width) * visible.duration, visible.start, visible.end);
}

function clamp(value, min, max) {
  return Math.min(Math.max(value, min), max);
}

function formatSeconds(value) {
  return Number(value || 0).toFixed(3);
}

function drawEmptyWaveform() {
  const rect = els.canvas.getBoundingClientRect();
  ctx.clearRect(0, 0, rect.width, rect.height);
  ctx.fillStyle = "#0e1010";
  ctx.fillRect(0, 0, rect.width, rect.height);
  ctx.strokeStyle = "#2f3432";
  ctx.beginPath();
  ctx.moveTo(0, rect.height / 2);
  ctx.lineTo(rect.width, rect.height / 2);
  ctx.stroke();
}

function drawWaveform() {
  const rect = els.canvas.getBoundingClientRect();
  drawEmptyWaveform();
  if (!state.audioBuffer) return;

  const data = state.audioBuffer.getChannelData(0);
  const width = Math.max(1, Math.floor(rect.width));
  const height = rect.height;
  const center = height / 2;
  const visible = visibleRange();
  const startSample = Math.max(0, Math.floor((visible.start / state.duration) * data.length));
  const endSample = Math.min(data.length, Math.ceil((visible.end / state.duration) * data.length));
  const visibleSamples = Math.max(1, endSample - startSample);
  const samplesPerPixel = Math.max(1, Math.floor(visibleSamples / width));

  ctx.strokeStyle = "#8fd3ff";
  ctx.lineWidth = 1;
  ctx.beginPath();
  for (let x = 0; x < width; x += 1) {
    const start = Math.min(endSample, startSample + x * samplesPerPixel);
    const end = Math.min(start + samplesPerPixel, endSample);
    let min = 1;
    let max = -1;
    for (let i = start; i < end; i += 1) {
      const sample = data[i];
      if (sample < min) min = sample;
      if (sample > max) max = sample;
    }
    ctx.moveTo(x, center + min * center * 0.88);
    ctx.lineTo(x, center + max * center * 0.88);
  }
  ctx.stroke();

  drawCropOverlay(rect.width, rect.height);
}

function drawCropOverlay(width, height) {
  const startX = secondsToX(state.cropStart);
  const endX = secondsToX(state.cropEnd || state.duration);
  ctx.fillStyle = "rgba(56, 193, 114, 0.18)";
  ctx.fillRect(startX, 0, Math.max(1, endX - startX), height);

  ctx.strokeStyle = "#7ee3a1";
  ctx.lineWidth = 2;
  ctx.beginPath();
  ctx.moveTo(startX, 0);
  ctx.lineTo(startX, height);
  ctx.moveTo(endX, 0);
  ctx.lineTo(endX, height);
  ctx.stroke();

  ctx.fillStyle = "#7ee3a1";
  ctx.fillRect(startX - 4, 0, 8, height);
  ctx.fillRect(endX - 4, 0, 8, height);
}

async function loadAudioFile(file) {
  const audioContext = ensureAudioContext();
  const buffer = await file.arrayBuffer();
  const decoded = await audioContext.decodeAudioData(buffer.slice(0));
  const importedPath = window.neurowaveDesktop?.importAudioFile
    ? await window.neurowaveDesktop.importAudioFile(file.name, buffer.slice(0))
    : "";

  state.audioBuffer = decoded;
  state.audioFileName = file.name;
  state.duration = decoded.duration;
  state.cropStart = 0;
  state.cropEnd = decoded.duration;
  state.zoomLevel = 1;

  els.fileName.textContent = file.name;
  els.fileMeta.textContent = `${decoded.numberOfChannels} ch | ${decoded.sampleRate} Hz | ${formatSeconds(decoded.duration)} s`;
  els.cropStart.value = formatSeconds(state.cropStart);
  els.cropEnd.value = formatSeconds(state.cropEnd);
  updateZoomDisplay();
  const desktopPath = window.neurowaveDesktop?.pathForFile?.(file);
  if (importedPath) {
    els.backendAudioPath.value = importedPath;
  } else if (desktopPath) {
    els.backendAudioPath.value = desktopPath;
  } else if (file.path) {
    els.backendAudioPath.value = file.path;
  } else if (!els.backendAudioPath.value) {
    els.backendAudioPath.value = `playground/${file.name}`;
  }
  setArtifactStatus(importedPath ? "Audio ready" : "Audio loaded");
  drawWaveform();
}

function updateCropInputs() {
  els.cropStart.value = formatSeconds(state.cropStart);
  els.cropEnd.value = formatSeconds(state.cropEnd);
}

function updateZoomDisplay() {
  els.zoomLevel.value = String(state.zoomLevel);
  els.zoomValue.textContent = `${Number(state.zoomLevel).toFixed(2)}x`;
}

function updateZoomFromInput() {
  state.zoomLevel = clamp(Number(els.zoomLevel.value) || 1, 1, 40);
  updateZoomDisplay();
  drawWaveform();
}

function resetZoom() {
  state.zoomLevel = 1;
  updateZoomDisplay();
  drawWaveform();
}

function fitZoomToCrop() {
  if (!state.audioBuffer || !state.duration) return;
  const cropDuration = Math.max(0.01, state.cropEnd - state.cropStart);
  state.zoomLevel = clamp((state.duration / cropDuration) * 0.75, 1, 40);
  updateZoomDisplay();
  drawWaveform();
}

function updateCropFromInputs() {
  if (!state.audioBuffer) return;
  const start = clamp(Number(els.cropStart.value), 0, state.duration);
  const end = clamp(Number(els.cropEnd.value), 0, state.duration);
  if (end <= start) {
    return;
  }
  state.cropStart = start;
  state.cropEnd = end;
  drawWaveform();
}

function canvasMouseX(event) {
  const rect = els.canvas.getBoundingClientRect();
  return event.clientX - rect.left;
}

function beginCropDrag(event) {
  if (!state.audioBuffer) return;
  const x = canvasMouseX(event);
  const startX = secondsToX(state.cropStart);
  const endX = secondsToX(state.cropEnd);
  const handleSize = 12;
  if (Math.abs(x - startX) <= handleSize) {
    state.dragMode = "start";
  } else if (Math.abs(x - endX) <= handleSize) {
    state.dragMode = "end";
  } else if (x > startX && x < endX) {
    state.dragMode = "region";
    state.dragOffsetSeconds = xToSeconds(x) - state.cropStart;
    state.dragDurationSeconds = state.cropEnd - state.cropStart;
  } else {
    state.dragMode = x < startX ? "start" : "end";
  }
  moveCropDrag(event);
}

function moveCropDrag(event) {
  if (!state.audioBuffer || !state.dragMode) return;
  const seconds = xToSeconds(canvasMouseX(event));
  const minGap = Math.min(0.01, state.duration / 100);

  if (state.dragMode === "start") {
    state.cropStart = clamp(seconds, 0, state.cropEnd - minGap);
  } else if (state.dragMode === "end") {
    state.cropEnd = clamp(seconds, state.cropStart + minGap, state.duration);
  } else if (state.dragMode === "region") {
    const duration = state.dragDurationSeconds;
    let nextStart = seconds - state.dragOffsetSeconds;
    nextStart = clamp(nextStart, 0, state.duration - duration);
    state.cropStart = nextStart;
    state.cropEnd = nextStart + duration;
  }
  updateCropInputs();
  drawWaveform();
}

function endCropDrag() {
  state.dragMode = null;
}

function stopPlayback() {
  if (state.playbackSource) {
    try {
      state.playbackSource.stop();
    } catch (_error) {
      // Already stopped.
    }
    state.playbackSource = null;
  }
}

function playCrop() {
  if (!state.audioBuffer) return;
  stopPlayback();
  stopResultPlayback();
  const audioContext = ensureAudioContext();
  const source = audioContext.createBufferSource();
  source.buffer = state.audioBuffer;
  source.connect(audioContext.destination);
  source.onended = () => {
    if (state.playbackSource === source) state.playbackSource = null;
  };
  state.playbackSource = source;
  source.start(0, state.cropStart, Math.max(0.01, state.cropEnd - state.cropStart));
}

function stopResultPlayback() {
  for (const audio of [els.targetAudio, els.predictedAudio]) {
    audio.pause();
    audio.currentTime = 0;
  }
}

function playResultAudio(kind) {
  const audio = kind === "target" ? els.targetAudio : els.predictedAudio;
  const other = kind === "target" ? els.predictedAudio : els.targetAudio;
  if (!audio.src) return;
  stopPlayback();
  other.pause();
  other.currentTime = 0;
  audio.currentTime = 0;
  audio.play().catch((error) => {
    setArtifactStatus(error.message, "error");
  });
}

function noteToFrequency(note) {
  const match = String(note).trim().match(/^([A-Ga-g])([#b]?)(-?\d+)$/);
  if (!match) return null;
  const [, letterRaw, accidental, octaveRaw] = match;
  const letter = letterRaw.toUpperCase();
  const semitones = { C: -9, D: -7, E: -5, F: -4, G: -2, A: 0, B: 2 };
  let offset = semitones[letter];
  if (accidental === "#") offset += 1;
  if (accidental === "b") offset -= 1;
  const octave = Number(octaveRaw);
  const distanceFromA4 = offset + (octave - 4) * 12;
  return 440 * Math.pow(2, distanceFromA4 / 12);
}

function applyNoteInput() {
  const freq = noteToFrequency(els.noteName.value);
  if (freq) {
    els.freqHz.value = freq.toFixed(2);
  }
}

function clearSpectrogramCanvas(canvas) {
  const rect = canvas.getBoundingClientRect();
  const scale = window.devicePixelRatio || 1;
  canvas.width = Math.max(1, Math.floor(rect.width * scale));
  canvas.height = Math.max(1, Math.floor(rect.height * scale));
  const specCtx = canvas.getContext("2d");
  specCtx.setTransform(scale, 0, 0, scale, 0, 0);
  specCtx.clearRect(0, 0, rect.width, rect.height);
  specCtx.fillStyle = "#0e1010";
  specCtx.fillRect(0, 0, rect.width, rect.height);
}

function spectrogramColor(value) {
  const t = clamp(value, 0, 1);
  const hue = 225 - t * 175;
  const light = 14 + t * 62;
  return `hsl(${hue} 78% ${light}%)`;
}

function drawSpectrogram(canvas, payload) {
  clearSpectrogramCanvas(canvas);
  if (!payload || !Array.isArray(payload.values) || payload.values.length === 0) return;

  const values = payload.values;
  const melCount = values.length;
  const frameCount = Array.isArray(values[0]) ? values[0].length : 0;
  if (frameCount <= 0) return;

  const rect = canvas.getBoundingClientRect();
  const specCtx = canvas.getContext("2d");
  const minDb = Number.isFinite(payload.min_db) ? payload.min_db : -80;
  const maxDb = Number.isFinite(payload.max_db) ? payload.max_db : 0;
  const dbRange = Math.max(1e-6, maxDb - minDb);

  for (let x = 0; x < rect.width; x += 1) {
    const frame = Math.min(frameCount - 1, Math.floor((x / rect.width) * frameCount));
    for (let y = 0; y < rect.height; y += 1) {
      const mel = Math.min(melCount - 1, melCount - 1 - Math.floor((y / rect.height) * melCount));
      const db = Number(values[mel][frame]);
      const normalized = Number.isFinite(db) ? (db - minDb) / dbRange : 0;
      specCtx.fillStyle = spectrogramColor(normalized);
      specCtx.fillRect(x, y, 1, 1);
    }
  }
}

function drawStoredSpectrograms() {
  drawSpectrogram(els.targetSpectrogramCanvas, state.targetSpectrogram);
  drawSpectrogram(els.predictedSpectrogramCanvas, state.predictedSpectrogram);
}

function clearArtifacts() {
  state.lastResult = null;
  state.targetSpectrogram = null;
  state.predictedSpectrogram = null;
  els.targetAudio.removeAttribute("src");
  els.predictedAudio.removeAttribute("src");
  els.targetAudio.load();
  els.predictedAudio.load();
  els.patchJson.textContent = "{}";
  drawStoredSpectrograms();
  setArtifactStatus("No prediction yet");
}

async function fetchArtifact(path) {
  const response = await fetch(artifactUrl(path));
  if (!response.ok) {
    let message = `HTTP ${response.status}`;
    try {
      const payload = await response.json();
      message = payload.error?.message || message;
    } catch (_error) {
      // Non-JSON artifact errors still get a useful HTTP fallback.
    }
    throw new Error(message);
  }
  return response;
}

async function fetchJsonArtifact(path) {
  const response = await fetchArtifact(path);
  return response.json();
}

async function loadPredictionArtifacts(result) {
  const requiredPaths = [
    "target_crop_wav",
    "predicted_patch_json",
    "predicted_wav",
    "target_spectrogram",
    "predicted_spectrogram",
  ];
  for (const key of requiredPaths) {
    if (!result[key]) throw new Error(`Missing ${key}`);
  }

  els.targetAudio.src = artifactUrl(result.target_crop_wav);
  els.predictedAudio.src = artifactUrl(result.predicted_wav);
  els.targetAudio.load();
  els.predictedAudio.load();

  const [patch, targetSpectrogram, predictedSpectrogram] = await Promise.all([
    fetchJsonArtifact(result.predicted_patch_json),
    fetchJsonArtifact(result.target_spectrogram),
    fetchJsonArtifact(result.predicted_spectrogram),
  ]);

  state.targetSpectrogram = targetSpectrogram;
  state.predictedSpectrogram = predictedSpectrogram;
  els.patchJson.textContent = JSON.stringify(patch, null, 2);
  setParameterSummary(patch, result);
  drawStoredSpectrograms();
  setArtifactStatus("Artifacts loaded", "ok");
}

function artifactFileName(path, fallback) {
  const text = String(path || "");
  const parts = text.split(/[\\/]/).filter(Boolean);
  return parts[parts.length - 1] || fallback;
}

async function downloadArtifact(path, fallbackName) {
  if (!path) throw new Error("No artifact path available");
  const response = await fetchArtifact(path);
  const blob = await response.blob();
  const url = URL.createObjectURL(blob);
  const link = document.createElement("a");
  link.href = url;
  link.download = artifactFileName(path, fallbackName);
  document.body.appendChild(link);
  link.click();
  link.remove();
  URL.revokeObjectURL(url);
}

async function exportPatch() {
  try {
    await downloadArtifact(state.lastResult?.predicted_patch_json, "predicted_patch.json");
    setArtifactStatus("Predicted JSON exported", "ok");
  } catch (error) {
    setArtifactStatus(error.message, "error");
  }
}

async function exportWav() {
  try {
    await downloadArtifact(state.lastResult?.predicted_wav, "predicted.wav");
    setArtifactStatus("Predicted WAV exported", "ok");
  } catch (error) {
    setArtifactStatus(error.message, "error");
  }
}

async function openRunFolder() {
  try {
    const runDir = state.lastResult?.run_dir;
    if (!runDir) throw new Error("No run folder available");
    const response = await fetch(`${backendBaseUrl()}/open-folder`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ path: runDir }),
    });
    const payload = await response.json();
    if (!response.ok) {
      throw new Error(payload.error?.message || `HTTP ${response.status}`);
    }
    setArtifactStatus("Run folder opened", "ok");
  } catch (error) {
    setArtifactStatus(error.message, "error");
  }
}

async function checkBackend() {
  const backend = backendBaseUrl();
  setStatus("Checking", "busy");
  try {
    const response = await fetch(`${backend}/health`);
    if (!response.ok) throw new Error(`HTTP ${response.status}`);
    const payload = await response.json();
    setStatus(payload.status === "ok" ? "Online" : "Backend", payload.status === "ok" ? "ok" : "error");
  } catch (error) {
    setStatus("Offline", "error");
    setResponse({ error: error.message, desktop_backend: desktopBackendDiagnostic() || undefined });
  }
}

function predictPayload() {
  let cropEnd = Number(els.cropEnd.value);
  if (state.duration && Number.isFinite(cropEnd)) {
    cropEnd = Math.min(cropEnd, state.duration);
  }
  return {
    audio_path: els.backendAudioPath.value,
    model_path: els.modelPath.value,
    freq_hz: Number(els.freqHz.value),
    crop_start_seconds: Number(els.cropStart.value),
    crop_end_seconds: Number.isFinite(cropEnd) && cropEnd > 0 ? cropEnd : null,
    output_dir: els.outputDir.value,
    device: "cpu",
  };
}

function validatePredictPayload(payload) {
  if (!payload.audio_path) throw new Error("Audio path is required");
  if (!payload.model_path) throw new Error("Model path is required");
  if (!Number.isFinite(payload.freq_hz) || payload.freq_hz <= 0) throw new Error("Frequency must be positive");
  if (!Number.isFinite(payload.crop_start_seconds) || payload.crop_start_seconds < 0) throw new Error("Crop start is invalid");
  if (payload.crop_end_seconds !== null && payload.crop_end_seconds <= payload.crop_start_seconds) throw new Error("Crop end must be greater than start");
}

async function runPredict() {
  const backend = backendBaseUrl();
  const payload = predictPayload();
  try {
    validatePredictPayload(payload);
    els.predictButton.disabled = true;
    setStatus("Predicting", "busy");
    clearArtifacts();
    setArtifactStatus("Waiting for prediction", "busy");
    setResponse({ request: payload });
    const response = await fetch(`${backend}/predict`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });
    const responsePayload = await response.json();
    if (!response.ok) {
      throw new Error(responsePayload.error?.message || `HTTP ${response.status}`);
    }
    setStatus("Online", "ok");
    state.lastResult = responsePayload;
    setResponse(responsePayload);
    setResultSummary(responsePayload);
    try {
      await loadPredictionArtifacts(responsePayload);
    } catch (artifactError) {
      setArtifactStatus(artifactError.message, "error");
      setResponse({ response: responsePayload, artifact_error: artifactError.message });
    }
  } catch (error) {
    setStatus("Error", "error");
    setArtifactStatus(error.message || "Prediction failed", "error");
    setResponse({ error: error.message, request: payload });
  } finally {
    els.predictButton.disabled = false;
  }
}

function setResponse(payload) {
  els.responseJson.textContent = JSON.stringify(payload, null, 2);
}

function displayParamValue(value) {
  if (typeof value === "number") {
    return Number.isInteger(value) ? String(value) : value.toFixed(4);
  }
  if (value === undefined || value === null || value === "") {
    return "";
  }
  return String(value);
}

function setSummaryFields(fields) {
  els.resultSummary.innerHTML = "";
  for (const [label, value] of fields) {
    const dt = document.createElement("dt");
    const dd = document.createElement("dd");
    dt.textContent = label;
    dd.textContent = displayParamValue(value);
    els.resultSummary.append(dt, dd);
  }
}

function setResultSummary(result) {
  if (!result || !result.run_id) {
    setSummaryFields([["State", "No prediction yet"]]);
    return;
  }
  setSummaryFields([
    ["State", "Prediction complete"],
    ["Crop", `${formatSeconds(result.crop_start_seconds)} - ${formatSeconds(result.crop_end_seconds)}`],
  ]);
}

function setParameterSummary(patch, result) {
  setSummaryFields([
    ["Freq", patch.freq],
    ["Length", patch.length],
    ["Osc 1", `${patch.osc1_wave} @ ${displayParamValue(patch.osc1_level)}`],
    ["Osc 2", `${patch.osc2_wave} @ ${displayParamValue(patch.osc2_level)}`],
    ["Detune", patch.osc2_detune],
    ["Cutoff", patch.cutoff],
    ["Res", patch.resonance],
    ["ADSR", `${displayParamValue(patch.attack)} / ${displayParamValue(patch.decay)} / ${displayParamValue(patch.sustain)} / ${displayParamValue(patch.release)}`],
    ["Crop", `${formatSeconds(result.crop_start_seconds)} - ${formatSeconds(result.crop_end_seconds)}`],
  ]);
}

function bindEvents() {
  els.checkBackend.addEventListener("click", checkBackend);
  for (const element of [els.backendUrl, els.modelPath, els.outputDir]) {
    element.addEventListener("change", saveSettings);
  }
  els.dropZone.addEventListener("click", () => els.fileInput.click());
  els.dropZone.addEventListener("dragover", (event) => {
    event.preventDefault();
    els.dropZone.classList.add("dragging");
  });
  els.dropZone.addEventListener("dragleave", () => {
    els.dropZone.classList.remove("dragging");
  });
  els.dropZone.addEventListener("drop", async (event) => {
    event.preventDefault();
    els.dropZone.classList.remove("dragging");
    const file = event.dataTransfer.files[0];
    if (file) await loadAudioFile(file);
  });
  els.fileInput.addEventListener("change", async () => {
    const file = els.fileInput.files[0];
    if (file) await loadAudioFile(file);
  });
  els.cropStart.addEventListener("change", updateCropFromInputs);
  els.cropEnd.addEventListener("change", updateCropFromInputs);
  els.zoomLevel.addEventListener("input", updateZoomFromInput);
  els.fitCropZoom.addEventListener("click", fitZoomToCrop);
  els.resetZoom.addEventListener("click", resetZoom);
  els.noteName.addEventListener("change", applyNoteInput);
  els.playCrop.addEventListener("click", playCrop);
  els.stopPlayback.addEventListener("click", () => {
    stopPlayback();
    stopResultPlayback();
  });
  els.playTargetResult.addEventListener("click", () => playResultAudio("target"));
  els.playPredictedResult.addEventListener("click", () => playResultAudio("predicted"));
  els.stopResultPlayback.addEventListener("click", stopResultPlayback);
  els.exportPatch.addEventListener("click", exportPatch);
  els.exportWav.addEventListener("click", exportWav);
  els.openRunFolder.addEventListener("click", openRunFolder);
  els.predictButton.addEventListener("click", runPredict);
  els.canvas.addEventListener("mousedown", beginCropDrag);
  window.addEventListener("mousemove", moveCropDrag);
  window.addEventListener("mouseup", endCropDrag);
  window.addEventListener("resize", resizeCanvas);
}

function init() {
  applySettings();
  bindEvents();
  resizeCanvas();
  setResultSummary({});
  clearArtifacts();
  updateZoomDisplay();
  applyNoteInput();
  const startupDiagnostic = desktopBackendDiagnostic();
  if (startupDiagnostic) {
    setStatus("Backend issue", "error");
    setResponse({ desktop_backend: startupDiagnostic });
  }
}

init();
