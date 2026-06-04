const state = {
  audioBuffer: null,
  audioFileName: "",
  duration: 0,
  cropStart: 0,
  cropEnd: 0,
  dragMode: null,
  playbackSource: null,
  audioContext: null,
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
};

const ctx = els.canvas.getContext("2d");

function setStatus(text, kind = "idle") {
  els.backendStatus.textContent = text;
  els.backendStatus.className = `status-pill status-${kind}`;
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
}

function secondsToX(seconds) {
  const width = els.canvas.getBoundingClientRect().width;
  if (!state.duration) return 0;
  return (seconds / state.duration) * width;
}

function xToSeconds(x) {
  const width = els.canvas.getBoundingClientRect().width;
  if (!state.duration || width <= 0) return 0;
  return clamp((x / width) * state.duration, 0, state.duration);
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
  const samplesPerPixel = Math.max(1, Math.floor(data.length / width));

  ctx.strokeStyle = "#8fd3ff";
  ctx.lineWidth = 1;
  ctx.beginPath();
  for (let x = 0; x < width; x += 1) {
    const start = x * samplesPerPixel;
    const end = Math.min(start + samplesPerPixel, data.length);
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

  state.audioBuffer = decoded;
  state.audioFileName = file.name;
  state.duration = decoded.duration;
  state.cropStart = 0;
  state.cropEnd = decoded.duration;

  els.fileName.textContent = file.name;
  els.fileMeta.textContent = `${decoded.numberOfChannels} ch | ${decoded.sampleRate} Hz | ${formatSeconds(decoded.duration)} s`;
  els.cropStart.value = formatSeconds(state.cropStart);
  els.cropEnd.value = formatSeconds(state.cropEnd);
  if (!els.backendAudioPath.value) {
    els.backendAudioPath.value = `playground/${file.name}`;
  }
  drawWaveform();
}

function updateCropInputs() {
  els.cropStart.value = formatSeconds(state.cropStart);
  els.cropEnd.value = formatSeconds(state.cropEnd);
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

async function checkBackend() {
  const backend = els.backendUrl.value.replace(/\/$/, "");
  setStatus("Checking", "busy");
  try {
    const response = await fetch(`${backend}/health`);
    if (!response.ok) throw new Error(`HTTP ${response.status}`);
    const payload = await response.json();
    setStatus(payload.status === "ok" ? "Online" : "Backend", payload.status === "ok" ? "ok" : "error");
  } catch (error) {
    setStatus("Offline", "error");
    setResponse({ error: error.message });
  }
}

function predictPayload() {
  const cropEnd = Number(els.cropEnd.value);
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
  const backend = els.backendUrl.value.replace(/\/$/, "");
  const payload = predictPayload();
  try {
    validatePredictPayload(payload);
    els.predictButton.disabled = true;
    setStatus("Predicting", "busy");
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
    setResponse(responsePayload);
    setResultSummary(responsePayload);
  } catch (error) {
    setStatus("Error", "error");
    setResponse({ error: error.message, request: payload });
  } finally {
    els.predictButton.disabled = false;
  }
}

function setResponse(payload) {
  els.responseJson.textContent = JSON.stringify(payload, null, 2);
}

function setResultSummary(result) {
  const fields = [
    ["Run", result.run_id],
    ["Crop", `${formatSeconds(result.crop_start_seconds)} - ${formatSeconds(result.crop_end_seconds)}`],
    ["Patch", result.predicted_patch_json],
    ["WAV", result.predicted_wav],
    ["Target Spec", result.target_spectrogram],
    ["Pred Spec", result.predicted_spectrogram],
  ];
  els.resultSummary.innerHTML = "";
  for (const [label, value] of fields) {
    const dt = document.createElement("dt");
    const dd = document.createElement("dd");
    dt.textContent = label;
    dd.textContent = value || "";
    els.resultSummary.append(dt, dd);
  }
}

function bindEvents() {
  els.checkBackend.addEventListener("click", checkBackend);
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
  els.noteName.addEventListener("change", applyNoteInput);
  els.playCrop.addEventListener("click", playCrop);
  els.stopPlayback.addEventListener("click", stopPlayback);
  els.predictButton.addEventListener("click", runPredict);
  els.canvas.addEventListener("mousedown", beginCropDrag);
  window.addEventListener("mousemove", moveCropDrag);
  window.addEventListener("mouseup", endCropDrag);
  window.addEventListener("resize", resizeCanvas);
}

function init() {
  bindEvents();
  resizeCanvas();
  setResultSummary({});
  applyNoteInput();
}

init();
