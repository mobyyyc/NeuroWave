const { contextBridge, ipcRenderer, webUtils } = require("electron");

contextBridge.exposeInMainWorld("neurowaveDesktop", {
  isDesktop: true,
  platform: process.platform,
  pathForFile(file) {
    return webUtils.getPathForFile(file);
  },
  importAudioFile(name, bytes) {
    return ipcRenderer.invoke("neurowave:import-audio", { name, bytes });
  },
  settings: {
    backendUrl: new URLSearchParams(window.location.search).get("backendUrl"),
    modelPath: new URLSearchParams(window.location.search).get("modelPath"),
    outputDir: new URLSearchParams(window.location.search).get("outputDir"),
  },
});
