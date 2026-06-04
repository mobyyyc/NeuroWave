const { contextBridge } = require("electron");

contextBridge.exposeInMainWorld("neurowaveDesktop", {
  isDesktop: true,
  platform: process.platform,
  settings: {
    backendUrl: new URLSearchParams(window.location.search).get("backendUrl"),
    modelPath: new URLSearchParams(window.location.search).get("modelPath"),
    outputDir: new URLSearchParams(window.location.search).get("outputDir"),
  },
});
