const { contextBridge } = require("electron");

contextBridge.exposeInMainWorld("neurowaveDesktop", {
  isDesktop: true,
  platform: process.platform,
});
