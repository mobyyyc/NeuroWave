const runtimeDir = process.env.NEUROWAVE_RUNTIME_DIR || "runtime/python";

module.exports = {
  appId: "com.neurowave.app",
  productName: "NeuroWave",
  asar: true,
  npmRebuild: false,
  directories: {
    output: process.env.NEUROWAVE_OUTPUT_DIR || "dist",
  },
  files: [
    "app/**/*",
    "desktop/**/*",
    "package.json",
  ],
  extraResources: [
    {
      from: "scripts",
      to: "neurowave-python/scripts",
    },
    {
      from: "minisynth",
      to: "neurowave-python/minisynth",
    },
    {
      from: "requirements.txt",
      to: "neurowave-python/requirements.txt",
    },
    {
      from: "requirements-cuda.txt",
      to: "neurowave-python/requirements-cuda.txt",
    },
    {
      from: runtimeDir,
      to: "python-runtime",
      filter: ["**/*", "!pyvenv.cfg", "!.gitignore", "!.gitkeep"],
    },
    {
      from: "models/v3.5_noise_detune_loss.pt",
      to: "models/v3.5_noise_detune_loss.pt",
    },
  ],
  win: {
    signAndEditExecutable: false,
    target: [
      {
        target: "nsis-web",
        arch: ["x64"],
      },
    ],
  },
  nsisWeb: {
    artifactName: "${productName} Web Setup ${version}.${ext}",
  },
  publish: [
    {
      provider: "github",
      owner: "mobyyyc",
      repo: "NeuroWave",
      releaseType: "draft",
    },
  ],
};
