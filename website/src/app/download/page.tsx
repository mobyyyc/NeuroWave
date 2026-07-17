import type { Metadata } from "next";
import Link from "next/link";
import { currentRelease } from "../site";

export const metadata: Metadata = {
  title: "Download for Windows",
  description: "NeuroWave Windows requirements, installer steps, and current release status.",
  alternates: { canonical: "/download" },
  openGraph: {
    title: "Download NeuroWave for Windows",
    description: "Windows requirements and verified release status for the NeuroWave desktop app.",
    url: "/download",
  },
};

const requirements = [
  ["Platform", "Windows x64. NeuroWave is a Windows desktop application for this first release."],
  ["Connection", "An internet connection is needed during installation so the NSIS web installer can retrieve its versioned runtime payload."],
  ["Acceleration", "The app runs on CPU. Compatible NVIDIA CUDA hardware can be used by the bundled PyTorch runtime when available."],
  ["Privacy", "Your source clip stays on your machine. The website and desktop app do not upload it for inference."],
];

const installationSteps = [
  ["01", "Get the verified installer", "Use the versioned GitHub Release linked here when it is published. Do not rely on mirrors or renamed copies."],
  ["02", "Let setup finish", "The installer downloads and extracts the large bundled Python and ML dependencies once, rather than making every app launch unpack them again."],
  ["03", "Start with one clean note", "Import a short, single-note clip, enter the correct pitch, then use the predicted patch as an editable starting point."],
];

export default function DownloadPage() {
  return (
    <main className="download-page">
      <nav className="site-nav" aria-label="Main navigation">
        <Link className="wordmark" href="/" aria-label="NeuroWave home">
          Neuro<span>Wave</span>
        </Link>
        <div className="nav-links">
          <Link href="/how-it-works">How it works</Link>
          <Link href="/download" aria-current="page">Windows release</Link>
          <Link href="/privacy">Privacy</Link>
        </div>
      </nav>

      <section className="download-hero" aria-labelledby="download-heading">
        <p className="eyebrow reveal">Windows release</p>
        <div className="download-hero-grid">
          <h1 className="reveal reveal-one" id="download-heading">
            Install once.<br />
            <em>Then listen.</em>
          </h1>
          <p className="download-intro reveal reveal-two">
            NeuroWave is preparing its first Windows release. The installer is validated on a
            clean machine and will appear here only when its complete GitHub Release is ready.
          </p>
        </div>
      </section>

      <section className="download-status" aria-labelledby="status-heading">
        <div>
          <p className="eyebrow">Release {currentRelease.version}</p>
          <h2 id="status-heading">{currentRelease.heading}</h2>
        </div>
        <div>
          <p>{currentRelease.summary}</p>
          {currentRelease.isPublic && currentRelease.githubReleaseUrl ? (
            <a className="button button-primary download-action" href={currentRelease.githubReleaseUrl}>
              Download NeuroWave {currentRelease.version}
            </a>
          ) : (
            <Link className="text-link download-action" href="/changelog">
              Read preparation notes <span aria-hidden="true">↗</span>
            </Link>
          )}
        </div>
      </section>

      <section className="download-ledger" aria-label="Windows requirements">
        <div className="download-ledger-head">
          <p>Before installation</p>
          <p>What to expect</p>
        </div>
        <dl>
          {requirements.map(([term, detail], index) => (
            <div className={`reveal reveal-${Math.min(index + 1, 3)}`} key={term}>
              <dt>{term}</dt>
              <dd>{detail}</dd>
            </div>
          ))}
        </dl>
      </section>

      <section className="install-sequence" aria-labelledby="install-heading">
        <div className="install-sequence-intro">
          <p className="eyebrow">The installer</p>
          <h2 id="install-heading">Large dependencies, handled once.</h2>
        </div>
        <ol>
          {installationSteps.map(([number, title, detail]) => (
            <li key={number}>
              <span>{number}</span>
              <h3>{title}</h3>
              <p>{detail}</p>
            </li>
          ))}
        </ol>
      </section>

      <section className="download-note" aria-labelledby="limitations-heading">
        <p className="eyebrow">Current model boundary</p>
        <h2 id="limitations-heading">A considered starting point, not instrument reconstruction.</h2>
        <p>
          The bundled <strong>v3.5_noise_detune_loss</strong> model is strongest with clean,
          single-note input and an accurate pitch. It predicts an editable synth patch, not a
          definitive recreation of any sound.
        </p>
        <Link className="text-link download-link" href="/privacy">Read the privacy note <span aria-hidden="true">↗</span></Link>
      </section>

      <footer>
        <Link className="wordmark" href="/">Neuro<span>Wave</span></Link>
        <p>Audio is processed locally in the desktop app.</p>
        <p>Current model: v3.5_noise_detune_loss</p>
      </footer>
    </main>
  );
}
