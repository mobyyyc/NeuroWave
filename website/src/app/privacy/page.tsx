import type { Metadata } from "next";
import Link from "next/link";

export const metadata: Metadata = {
  title: "Privacy note",
  description: "How NeuroWave handles audio in the static website and local Windows desktop app.",
  alternates: { canonical: "/privacy" },
  openGraph: {
    title: "NeuroWave privacy note",
    description: "Audio stays on your machine. NeuroWave does not provide browser uploads or cloud inference.",
    url: "/privacy",
  },
};

export default function PrivacyPage() {
  return (
    <main className="privacy-page">
      <nav className="site-nav" aria-label="Main navigation">
        <Link className="wordmark" href="/" aria-label="NeuroWave home">
          Neuro<span>Wave</span>
        </Link>
        <div className="nav-links">
          <Link href="/how-it-works">How it works</Link>
          <Link href="/download">Windows release</Link>
          <Link href="/privacy" aria-current="page">Privacy</Link>
        </div>
      </nav>

      <section className="privacy-hero" aria-labelledby="privacy-heading">
        <p className="eyebrow reveal">Privacy note</p>
        <div className="privacy-hero-grid">
          <h1 className="reveal reveal-one" id="privacy-heading">
            Your audio stays<br />
            <em>with you.</em>
          </h1>
          <p className="privacy-intro reveal reveal-two">
            NeuroWave is a local Windows desktop workflow. The website does not receive audio,
            and the desktop app does not upload it for inference.
          </p>
        </div>
      </section>

      <section className="privacy-ledger" aria-label="Audio handling summary">
        <div className="privacy-ledger-head">
          <p>Where the work happens</p>
          <p>What that means for your clip</p>
        </div>
        <dl>
          <div className="reveal reveal-one">
            <dt>Website</dt>
            <dd>This static site has no audio upload, browser inference, account system, or audio database.</dd>
          </div>
          <div className="reveal reveal-two">
            <dt>Desktop app</dt>
            <dd>Prediction and comparison run locally on your Windows machine using the bundled application runtime.</dd>
          </div>
          <div className="reveal reveal-three">
            <dt>Local files</dt>
            <dd>Imported inputs, prediction runs, and backend logs are written under <code>%LOCALAPPDATA%\NeuroWave</code>.</dd>
          </div>
        </dl>
      </section>

      <section className="privacy-note" aria-labelledby="privacy-note-heading">
        <p className="eyebrow">If the product changes</p>
        <h2 id="privacy-note-heading">New data collection would be stated before it exists.</h2>
        <p>
          A future contact form, analytics service, cloud feature, or account system would require
          an updated privacy notice before it collects user information or audio.
        </p>
        <Link className="text-link privacy-link" href="/">Return to overview <span aria-hidden="true">↓</span></Link>
      </section>

      <footer>
        <Link className="wordmark" href="/">Neuro<span>Wave</span></Link>
        <p>Audio is processed locally in the desktop app.</p>
        <p>Current model: v3.5_noise_detune_loss</p>
      </footer>
    </main>
  );
}
