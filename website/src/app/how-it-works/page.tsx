import type { Metadata } from "next";
import Link from "next/link";

export const metadata: Metadata = {
  title: "How NeuroWave works",
  description: "See how NeuroWave turns a clean one-note clip into an editable synthesizer patch on Windows.",
  alternates: { canonical: "/how-it-works" },
  openGraph: {
    title: "How NeuroWave works",
    description: "A clear, local workflow from one clean note to an editable synth patch.",
    url: "/how-it-works",
  },
  twitter: {
    title: "How NeuroWave works",
    description: "A clear, local workflow from one clean note to an editable synth patch.",
  },
};

const steps = [
  {
    number: "01",
    title: "Start with a clean note",
    detail: "Choose a short single-note WAV where the sound you want to study is easy to hear.",
  },
  {
    number: "02",
    title: "Keep the useful moment",
    detail: "Crop the clip to the stable part of the note, leaving out silence and unrelated transients.",
  },
  {
    number: "03",
    title: "Name its pitch",
    detail: "Enter the intended pitch so the model can concentrate on tone and envelope instead of guessing context.",
  },
  {
    number: "04",
    title: "Generate a starting patch",
    detail: "NeuroWave predicts editable synth settings and renders audio from that patch locally on your Windows machine.",
  },
  {
    number: "05",
    title: "Listen, look, reshape",
    detail: "Compare the rendered result and spectrogram, then keep refining the patch with your own judgement.",
  },
];

export default function HowItWorksPage() {
  return (
    <main className="method-page">
      <nav className="site-nav" aria-label="Main navigation">
        <Link className="wordmark" href="/" aria-label="NeuroWave home">
          Neuro<span>Wave</span>
        </Link>
        <div className="nav-links">
          <Link href="/how-it-works" aria-current="page">How it works</Link>
          <Link href="/download">Windows release</Link>
          <Link href="/privacy">Privacy</Link>
        </div>
      </nav>

      <section className="method-hero" aria-labelledby="method-heading">
        <p className="eyebrow reveal">The method</p>
        <div className="method-hero-grid">
          <h1 className="reveal reveal-one" id="method-heading">
            A short path from<br />
            <em>sound to control.</em>
          </h1>
          <p className="method-intro reveal reveal-two">
            NeuroWave is designed for a focused act of listening. Give it one clean note,
            state the pitch, and use the result as an editable place to begin.
          </p>
        </div>
      </section>

      <section className="method-ledger" aria-label="NeuroWave workflow">
        <div className="method-ledger-head">
          <p>From a clip you recognise</p>
          <p>To a patch you can change</p>
        </div>
        <ol>
          {steps.map((step, index) => (
            <li className={`reveal reveal-${Math.min(index + 1, 3)}`} key={step.number}>
              <span>{step.number}</span>
              <h2>{step.title}</h2>
              <p>{step.detail}</p>
            </li>
          ))}
        </ol>
      </section>

      <section className="method-note" aria-labelledby="model-heading">
        <div className="method-note-stamp" aria-hidden="true">
          <span>Model</span>
          <b>v3.5</b>
          <i>noise + detune</i>
        </div>
        <div>
          <p className="eyebrow">A useful boundary</p>
          <h2 id="model-heading">The model offers a starting point, not a verdict.</h2>
          <p>
            Results are strongest when the source is a clean, single-note clip with the correct
            pitch. The app keeps the output editable because your ear remains part of the process.
          </p>
        </div>
      </section>

      <section className="method-close" aria-labelledby="close-heading">
        <p className="eyebrow">Windows desktop app</p>
        <h2 id="close-heading">The listening stays local.</h2>
        <p>Audio is processed on your machine. The website does not receive the clip.</p>
        <Link className="button button-primary" href="/download">See Windows release status</Link>
      </section>

      <footer>
        <Link className="wordmark" href="/">Neuro<span>Wave</span></Link>
        <p>Audio is processed locally in the desktop app.</p>
        <p>Current model: v3.5_noise_detune_loss</p>
      </footer>
    </main>
  );
}
