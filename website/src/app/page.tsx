const ArrowUpRight = () => (
  <svg aria-hidden="true" viewBox="0 0 16 16" fill="none">
    <path d="M3 13 13 3M6 3h7v7" stroke="currentColor" strokeWidth="1.5" />
  </svg>
);

const Waveform = () => (
  <svg className="waveform" viewBox="0 0 660 136" fill="none" aria-hidden="true">
    <path
      d="M0 68h26l8-10 9 30 9-58 9 78 9-95 9 102 9-81 9 56 9-27 9 11h27l9-7 9 18 9-41 9 50 9-67 9 70 9-49 9 27 9-8h42l9 5 9-14 9 29 9-52 9 66 9-77 9 69 9-50 9 29 9-10h34l8-6 9 15 9-32 9 40 9-48 9 43 9-29 9 18h32l9-6 9 12 9-22 9 21 9-13 9 7h46"
      stroke="currentColor"
      strokeWidth="2"
      strokeLinecap="round"
      strokeLinejoin="round"
    />
  </svg>
);

export default function Home() {
  return (
    <main>
      <nav className="site-nav" aria-label="Main navigation">
        <a className="wordmark" href="#top" aria-label="NeuroWave home">
          Neuro<span>Wave</span>
        </a>
        <div className="nav-links">
          <a href="#workflow">How it works</a>
          <a href="#release">Windows release</a>
        </div>
      </nav>

      <section className="hero" id="top" aria-labelledby="hero-heading">
        <div className="hero-copy reveal reveal-one">
          <p className="eyebrow">Audio to editable synthesis</p>
          <h1 id="hero-heading">
            Hear a sound.<br />
            <em>Make it yours.</em>
          </h1>
          <p className="hero-summary">
            NeuroWave listens to a clean one-note clip, predicts an editable synth patch, and
            renders a comparison you can refine.
          </p>
          <div className="hero-actions">
            <a className="button button-primary" href="#release">
              Windows release status <ArrowUpRight />
            </a>
            <a className="text-link" href="#workflow">
              See the process <span aria-hidden="true">↓</span>
            </a>
          </div>
        </div>

        <div className="signal-stage reveal reveal-two" aria-label="Illustration of an audio signal becoming a synthesizer patch">
          <div className="signal-label signal-label-top">Input clip</div>
          <div className="signal-label signal-label-bottom">Editable patch</div>
          <div className="signal-rule" />
          <div className="signal-source"><Waveform /></div>
          <div className="signal-crop"><span>crop</span></div>
          <div className="signal-patch">
            <span>osc</span><b>triangle</b>
            <span>filter</span><b>1.8 kHz</b>
            <span>envelope</span><b>short decay</b>
          </div>
          <p className="signal-caption">A visible chain, not a black box.</p>
        </div>
      </section>

      <section className="proof-band" aria-label="NeuroWave principles">
        <p>Local processing</p><span aria-hidden="true">/</span><p>Windows desktop</p><span aria-hidden="true">/</span><p>Editable results</p>
      </section>

      <section className="workflow-section" id="workflow" aria-labelledby="workflow-heading">
        <div className="section-intro reveal">
          <p className="eyebrow">A clear path from sound to control</p>
          <h2 id="workflow-heading">The result stays in your hands.</h2>
        </div>
        <ol className="workflow-list">
          <li className="reveal reveal-one"><span>01</span><div><h3>Bring a clean note</h3><p>Import a short, single-note WAV and select the part that matters.</p></div></li>
          <li className="reveal reveal-two"><span>02</span><div><h3>Set the musical context</h3><p>Confirm the pitch, then let the model focus on the character of the sound.</p></div></li>
          <li className="reveal reveal-three"><span>03</span><div><h3>Inspect and reshape</h3><p>Compare rendered audio, spectrograms, and the predicted patch. Export what you want to keep.</p></div></li>
        </ol>
      </section>

      <section className="release-section" id="release" aria-labelledby="release-heading">
        <p className="eyebrow">Windows desktop app</p>
        <div className="release-grid">
          <h2 id="release-heading">The first public release is being prepared.</h2>
          <div>
            <p>The installer, bundled runtime, and model are validated on clean Windows. The public release will appear here with clear installation notes.</p>
            <p className="release-note">For best results, start with a clean one-note clip and correct pitch.</p>
          </div>
        </div>
      </section>

      <footer>
        <a className="wordmark" href="#top">Neuro<span>Wave</span></a>
        <p>Audio is processed locally in the desktop app.</p>
        <p>Current model: v3.5_noise_detune_loss</p>
      </footer>
    </main>
  );
}
