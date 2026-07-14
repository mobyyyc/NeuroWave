<!-- SEED: re-run $impeccable document once the website has code to capture the actual tokens and components. -->
---
name: NeuroWave Website
description: A precise, high-quality introduction to audio-to-synth prediction.
---

# Design System: NeuroWave Website

## Overview

**Creative North Star: "The Quiet Instrument Manual"**

NeuroWave should feel as composed and intentional as a finely made instrument, not like a
software dashboard or a sci-fi experiment. The visual system is predominantly light and
near-monochrome, with generous space, exact typography, and real product imagery carrying
the story. Its quality bar is Apple-like clarity: each element is necessary, readable, and
given room to matter.

Motion is choreographed but never ornamental. Entrances and scroll transitions should reveal
the workflow in a deliberate sequence. Every motion treatment must have a reduced-motion
equivalent that shows the same information immediately.

**Key Characteristics:**

- Light, near-monochrome precision rather than stark black-and-white contrast.
- EB Garamond provides the human, crafted display voice; the supporting sans remains quiet.
- Product screenshots, audio comparisons, and spectrograms are evidence, not decoration.
- Animation guides attention through the workflow without slowing it down.

## Colors

The palette is restrained: warm near-white surfaces, ink-toned text, and a single subtle
supporting accent to be resolved during implementation.

### Primary

- **Instrument Ink**: [near-black neutral, to be resolved during implementation]. Use for
  primary type, key controls, and the strongest visual anchors. Never use pure `#000`.

### Neutral

- **Paper Light**: [near-white neutral, to be resolved during implementation]. Use as the
  main canvas. Never use pure `#fff`.
- **Graphite Quiet**: [muted neutral, to be resolved during implementation]. Use for
  secondary copy, rules, and quiet interface detail.

**The Evidence Rule.** Color must never compete with screenshots, spectrograms, or audio
controls. A supporting accent appears only when it clarifies a state or action.

## Typography

**Display Font:** EB Garamond (with Georgia serif fallback)
**Body Font:** [supporting sans to be chosen during implementation]

**Character:** EB Garamond brings a quiet, editorial sense of craft to the product name and
major statements. It must not turn the site into a magazine pastiche: body copy and controls
stay clean, direct, and highly legible.

### Hierarchy

- **Display**: [fluid scale to be resolved during implementation]. Use only for decisive
  page statements and feature moments.
- **Headline**: [scale to be resolved during implementation]. Use for page and section
  structure without competing with the display treatment.
- **Title**: [scale to be resolved during implementation]. Use for cards only when cards are
  genuinely necessary, examples, and release information.
- **Body**: [scale to be resolved during implementation]. Keep reading measure below 75ch.
- **Label**: [scale to be resolved during implementation]. Use restrained weight and case;
  do not repeat tiny uppercase labels above every section.

**The Contrast Rule.** Hierarchy comes from confident scale, weight, and spacing contrast,
not from a crowded collection of type styles.

## Elevation

The system is flat by default. Depth comes from whitespace, subtle tonal separation, and
precise rules, not floating glass panels or heavy shadows. Any elevation used for hover,
audio controls, or sticky navigation must be soft, short-lived, and structurally useful.

**The Flat-By-Default Rule.** A surface earns elevation only when it is interactive or needs
to separate from moving content. Decorative shadow is prohibited.

## Components

### Buttons

- **Shape:** restrained corners, exact value to be resolved during implementation.
- **Primary:** ink-toned action on paper-light surface, with concise action-led copy.
- **Hover / Focus:** a clear contrast shift and visible focus treatment; no bounce or glow.
- **Secondary:** quiet text or outline treatment for supporting actions.

### Cards / Containers

- **Corner Style:** minimal rounding or square edges, selected for the final layout rather
  than applied indiscriminately.
- **Background:** tonal separation only when it improves grouping.
- **Shadow Strategy:** flat at rest; interaction-driven elevation only.
- **Internal Padding:** generous around screenshots and audio examples, tighter around facts.

### Audio Examples

- **Style:** labelled target and predicted controls with model/version context and source
  notes visible before playback.
- **State:** keyboard operable, with visible focus, readable timing, and a no-autoplay rule.

### Navigation

- **Style:** compact, calm, and content-led. Mobile navigation must preserve page hierarchy
  without hiding primary release information.

## Do's and Don'ts

### Do:

- **Do** use real app screenshots, spectrograms, and approved audio examples as the primary
  visual evidence.
- **Do** use EB Garamond sparingly for high-impact statements and pair it with a highly
  legible supporting sans once implementation begins.
- **Do** build choreographed transitions with a complete reduced-motion fallback.
- **Do** make release state, model version, limitations, and local-processing privacy clear.

### Don't:

- **Don't** use neon, cyberpunk, sci-fi control-room, or glowing-grid aesthetics.
- **Don't** use generic AI SaaS cards, oversized metrics, empty claims, or repeated
  icon-heading-text grids.
- **Don't** add complicated decorative elements, dense dashboards, glassmorphism, gradient
  text, or animation that delays reading.
- **Don't** use pure `#000` or `#fff`, or rely on pure monochrome contrast to create quality.
- **Don't** imply cloud inference, general audio recreation, signing, or model accuracy that
  NeuroWave cannot substantiate.
