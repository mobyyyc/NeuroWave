---
name: NeuroWave Website
description: A precise, high-quality introduction to audio-to-synth prediction.
colors:
  paper: "#f6f5f1"
  paper-deep: "#ebe9e2"
  ink: "#171716"
  graphite: "#5f5e59"
  quiet: "#a7a49b"
  hairline: "#cfcdc4"
  signal: "#46605a"
typography:
  display:
    fontFamily: "EB Garamond Variable, Georgia, serif"
    fontSize: "clamp(4rem, 8vw, 7.7rem)"
    fontWeight: 500
    lineHeight: 0.82
    letterSpacing: "-0.067em"
  body:
    fontFamily: "Manrope Variable, Arial, sans-serif"
    fontSize: "0.94rem"
    fontWeight: 400
    lineHeight: 1.75
  label:
    fontFamily: "Manrope Variable, Arial, sans-serif"
    fontSize: "0.67rem"
    fontWeight: 700
    letterSpacing: "0.12em"
rounded:
  control: "0px"
spacing:
  compact: "16px"
  control: "28px"
  section: "150px"
components:
  button-primary:
    backgroundColor: "{colors.ink}"
    textColor: "{colors.paper}"
    rounded: "{rounded.control}"
    padding: "14px 17px"
  button-primary-hover:
    backgroundColor: "{colors.signal}"
    textColor: "{colors.paper}"
    rounded: "{rounded.control}"
    padding: "14px 17px"
---

# Design System: NeuroWave Website

## Overview

**Creative North Star: "The Quiet Instrument Manual"**

The site is a light, near-monochrome brand surface built around proof, not spectacle. It has
the visual composure of a finely made instrument manual: sparse, exact, and deliberate.
Product screenshots, audio examples, and spectrograms will carry the evidence as those assets
arrive. The page should never feel like a software dashboard or a sci-fi experiment.

The hero creates one visual chain from waveform to editable patch, then the rest of the page
slows down into clear, generous sections. Motion is choreographed to reveal that sequence,
but it disappears under reduced-motion preferences without withholding any information.

**Key Characteristics:**

- Warm paper surfaces and ink-toned structure, never stark pure black on pure white.
- EB Garamond is the crafted display voice; Manrope keeps controls and explanation precise.
- Flat structure, fine rules, and tonal layers replace floating panels and decorative shadow.
- Every call to action reflects actual release state.

## Colors

The palette is quietly tactile: mineral paper, soft graphite, and a deep muted green that
signals editable sound without turning into a product accent spectacle.

### Primary

- **Instrument Ink**: use the `ink` token for page anchors, primary actions, and high-contrast
  structural rules. It is never a full black.
- **Signal Green**: use the `signal` token for the predicted/editable transformation, focused
  emphasis, and stateful links. It must remain a minority of any viewport.

### Neutral

- **Paper Light**: use `paper` as the canvas and body text contrast surface.
- **Paper Deep**: use `paper-deep` only behind the waveform-to-patch illustration.
- **Graphite Quiet**: use `graphite` for explanatory copy and quiet navigation.
- **Fine Rule**: use `hairline` for structural separators, never as card decoration.

**The Evidence Rule.** Signal Green clarifies a transformation or action. It must never
compete with product screenshots, spectrograms, or audio controls.

## Typography

**Display Font:** EB Garamond Variable (with Georgia fallback)
**Body Font:** Manrope Variable (with Arial fallback)

**Character:** EB Garamond adds a human, crafted note to decisive statements. Manrope makes
the explanation and interactive surface calm and legible. The pairing is not an editorial
costume: body copy remains plainspoken, and labels never become a repeated decorative system.

### Hierarchy

- **Display** (500, `clamp(4rem, 8vw, 7.7rem)`, 0.82): decisive hero and release statements.
- **Headline** (500, `clamp(3rem, 5vw, 5.6rem)`, 0.9): section-level promise statements.
- **Title** (500, 1.7rem): workflow steps and compact content moments.
- **Body** (400, 0.94rem, 1.75): explanation, capped near 42ch for the denser narrative blocks.
- **Label** (700, 0.67rem, 0.12em): short contextual labels only.

**The Contrast Rule.** Hierarchy comes from confident scale, weight, and spatial contrast,
never from a crowded collection of font styles.

## Elevation

Surfaces are flat by default. Depth is created through paper-tone contrast, fine rules, and
generous whitespace. The patch panel uses a single offset shadow as a diagrammatic cue, not
as a reusable card treatment.

**The Flat-By-Default Rule.** Elevation appears only when an object must separate from an
illustration or respond to interaction. Glassmorphism and ambient floating shadows are
prohibited.

## Components

### Buttons

- **Shape:** square (`0px`) and concise, with no pill treatment.
- **Primary:** Instrument Ink background with Paper Light text and `14px 17px` padding.
- **Hover / Focus:** Signal Green on hover with a 2px upward motion; keyboard focus uses a
  visible Signal Green outline with 4px offset.
- **Secondary:** text-led link with a subtle Signal Green color shift.

### Cards / Containers

- **Corner Style:** square by default. Do not introduce rounded card grids.
- **Background:** tonal separation only when grouping is needed, such as the signal stage.
- **Shadow Strategy:** no general card shadow. The signal patch alone uses an offset diagram
  shadow to read as a generated object.
- **Internal Padding:** vary deliberately, from 20px for dense patch data to 150px section
  breathing room on desktop.

### Audio Examples

- **Style:** target and predicted controls must show model/version context and source notes
  before playback.
- **State:** controls are keyboard operable, visibly focused, and never autoplay.

### Navigation

- **Style:** compact, content-led, and kept to the two primary destinations until more pages
  exist. Mobile preserves the same visible hierarchy rather than hiding essential links.

## Do's and Don'ts

### Do:

- **Do** use the `paper`, `ink`, and `signal` tokens exactly as defined in frontmatter.
- **Do** show the product workflow through real screenshots, spectrograms, and approved audio
  examples as they become available.
- **Do** use choreographed reveals with a complete reduced-motion fallback.
- **Do** make local processing, model version, limitations, and release state explicit.

### Don't:

- **Don't** use neon, cyberpunk, sci-fi control-room, or glowing-grid aesthetics.
- **Don't** use generic AI SaaS cards, oversized metrics, empty claims, or repeated
  icon-heading-text grids.
- **Don't** add complicated decorative elements, dense dashboards, glassmorphism, gradient
  text, or animation that delays reading.
- **Don't** use pure `#000` or `#fff`, or turn the EB Garamond display face into a magazine
  pastiche.
- **Don't** imply cloud inference, general audio recreation, signing, or model accuracy that
  NeuroWave cannot substantiate.
