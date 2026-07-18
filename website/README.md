# NeuroWave Website

Static-first Next.js marketing site for NeuroWave. It explains the local Windows desktop app,
shows approved examples, and links to verified desktop releases. It does not upload audio,
run browser inference, or host the multi-gigabyte Windows installer payload.

## Local Development

From this directory:

```powershell
npm install
npm run dev
```

Open `http://localhost:3000`.

Run release checks before committing:

```powershell
npm run lint
npm run typecheck
npm run build
```

To preview the production build locally, build first, then run:

```powershell
npm run preview
```

Open `http://localhost:3000`. Use `Ctrl+C` to stop either local server.

Fonts are self-hosted through npm packages so the production build does not depend on a
Google Fonts network request.

## Deployment

Create one Vercel project from the repository and set its root directory to `website`.
Connect it to GitHub so branch changes receive preview deployments and `main` deploys
production after review.

The future Download page must link to a verified GitHub Release. The release owns the NSIS
web bootstrapper, its versioned payload, and `latest.yml`; Vercel hosts this website only.

## Verification Checklist

Before merging or publishing a website change:

- Run `npm run lint`, `npm run typecheck`, and `npm run build`.
- Use `npm run dev` or `npm run preview` to check the changed route on desktop and at a narrow
  mobile viewport. Confirm there is no horizontal overflow.
- Check visible focus states and keyboard navigation for the changed links or controls.
- Confirm the release CTA matches `src/app/site.ts`: no download is shown before the GitHub
  Release URL is verified.
- Confirm privacy and contact copy still states that the site does not upload audio.
- After a Git-driven Vercel production build is ready, verify the public route. The manual
  `neurowave-synth.vercel.app` alias must be reassigned to the new deployment until a custom
  domain replaces this workflow.

See the repository `PLAN.md`, `PRODUCT.md`, and `DESIGN.md` for product, content, and design
requirements.
