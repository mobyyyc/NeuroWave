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
npm run build
```

Fonts are self-hosted through npm packages so the production build does not depend on a
Google Fonts network request.

## Deployment

Create one Vercel project from the repository and set its root directory to `website`.
Connect it to GitHub so branch changes receive preview deployments and `main` deploys
production after review.

The future Download page must link to a verified GitHub Release. The release owns the NSIS
web bootstrapper, its versioned payload, and `latest.yml`; Vercel hosts this website only.

See the repository `PLAN.md`, `PRODUCT.md`, and `DESIGN.md` for product, content, and design
requirements.
