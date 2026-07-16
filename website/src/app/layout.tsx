import type { Metadata } from "next";
import "./globals.css";
import { siteUrl } from "./site";

export const metadata: Metadata = {
  metadataBase: new URL(siteUrl),
  title: {
    default: "NeuroWave | Audio to editable synthesis",
    template: "%s | NeuroWave",
  },
  description: "Turn a clean one-note clip into an editable synthesizer patch with NeuroWave.",
  applicationName: "NeuroWave",
  keywords: ["audio to synth", "synthesizer patch", "sound design", "Windows audio app"],
  alternates: { canonical: "/" },
  openGraph: {
    type: "website",
    locale: "en_US",
    url: "/",
    siteName: "NeuroWave",
    title: "NeuroWave | Audio to editable synthesis",
    description: "Turn a clean one-note clip into an editable synthesizer patch with NeuroWave.",
  },
  twitter: {
    card: "summary_large_image",
    title: "NeuroWave | Audio to editable synthesis",
    description: "Turn a clean one-note clip into an editable synthesizer patch with NeuroWave.",
  },
  icons: {
    icon: "/icon.svg",
    shortcut: "/icon.svg",
  },
};

export default function RootLayout({ children }: Readonly<{ children: React.ReactNode }>) {
  return (
    <html lang="en">
      <body>{children}</body>
    </html>
  );
}
