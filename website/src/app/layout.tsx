import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "NeuroWave | Audio to editable synthesis",
  description: "Turn a clean one-note clip into an editable synthesizer patch with NeuroWave.",
};

export default function RootLayout({ children }: Readonly<{ children: React.ReactNode }>) {
  return (
    <html lang="en">
      <body>{children}</body>
    </html>
  );
}
