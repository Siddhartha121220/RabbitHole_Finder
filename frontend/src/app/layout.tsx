import type { Metadata, Viewport } from "next";
import { Fraunces, IBM_Plex_Mono, Instrument_Sans } from "next/font/google";
import { Header } from "@/components/Header";
import "./globals.css";

const fraunces = Fraunces({ subsets: ["latin"], variable: "--font-fraunces", style: ["normal", "italic"] });
const instrument = Instrument_Sans({ subsets: ["latin"], variable: "--font-instrument" });
const plexMono = IBM_Plex_Mono({ subsets: ["latin"], weight: ["400", "500"], variable: "--font-plex-mono" });

export const metadata: Metadata = {
  title: "Rabbit Hole Finder",
  description:
    "Tell it a few things you love; it finds the Wikipedia rabbit hole waiting next door, how deep it goes, and a trail to start with.",
};

export const viewport: Viewport = {
  themeColor: [
    { media: "(prefers-color-scheme: light)", color: "#f4ede1" },
    { media: "(prefers-color-scheme: dark)", color: "#17110c" },
  ],
};

// Apply a saved theme before first paint (no flash). The value is only "light" or "dark".
const THEME_SCRIPT = `try{var t=localStorage.getItem("rhf-theme");if(t==="light"||t==="dark")document.documentElement.dataset.theme=t}catch(e){}`;

export default function RootLayout({ children }: Readonly<{ children: React.ReactNode }>) {
  return (
    <html lang="en" suppressHydrationWarning className={`${fraunces.variable} ${instrument.variable} ${plexMono.variable}`}>
      <head>
        <script dangerouslySetInnerHTML={{ __html: THEME_SCRIPT }} />
      </head>
      <body className="strata font-sans antialiased">
        <Header />
        <main className="mx-auto max-w-5xl px-4 pb-24 sm:px-6">{children}</main>
        <footer className="mx-auto max-w-5xl px-4 pb-10 font-mono text-xs text-ink-soft sm:px-6">
          Built from Wikipedia clickstream data (CC0) and article text (CC BY-SA). Not affiliated with
          the Wikimedia Foundation.
        </footer>
      </body>
    </html>
  );
}
