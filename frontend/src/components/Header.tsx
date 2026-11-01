"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";

function ThemeToggle() {
  // Reads the live theme at click time, so no React state is needed.
  function toggle() {
    const root = document.documentElement;
    const current = root.dataset.theme;
    const dark =
      current === "dark" || (current === undefined && window.matchMedia("(prefers-color-scheme: dark)").matches);
    const next = dark ? "light" : "dark";
    root.dataset.theme = next;
    try {
      localStorage.setItem("rhf-theme", next);
    } catch {
      // private mode: the toggle still works for this page view
    }
  }
  return (
    <button onClick={toggle} className="rounded-full border border-line px-3 py-1 font-mono text-xs hover:border-rust" aria-label="Toggle light or dark theme">
      ◐ theme
    </button>
  );
}

export function Header() {
  const path = usePathname();
  const link = (href: string, label: string) => (
    <Link
      href={href}
      aria-current={path === href ? "page" : undefined}
      className={`font-mono text-sm hover:text-rust ${path === href ? "text-rust" : "text-ink-soft"}`}
    >
      {label}
    </Link>
  );
  return (
    <header className="mx-auto flex max-w-5xl items-center justify-between gap-4 px-4 py-5 sm:px-6">
      <Link href="/" className="flex items-center gap-2">
        <svg width="34" height="22" viewBox="0 0 34 22" aria-hidden>
          <ellipse cx="17" cy="13" rx="16" ry="8" fill="var(--shaft)" />
          <ellipse cx="17" cy="11" rx="12" ry="5" fill="var(--lantern)" opacity="0.25" />
          <path d="M11 6 q1 -6 3 -5 q1 1 -1 6 M20 6 q2 -6 4 -4 q0 2 -3 5" stroke="var(--ink)" strokeWidth="1.6" fill="none" strokeLinecap="round" />
        </svg>
        <span className="font-display text-xl italic tracking-tight">Rabbit Hole Finder</span>
      </Link>
      <nav className="flex items-center gap-4">
        {link("/", "find")}
        {link("/map", "map")}
        <ThemeToggle />
      </nav>
    </header>
  );
}
