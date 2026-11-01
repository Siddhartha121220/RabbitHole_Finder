import { cleanTitle } from "@/lib/format";
import type { ArticleRef } from "@/lib/api";

/** Seed -> ... -> entry article, as real links. Paths come only from the click graph. */
export function BridgePath({ path }: { path: ArticleRef[] }) {
  if (path.length < 2) return null;
  return (
    <ol className="flex flex-wrap items-center gap-x-1.5 gap-y-1 font-mono text-xs" aria-label="How readers get there">
      {path.map((a, i) => (
        <li key={a.id} className="flex items-center gap-1.5">
          {i > 0 && <span aria-hidden className="text-rust">↘</span>}
          <a
            href={a.url}
            target="_blank"
            rel="noreferrer"
            className={`rounded px-1.5 py-0.5 hover:underline ${i === 0 ? "bg-moss/15 text-moss" : i === path.length - 1 ? "bg-rust/15 text-rust" : "bg-line/40"}`}
          >
            {cleanTitle(a.title)}
          </a>
        </li>
      ))}
    </ol>
  );
}
