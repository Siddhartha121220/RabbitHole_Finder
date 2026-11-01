import type { TrailOut } from "@/lib/api";
import { cleanTitle } from "@/lib/format";

/** The trail as a descending staircase: each step sits a little deeper. */
export function TrailView({ trail }: { trail: TrailOut }) {
  return (
    <section aria-labelledby="trail-title">
      <h2 id="trail-title" className="font-display text-2xl">
        Your trail down
      </h2>
      {trail.intro && <p className="mt-1 text-ink-soft">{trail.intro}</p>}
      <ol className="mt-5 space-y-3">
        {trail.steps.map((s, i) => (
          <li
            key={s.id}
            className="rise relative rounded-md border border-line bg-card p-4"
            style={{ marginLeft: `min(${i * 1.25}rem, ${i * 3}vw)`, animationDelay: `${i * 80}ms` }}
            data-testid="trail-step"
          >
            <span className="absolute -left-3 top-4 flex h-6 w-6 items-center justify-center rounded-full bg-shaft font-mono text-xs text-lantern ring-2 ring-surface">
              {i + 1}
            </span>
            <a href={s.url} target="_blank" rel="noreferrer" className="font-display text-lg hover:text-rust hover:underline">
              {cleanTitle(s.title)}
            </a>
            <p className="mt-1 text-sm text-ink-soft">{s.why}</p>
          </li>
        ))}
      </ol>
      <p className="mt-4 font-mono text-xs text-ink-soft">
        {trail.model === "template"
          ? "Trail built from the articles' own opening lines (no LLM)."
          : `Explanations written by ${trail.model} from passages of these articles; every link is checked against our article table.`}
      </p>
    </section>
  );
}
