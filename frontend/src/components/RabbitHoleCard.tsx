import Link from "next/link";
import type { Recommendation } from "@/lib/api";
import { cleanTitle, splitLabel } from "@/lib/format";
import { BridgePath } from "./BridgePath";
import { DepthMeter } from "./DepthMeter";
import { FeedbackButtons } from "./FeedbackButtons";

export function RabbitHoleCard(props: {
  hole: Recommendation;
  rank: number;
  href: string;
  seedTitles: string[];
  surprise: number;
}) {
  const { hole, rank } = props;
  const { name, tags } = splitLabel(hole.label);
  return (
    <article
      className="rise grid grid-cols-[auto_1fr] gap-x-5 gap-y-3 rounded-lg border border-line bg-card p-5 shadow-[0_1px_0_var(--line)] sm:p-6"
      style={{ animationDelay: `${rank * 70}ms` }}
      data-testid="rabbit-hole"
    >
      <div className="row-span-2">
        <DepthMeter score={hole.rh_score} />
      </div>
      <header>
        <p className="font-mono text-xs text-ink-soft">
          level {rank + 1} · {hole.size} articles{tags && ` · ${tags}`}
        </p>
        <h2 className="font-display text-2xl leading-tight sm:text-3xl">
          <Link href={props.href} className="hover:text-rust">
            {name}
          </Link>
        </h2>
        {hole.hook && <p className="mt-1 text-ink-soft">{hole.hook}</p>}
      </header>
      <div className="col-span-2 space-y-3 sm:col-span-1">
        <div>
          <h3 className="font-mono text-[11px] uppercase tracking-widest text-ink-soft">Why this?</h3>
          <p className="mt-1 text-sm">{hole.why}</p>
          <div className="mt-2">
            <BridgePath path={hole.bridge_path} />
          </div>
        </div>
        <div>
          <h3 className="font-mono text-[11px] uppercase tracking-widest text-ink-soft">Inside</h3>
          <ul className="mt-1 flex flex-wrap gap-x-3 gap-y-1 text-sm">
            {hole.top_articles.map((a) => (
              <li key={a.id}>
                <a href={a.url} target="_blank" rel="noreferrer" className="underline decoration-line underline-offset-4 hover:decoration-rust">
                  {cleanTitle(a.title)}
                </a>
              </li>
            ))}
          </ul>
        </div>
        <div className="flex flex-wrap items-center justify-between gap-3 pt-1">
          <FeedbackButtons communityId={hole.id} seedTitles={props.seedTitles} surprise={props.surprise} />
          <Link href={props.href} className="font-mono text-sm text-rust hover:underline">
            start the trail ↓
          </Link>
        </div>
      </div>
    </article>
  );
}
