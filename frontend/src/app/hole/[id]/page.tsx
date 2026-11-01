"use client";

import Link from "next/link";
import { useParams, useSearchParams } from "next/navigation";
import { Suspense, useState } from "react";
import { DepthBreakdown, DepthMeter } from "@/components/DepthMeter";
import { FeedbackButtons } from "@/components/FeedbackButtons";
import { ErrorBox, Loading } from "@/components/Status";
import { TrailView } from "@/components/TrailView";
import { api } from "@/lib/api";
import { cleanTitle, splitLabel } from "@/lib/format";
import { readSearch, searchQuery } from "@/lib/query";
import { useAsync } from "@/lib/useAsync";

function Hole() {
  const id = Number(useParams<{ id: string }>().id);
  const params = useSearchParams();
  const search = readSearch(params);
  const seeds = params.getAll("seed");
  const bridge = params.getAll("b").map(Number).filter(Number.isInteger).slice(0, 10);
  const [retry, setRetry] = useState(0);

  const detail = useAsync(Number.isInteger(id) ? `${id}#${retry}` : null, (signal) => api.hole(id, signal));
  const trail = useAsync(Number.isInteger(id) ? `${id}?${params.toString()}#${retry}` : null, (signal) =>
    api.trail(id, { interests: search.interests, seed_titles: seeds.slice(0, 10), bridge_article_ids: bridge }, signal),
  );

  if (detail.status === "error") return <ErrorBox error={detail.error} onRetry={() => setRetry((r) => r + 1)} />;
  if (detail.status !== "done") return <Loading label="lowering the lantern…" />;
  const hole = detail.data;
  const { name, tags } = splitLabel(hole.label);

  return (
    <div className="space-y-10 pt-4">
      {search.interests.length > 0 && (
        <p>
          <Link href={`/?${searchQuery(search)}`} className="font-mono text-sm text-ink-soft hover:text-rust">
            ← back to your rabbit holes
          </Link>
        </p>
      )}
      <section className="grid gap-6 sm:grid-cols-[auto_1fr]">
        <DepthMeter score={hole.rh_score} size="lg" />
        <div>
          <p className="font-mono text-xs text-ink-soft">
            {hole.size} articles{tags && ` · ${tags}`}
          </p>
          <h1 className="font-display text-4xl leading-tight sm:text-5xl">{name}</h1>
          <p className="mt-3 max-w-2xl text-lg text-ink-soft">{hole.about}</p>
          <p className="mt-2 text-sm">
            Heart of it:{" "}
            <a href={hole.central_article.url} target="_blank" rel="noreferrer" className="text-rust underline underline-offset-4">
              {cleanTitle(hole.central_article.title)}
            </a>
          </p>
          <div className="mt-5">
            <FeedbackButtons communityId={hole.id} seedTitles={seeds} surprise={search.surprise} />
          </div>
        </div>
      </section>

      <section className="grid gap-8 lg:grid-cols-[1fr_20rem]">
        <div>
          {trail.status === "loading" && <Loading label="choosing the way down…" />}
          {trail.status === "error" && <ErrorBox error={trail.error} onRetry={() => setRetry((r) => r + 1)} />}
          {trail.status === "done" && <TrailView trail={trail.data} />}
        </div>
        <aside className="space-y-4 rounded-lg border border-line bg-card p-5">
          <h2 className="font-display text-lg">Why it&apos;s deep</h2>
          <p className="text-xs text-ink-soft">
            Each bar compares this rabbit hole with the others, from real reader behaviour. Right of the line = more than average.
          </p>
          <DepthBreakdown breakdown={hole.rh_breakdown} />
        </aside>
      </section>
    </div>
  );
}

export default function HolePage() {
  return (
    <Suspense fallback={<Loading label="loading…" />}>
      <Hole />
    </Suspense>
  );
}
