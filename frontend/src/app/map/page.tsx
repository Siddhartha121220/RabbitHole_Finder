"use client";

import dynamic from "next/dynamic";
import { useRouter, useSearchParams } from "next/navigation";
import { Suspense, useState } from "react";
import { ErrorBox, Loading } from "@/components/Status";
import { api } from "@/lib/api";
import { cleanTitle } from "@/lib/format";
import { readSearch, searchQuery } from "@/lib/query";
import { useAsync } from "@/lib/useAsync";

// deck.gl needs WebGL: browser only.
const MapView = dynamic(() => import("@/components/MapView"), {
  ssr: false,
  loading: () => <Loading label="unrolling the map…" />,
});

function MapPage() {
  const router = useRouter();
  const params = useSearchParams();
  const search = readSearch(params);
  const [retry, setRetry] = useState(0);
  const points = useAsync(`map#${retry}`, (signal) => api.map(signal));
  const recs = useAsync(search.interests.length ? `${searchQuery(search)}#${retry}` : null, (signal) =>
    api.recommend({ interests: search.interests, surprise: search.surprise, k: 5 }, signal),
  );

  const seeds =
    recs.status === "done"
      ? recs.data.seeds.flatMap((s) => (s.x == null || s.y == null ? [] : [{ x: s.x, y: s.y, title: cleanTitle(s.title) }]))
      : [];
  const picked = recs.status === "done" ? recs.data.rabbit_holes.map((h) => h.id) : [];
  const q = search.interests.length ? `?${searchQuery(search)}` : "";

  return (
    <div className="space-y-4 pt-4">
      <h1 className="font-display text-3xl sm:text-4xl">The warren</h1>
      <p className="max-w-2xl text-ink-soft">
        Every rabbit hole, placed by what its articles are about. Brighter = deeper.
        {search.interests.length > 0 && " Your interests glow green; your recommendations wear a rust ring."}
      </p>
      {points.status === "error" && <ErrorBox error={points.error} onRetry={() => setRetry((r) => r + 1)} />}
      {recs.status === "error" && <ErrorBox error={recs.error} onRetry={() => setRetry((r) => r + 1)} />}
      <div className="relative h-[70vh] min-h-[22rem] overflow-hidden rounded-lg border border-line bg-[#120d09]" data-testid="map">
        {points.status === "done" ? (
          <MapView points={points.data} seeds={seeds} picked={picked} onSelect={(id) => router.push(`/hole/${id}${q}`)} />
        ) : (
          points.status === "loading" && <Loading label="unrolling the map…" />
        )}
      </div>
      <ul className="flex flex-wrap gap-x-6 gap-y-2 font-mono text-xs text-ink-soft" aria-label="Legend">
        <li><span className="mr-2 inline-block h-3 w-3 rounded-full bg-[#5c4632] align-middle" />shallow</li>
        <li><span className="mr-2 inline-block h-3 w-3 rounded-full bg-[#f2b84b] align-middle" />deep</li>
        <li><span className="mr-2 inline-block h-3 w-3 rounded-full bg-[#63b79e] align-middle" />your interests</li>
        <li><span className="mr-2 inline-block h-3 w-3 rounded-full border-2 border-[#e58049] align-middle" />recommended</li>
        <li>circle size = number of articles · drag to pan, scroll to zoom, click to open</li>
      </ul>
    </div>
  );
}

export default function Page() {
  return (
    <Suspense fallback={<Loading label="loading…" />}>
      <MapPage />
    </Suspense>
  );
}
