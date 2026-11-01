"use client";

import { useRouter, useSearchParams } from "next/navigation";
import { Suspense, useState } from "react";
import { InterestInput } from "@/components/InterestInput";
import { RabbitHoleCard } from "@/components/RabbitHoleCard";
import { ErrorBox, Loading } from "@/components/Status";
import { SurpriseSlider } from "@/components/SurpriseSlider";
import { api } from "@/lib/api";
import { cleanTitle } from "@/lib/format";
import { readSearch, searchQuery } from "@/lib/query";
import { useAsync } from "@/lib/useAsync";

function Finder() {
  const router = useRouter();
  const params = useSearchParams();
  const search = readSearch(params);
  const [interests, setInterests] = useState(search.interests);
  const [surprise, setSurprise] = useState(search.surprise);
  const [retry, setRetry] = useState(0);

  const key = search.interests.length ? `${searchQuery(search)}#${retry}` : null;
  const result = useAsync(key, (signal) =>
    api.recommend({ interests: search.interests, surprise: search.surprise, k: 5 }, signal),
  );

  function submit(e: React.FormEvent) {
    e.preventDefault();
    if (interests.length) router.push(`/?${searchQuery({ interests, surprise })}`);
  }

  return (
    <>
      <section className="pb-10 pt-6 sm:pt-12">
        <h1 className="max-w-3xl font-display text-4xl leading-[1.05] sm:text-6xl">
          Find the <em className="text-rust">next obsession</em>, one burrow over.
        </h1>
        <p className="mt-4 max-w-2xl text-lg text-ink-soft">
          Tell it a few things you already love. It looks at where millions of Wikipedia readers actually
          wander, and finds the rabbit holes close enough to feel familiar and deep enough to lose a weekend in.
        </p>
        <form onSubmit={submit} className="mt-8 grid gap-6 rounded-lg border border-line bg-card/70 p-5 backdrop-blur sm:grid-cols-[1fr_16rem] sm:p-6">
          <InterestInput value={interests} onChange={setInterests} />
          <div className="flex flex-col justify-between gap-4">
            <SurpriseSlider value={surprise} onChange={setSurprise} />
            <button
              type="submit"
              disabled={!interests.length}
              className="rounded-md bg-rust px-4 py-2.5 font-medium text-card transition hover:brightness-110 disabled:opacity-40"
            >
              Dig ↓
            </button>
          </div>
        </form>
      </section>

      {result.status === "loading" && <Loading label="digging through the click graph…" />}
      {result.status === "error" && <ErrorBox error={result.error} onRetry={() => setRetry((r) => r + 1)} />}
      {result.status === "done" && (
        <section aria-label="Recommended rabbit holes" className="space-y-5">
          <p className="font-mono text-xs text-ink-soft">
            starting from{" "}
            {result.data.seeds.map((s) => cleanTitle(s.title)).join(" · ")} ·{" "}
            <a href={`/map?${searchQuery(search)}`} className="text-rust underline-offset-4 hover:underline">
              see them on the map
            </a>
          </p>
          {result.data.rabbit_holes.map((hole, rank) => (
            <RabbitHoleCard
              key={hole.id}
              hole={hole}
              rank={rank}
              surprise={search.surprise}
              seedTitles={result.data.seeds.map((s) => s.title)}
              href={`/hole/${hole.id}?${searchQuery(search, {
                b: hole.bridge_path.map((a) => String(a.id)),
                seed: result.data.seeds.map((s) => s.title),
              })}`}
            />
          ))}
        </section>
      )}
    </>
  );
}

export default function Home() {
  return (
    <Suspense fallback={<Loading label="loading…" />}>
      <Finder />
    </Suspense>
  );
}
