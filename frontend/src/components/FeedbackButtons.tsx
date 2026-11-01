"use client";

import { useState } from "react";
import { api, type Rating } from "@/lib/api";
import { sessionId } from "@/lib/session";

const OPTIONS: { rating: Rating; emoji: string; label: string }[] = [
  { rating: "up", emoji: "👍", label: "going down it" },
  { rating: "down", emoji: "👎", label: "meh" },
  { rating: "obsessed", emoji: "🤯", label: "already obsessed" },
];

export function FeedbackButtons(props: { communityId: number; seedTitles: string[]; surprise: number }) {
  const [sent, setSent] = useState<Rating | null>(null);
  const [error, setError] = useState(false);
  async function send(rating: Rating) {
    setError(false);
    try {
      await api.feedback({
        session_id: sessionId(),
        community_id: props.communityId,
        seed_titles: props.seedTitles.slice(0, 10),
        surprise: props.surprise,
        rating,
      });
      setSent(rating);
    } catch {
      setError(true);
    }
  }
  return (
    <div className="flex flex-wrap items-center gap-2" role="group" aria-label="Your verdict">
      {OPTIONS.map((o) => (
        <button
          key={o.rating}
          onClick={() => send(o.rating)}
          disabled={sent !== null}
          aria-pressed={sent === o.rating}
          className={`rounded-full border px-3 py-1 text-xs transition ${sent === o.rating ? "border-rust bg-rust text-card" : "border-line hover:border-rust disabled:opacity-40"}`}
        >
          <span aria-hidden>{o.emoji}</span> {o.label}
        </button>
      ))}
      {sent && <span className="font-mono text-xs text-moss">noted, thanks</span>}
      {error && <span className="font-mono text-xs text-danger">couldn&apos;t save, try again</span>}
    </div>
  );
}
