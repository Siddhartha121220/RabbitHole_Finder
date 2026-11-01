// The search lives in the URL so back/forward and shared links work.
export type Search = { interests: string[]; surprise: number };

export function readSearch(params: URLSearchParams): Search {
  const interests = params.getAll("i").map((s) => s.trim()).filter(Boolean).slice(0, 5);
  const s = Number(params.get("s") ?? "0.5");
  return { interests, surprise: Number.isFinite(s) ? Math.min(1, Math.max(0, s)) : 0.5 };
}

export function searchQuery({ interests, surprise }: Search, extra: Record<string, string[]> = {}) {
  const q = new URLSearchParams();
  interests.forEach((i) => q.append("i", i));
  q.set("s", surprise.toFixed(2));
  for (const [k, vs] of Object.entries(extra)) vs.forEach((v) => q.append(k, v));
  return q.toString();
}
