import { FEATURE_LABELS } from "@/lib/format";
import type { FeatureValue } from "@/lib/api";

/** A vertical burrow shaft; the lantern light reaches as deep as the score. */
export function DepthMeter({ score, size = "md" }: { score: number | null; size?: "md" | "lg" }) {
  const value = score === null ? null : Math.round(score);
  const h = size === "lg" ? "h-40" : "h-24";
  return (
    <div className="flex flex-col items-center gap-1" aria-label={`Depth ${value ?? "unknown"} out of 100`}>
      <div className={`relative ${h} w-5 overflow-hidden rounded-b-full rounded-t-sm bg-shaft ring-1 ring-line`}>
        <div
          className="absolute inset-x-0 top-0 bg-gradient-to-b from-lantern to-rust"
          style={{ height: `${value ?? 0}%` }}
        />
      </div>
      <span className="font-mono text-xs text-ink-soft">depth</span>
      <span className="font-display text-xl leading-none">{value ?? "–"}</span>
    </div>
  );
}

/** Feature-by-feature reasons, as z-scores: bars to the right = deeper than average. */
export function DepthBreakdown({ breakdown }: { breakdown: Record<string, FeatureValue> }) {
  const rows = Object.entries(FEATURE_LABELS).filter(([k]) => k in breakdown);
  if (rows.length === 0) return null;
  return (
    <dl className="grid gap-2 text-sm">
      {rows.map(([key, { label, hint }]) => {
        const z = breakdown[key]?.z ?? 0;
        const width = Math.min(Math.abs(z), 3) / 3;
        return (
          <div key={key} className="grid grid-cols-[minmax(0,10rem)_1fr] items-center gap-3" title={hint}>
            <dt className="truncate text-ink-soft">{label}</dt>
            <dd className="relative h-2 rounded-full bg-line/50" aria-label={`${label}: ${z >= 0 ? "+" : ""}${z.toFixed(1)} standard deviations`}>
              <span className="absolute inset-y-0 left-1/2 w-px bg-ink-soft/50" />
              <span
                className={`absolute inset-y-0 rounded-full ${z >= 0 ? "left-1/2 bg-lantern" : "right-1/2 bg-ink-soft/60"}`}
                style={{ width: `${width * 50}%` }}
              />
            </dd>
          </div>
        );
      })}
    </dl>
  );
}
