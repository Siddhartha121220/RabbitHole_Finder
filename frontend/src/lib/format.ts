// Plain-language names for the Depth Meter features (DATA_AND_ML §3).
export const FEATURE_LABELS: Record<string, { label: string; hint: string }> = {
  expected_depth: { label: "How deep readers go", hint: "pages read inside before leaving" },
  stickiness: { label: "Stays inside", hint: "share of clicks that stay in the hole" },
  onward_ratio: { label: "Keeps clicking", hint: "onward clicks per visit" },
  breadth: { label: "Room to roam", hint: "how many articles it holds" },
  content_richness: { label: "Long reads", hint: "typical article length" },
  discovery_share: { label: "Found by wandering", hint: "arrived by clicking, not searching" },
};

export const cleanTitle = (t: string) => t.replaceAll("_", " ");

/** "The Beatles (beatles, album, song)" -> name and keywords; the keywords are the LAST
 *  parenthesised group, so "Control (video game) (remedy, game)" keeps "(video game)". */
export function splitLabel(label: string): { name: string; tags: string } {
  const m = /^(.*)\s+\(([^()]*)\)\s*$/.exec(label);
  return m ? { name: m[1] ?? label, tags: m[2] ?? "" } : { name: label, tags: "" };
}
