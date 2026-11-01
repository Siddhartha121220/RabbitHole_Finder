"use client";

import { OrthographicView } from "@deck.gl/core";
import { ScatterplotLayer, TextLayer } from "@deck.gl/layers";
import DeckGL from "@deck.gl/react";
import type { MapPoint } from "@/lib/api";
import { splitLabel } from "@/lib/format";

export type Seed = { x: number; y: number; title: string };
type RGB = [number, number, number];

// The map is always "underground" (dark canvas), so one palette serves both themes.
const LOW: RGB = [92, 70, 50]; // shallow holes: packed earth
const HIGH: RGB = [242, 184, 75]; // deep holes: lantern light
const PICK: RGB = [229, 128, 73]; // recommended: rust ring
const SEED: RGB = [99, 183, 158]; // your interests: moss glow
const SCALE = 1000; // UMAP coords are 0..1

const depthColor = (score: number | null): RGB => {
  const t = (score ?? 0) / 100;
  return [0, 1, 2].map((i) => Math.round(LOW[i]! + (HIGH[i]! - LOW[i]!) * t)) as RGB;
};
const radius = (size: number) => 3 + Math.sqrt(size) * 0.9;

export default function MapView(props: {
  points: MapPoint[];
  seeds: Seed[];
  picked: number[];
  onSelect: (id: number) => void;
}) {
  const picked = new Set(props.picked);
  const labelled = props.points.filter((p) => picked.has(p.id) || p.size >= 150);
  const layers = [
    new ScatterplotLayer<MapPoint>({
      id: "holes",
      data: props.points,
      getPosition: (p) => [p.x * SCALE, p.y * SCALE],
      getRadius: (p) => radius(p.size),
      radiusUnits: "pixels",
      getFillColor: (p) => [...depthColor(p.rh_score), picked.size && !picked.has(p.id) ? 140 : 235],
      getLineColor: (p) => (picked.has(p.id) ? [...PICK, 255] : [0, 0, 0, 0]),
      getLineWidth: 3,
      lineWidthUnits: "pixels",
      stroked: true,
      pickable: true,
      autoHighlight: true,
      highlightColor: [255, 255, 255, 60],
      onClick: ({ object }) => object && props.onSelect(object.id),
      updateTriggers: { getFillColor: props.picked, getLineColor: props.picked },
    }),
    new ScatterplotLayer<Seed>({
      id: "seed-glow",
      data: props.seeds,
      getPosition: (s) => [s.x * SCALE, s.y * SCALE],
      getRadius: 18,
      radiusUnits: "pixels",
      getFillColor: [...SEED, 55],
    }),
    new ScatterplotLayer<Seed>({
      id: "seeds",
      data: props.seeds,
      getPosition: (s) => [s.x * SCALE, s.y * SCALE],
      getRadius: 5,
      radiusUnits: "pixels",
      getFillColor: [...SEED, 255],
      pickable: true,
    }),
    new TextLayer<MapPoint>({
      id: "labels",
      data: labelled,
      getPosition: (p) => [p.x * SCALE, p.y * SCALE],
      getText: (p) => splitLabel(p.label).name,
      getSize: 12,
      getColor: (p) => (picked.has(p.id) ? [...PICK, 255] : [241, 230, 211, 200]),
      getPixelOffset: (p) => [0, -radius(p.size) - 9],
      fontFamily: "ui-monospace, monospace",
      characterSet: "auto",
      updateTriggers: { getColor: props.picked },
    }),
  ];
  return (
    <DeckGL
      views={new OrthographicView({ id: "map" })}
      initialViewState={{ target: [SCALE / 2, SCALE / 2, 0], zoom: -0.75, minZoom: -2, maxZoom: 4 }}
      controller
      layers={layers}
      getCursor={({ isHovering }) => (isHovering ? "pointer" : "grab")}
      getTooltip={({ object, layer }) => {
        if (!object) return null;
        if (layer?.id === "seeds") return { text: `your interest: ${(object as Seed).title}` };
        const p = object as MapPoint;
        const depth = p.rh_score === null ? "–" : Math.round(p.rh_score);
        return { text: `${splitLabel(p.label).name}\n${p.size} articles · depth ${depth}` };
      }}
    />
  );
}
