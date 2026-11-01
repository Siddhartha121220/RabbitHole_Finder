// Typed client for the FastAPI service. Types come from openapi.json (`make api-types`).
import type { components } from "./api-types";

export type Schemas = components["schemas"];
export type Recommendation = Schemas["Recommendation"];
export type RecommendOut = Schemas["RecommendOut"];
export type RabbitHoleDetail = Schemas["RabbitHoleDetail"];
export type TrailOut = Schemas["TrailOut"];
export type MapPoint = Schemas["MapPoint"];
export type ArticleHit = Schemas["ArticleHitOut"];
export type ArticleRef = Schemas["ArticleRef"];
export type FeatureValue = Schemas["FeatureValue"];
export type Rating = Schemas["FeedbackIn"]["rating"];

export const API_URL = process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:8000";

export class ApiError extends Error {
  constructor(
    public status: number,
    message: string,
  ) {
    super(message);
  }
}

async function call<T>(path: string, body?: unknown, signal?: AbortSignal): Promise<T> {
  let res: Response;
  try {
    res = await fetch(`${API_URL}${path}`, {
      method: body === undefined ? "GET" : "POST",
      headers: body === undefined ? undefined : { "Content-Type": "application/json" },
      body: body === undefined ? undefined : JSON.stringify(body),
      signal,
    });
  } catch (e) {
    if (e instanceof DOMException && e.name === "AbortError") throw e;
    throw new ApiError(0, "Can't reach the Rabbit Hole Finder API. Is it running?");
  }
  if (!res.ok) {
    let detail = res.statusText;
    try {
      const data = (await res.json()) as { detail?: unknown };
      if (typeof data.detail === "string") detail = data.detail;
      else if (res.status === 422) detail = "Some of that input isn't valid.";
    } catch {
      // keep statusText
    }
    if (res.status === 429) detail = "Too many trail requests. Give it a minute.";
    throw new ApiError(res.status, detail);
  }
  return (await res.json()) as T;
}

export const api = {
  seeds: (query: string, signal?: AbortSignal) =>
    call<ArticleHit[]>("/seeds/search", { query, k: 6 }, signal),
  recommend: (body: Schemas["RecommendIn"], signal?: AbortSignal) =>
    call<RecommendOut>("/recommend", body, signal),
  hole: (id: number, signal?: AbortSignal) =>
    call<RabbitHoleDetail>(`/rabbit-holes/${id}`, undefined, signal),
  trail: (id: number, body: Schemas["TrailIn"], signal?: AbortSignal) =>
    call<TrailOut>(`/rabbit-holes/${id}/trail`, body, signal),
  map: (signal?: AbortSignal) => call<MapPoint[]>("/map", undefined, signal),
  feedback: (body: Schemas["FeedbackIn"]) => call<{ status: string }>("/feedback", body),
};
