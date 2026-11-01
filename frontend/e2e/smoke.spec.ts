import { expect, test, type Page } from "@playwright/test";

// Smoke tests with the API mocked, so CI needs no backend or data.
const API = "http://localhost:8000";
const art = (id: number, title: string) => ({ id, title, url: `https://en.wikipedia.org/wiki/${title}` });

const HOLE = {
  id: 7,
  label: "Titanic (titanic, ship, rms)",
  keywords: ["titanic", "ship"],
  size: 220,
  rh_score: 68,
  rh_breakdown: { stickiness: { value: 0.6, z: 1.2 }, expected_depth: { value: 1.3, z: 0.8 } },
  top_articles: [art(1, "Titanic"), art(2, "RMS_Olympic")],
};

async function mockApi(page: Page, opts: { down?: boolean } = {}) {
  const feedback: unknown[] = [];
  await page.route(`${API}/**`, async (route) => {
    if (opts.down) return route.abort();
    const path = new URL(route.request().url()).pathname;
    const json = (body: unknown) => route.fulfill({ json: body });
    if (path === "/seeds/search") return json([{ ...art(3, "Rook_(chess)"), community_id: 1, similarity: 0.81 }]);
    if (path === "/recommend")
      return json({
        seeds: [{ ...art(3, "Rook_(chess)"), community_id: 1, similarity: 0.81, x: 0.2, y: 0.3 }],
        rabbit_holes: [
          {
            ...HOLE,
            hook: "RMS Titanic was a British ocean liner.",
            x: 0.5,
            y: 0.5,
            bridge_path: [art(3, "Rook_(chess)"), art(4, "Lewis_chessmen"), art(5, "Stornoway")],
            why: "Readers of Rook (chess) reach Stornoway in 2 clicks via Lewis chessmen.",
          },
        ],
      });
    if (path === "/rabbit-holes/7")
      return json({ ...HOLE, central_article: art(1, "Titanic"), about: "RMS Titanic was a liner.", umap_x: 0.5, umap_y: 0.5 });
    if (path === "/rabbit-holes/7/trail")
      return json({
        rabbit_hole_id: 7,
        model: "template",
        intro: "A gentle way in.",
        steps: [1, 2].map((i) => ({ ...art(i, i === 1 ? "Titanic" : "RMS_Olympic"), why: "Worth reading." })),
      });
    if (path === "/map") return json([{ id: 7, label: HOLE.label, size: 220, rh_score: 68, x: 0.5, y: 0.5 }]);
    if (path === "/feedback") {
      feedback.push(route.request().postDataJSON());
      return route.fulfill({ status: 201, json: { status: "stored" } });
    }
    return route.fulfill({ status: 404, json: { detail: "not mocked" } });
  });
  return feedback;
}

test("find: autocomplete, recommend, feedback", async ({ page }) => {
  const feedback = await mockApi(page);
  await page.goto("/");
  await page.getByRole("combobox").fill("rook");
  await page.getByRole("option", { name: /Rook \(chess\)/ }).click();
  await page.getByRole("button", { name: /Dig/ }).click();
  const card = page.getByTestId("rabbit-hole");
  await expect(card).toHaveCount(1);
  await expect(card.getByRole("heading", { name: "Titanic" })).toBeVisible();
  await expect(card.getByRole("link", { name: "Stornoway" })).toHaveAttribute("href", "https://en.wikipedia.org/wiki/Stornoway");
  await card.getByRole("button", { name: /going down it/ }).click();
  await expect(card.getByText("noted, thanks")).toBeVisible();
  expect(feedback).toHaveLength(1);
  expect(feedback[0]).toMatchObject({ community_id: 7, rating: "up" });
});

test("hole page shows the trail with real links", async ({ page }) => {
  await mockApi(page);
  await page.goto("/hole/7?i=chess&s=0.50");
  await expect(page.getByRole("heading", { name: "Titanic", level: 1 })).toBeVisible();
  const steps = page.getByTestId("trail-step");
  await expect(steps).toHaveCount(2);
  await expect(steps.first().getByRole("link")).toHaveAttribute("href", /^https:\/\/en\.wikipedia\.org\/wiki\//);
});

test("map renders", async ({ page }) => {
  await mockApi(page);
  await page.goto("/map?i=chess");
  await expect(page.getByTestId("map").locator("canvas")).toBeVisible();
});

test("API down shows a friendly error", async ({ page }) => {
  await mockApi(page, { down: true });
  await page.goto("/?i=chess");
  await expect(page.getByRole("alert").filter({ hasText: "Something caved in" })).toContainText("Can't reach");
});
