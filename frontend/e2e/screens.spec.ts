import { expect, test, type Page } from "@playwright/test";

// Needs the real API on :8000 with the DEV data loaded.
const OUT = "../reports/screens";
const Q = "i=Byzantine+Empire&i=chess&i=mechanical+keyboard&s=0.50";

async function shot(page: Page, name: string) {
  await page.waitForTimeout(700); // let the entrance animations settle
  await page.screenshot({ path: `${OUT}/${name}.png`, fullPage: true });
}

for (const scheme of ["light", "dark"] as const) {
  test(`full flow, ${scheme}`, async ({ page }) => {
    await page.emulateMedia({ colorScheme: scheme });
    await page.setViewportSize({ width: 1280, height: 900 });
    await page.goto("/");
    await shot(page, `01_home_${scheme}`);

    await page.goto(`/?${Q}`);
    await expect(page.getByTestId("rabbit-hole")).toHaveCount(5, { timeout: 30_000 });
    await shot(page, `02_results_${scheme}`);

    await page.getByRole("link", { name: "start the trail ↓" }).first().click();
    await expect(page.getByTestId("trail-step").first()).toBeVisible({ timeout: 30_000 });
    await shot(page, `03_trail_${scheme}`);

    await page.goto(`/map?${Q}`);
    await expect(page.locator("canvas")).toBeVisible({ timeout: 30_000 });
    await page.waitForTimeout(1500);
    await page.screenshot({ path: `${OUT}/04_map_${scheme}.png` });
  });
}

test("phone width", async ({ page }) => {
  await page.setViewportSize({ width: 390, height: 844 });
  await page.goto(`/?${Q}`);
  await expect(page.getByTestId("rabbit-hole")).toHaveCount(5, { timeout: 30_000 });
  await shot(page, "05_results_phone");
  await page.getByRole("link", { name: "start the trail ↓" }).first().click();
  await expect(page.getByTestId("trail-step").first()).toBeVisible({ timeout: 30_000 });
  await shot(page, "06_trail_phone");
});
