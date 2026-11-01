import { defineConfig, devices } from "@playwright/test";

// smoke: API mocked with page.route, runs in CI. screens: real local API (make api), writes
// reports/screens/*.png; run with `pnpm screens`.
const PORT = 3000; // the origin the API allows (FRONTEND_ORIGIN)

export default defineConfig({
  testDir: "e2e",
  timeout: 60_000,
  use: { baseURL: `http://localhost:${PORT}`, ...devices["Desktop Chrome"] },
  projects: [
    { name: "smoke", testMatch: /smoke\.spec\.ts/ },
    { name: "screens", testMatch: /screens\.spec\.ts/ },
  ],
  webServer: {
    command: `pnpm start --port ${PORT}`,
    url: `http://localhost:${PORT}`,
    reuseExistingServer: !process.env.CI,
    timeout: 60_000,
  },
});
