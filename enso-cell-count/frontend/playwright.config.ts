import { defineConfig, devices } from "@playwright/test";

export default defineConfig({
  testDir: "./tests/visual",
  testMatch: /.*\.spec\.ts/,
  reporter: "list",
  use: {
    viewport: { width: 1600, height: 950 },
    ignoreHTTPSErrors: true,
    headless: true,
    actionTimeout: 15000,
  },
  projects: [
    {
      name: "chromium",
      use: { ...devices["Desktop Chrome"] },
    },
  ],
});
