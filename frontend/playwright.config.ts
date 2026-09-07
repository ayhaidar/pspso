import { defineConfig, devices } from "@playwright/test";

export default defineConfig({
  testDir: "./e2e",
  timeout: 90000,
  expect: { timeout: 30000 },
  workers: 1,
  retries: 0,
  reporter: [["list"], ["html", { open: "never" }]],
  use: {
    baseURL: "http://127.0.0.1:8371",
    ...devices["Desktop Chrome"],
    trace: "retain-on-failure",
    screenshot: "only-on-failure",
  },
  webServer: {
    command: "uv run --no-sync python scripts/browser_server.py",
    cwd: "..",
    url: "http://127.0.0.1:8371/api/v1/estimators",
    timeout: 60000,
    reuseExistingServer: false,
  },
});
