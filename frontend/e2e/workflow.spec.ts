import { expect, test, type Page } from "@playwright/test";

async function prepare(page: Page, dataset: string) {
  await page.goto("/data");
  await page.getByLabel("Example dataset").selectOption(dataset);
  await page.getByLabel("Folds", { exact: true }).fill("2");
  await page.getByRole("button", { name: "Inspect data", exact: true }).click();
  await expect(page.getByText("No feature imputation needed")).toBeVisible();
  await page.getByRole("button", { name: "Validate and continue" }).click();
  await expect(page).toHaveURL(/\/model$/);
  await page.getByRole("button", { name: /^Random Forest/ }).click();
  await page.getByRole("button", { name: "Show advanced specification" }).click();
  await page.locator("textarea.codeEditor").fill(JSON.stringify({
    fixed_params: { n_jobs: 1, random_state: 42 },
    search_space: { n_estimators: { type: "choice", values: [5] }, max_depth: { type: "choice", values: [3] } },
  }));
  await page.locator("textarea.codeEditor").blur();
  await page.getByRole("button", { name: "Validate and continue" }).click();
  await expect(page).toHaveURL(/\/search$/);
  await page.getByLabel("Number of trials").fill("1");
  await expect(page.getByText("2 candidate model fits + 1 winner refit planned per model")).toBeVisible();
}

for (const [dataset, task] of [
  ["diabetes", "regression"],
  ["breast_cancer", "binary classification"],
  ["wine", "multiclass classification"],
] as const) {
  test(`${task}: six stages, refresh, saved results and full downloads`, async ({ page }, testInfo) => {
    const errors: string[] = [];
    page.on("pageerror", (error) => errors.push(error.message));
    await prepare(page, dataset);
    if (dataset === "diabetes") await page.getByLabel("Optimization metric").selectOption("mae");
    if (dataset === "breast_cancer") await page.getByLabel("Positive class", { exact: true }).selectOption("0");
    await page.getByRole("button", { name: "Start experiment run" }).click();
    await expect(page).toHaveURL(/\/live\/[\w-]+$/);
    const runUrl = page.url();
    await page.reload();
    await expect(page).toHaveURL(runUrl);
    await expect(page.locator(".metric").filter({ hasText: "Candidate model fits" })).toContainText("2 / 2");
    await expect(page.locator(".metric").filter({ hasText: "Winner refit" })).toContainText("1 / 1");
    await page.getByRole("button", { name: "View results" }).click();
    await expect(page.getByRole("heading", { name: "Metric summary", exact: true })).toBeVisible();
    const predictionTool = page.getByRole("button", { name: /^Prediction table/ });
    if (!(await predictionTool.getAttribute("class"))?.includes("selected")) await predictionTool.click();
    await page.getByRole("button", { name: "Load rows" }).click();
    await expect(page.getByRole("columnheader", { name: "Actual", exact: true })).toBeVisible();
    await page.getByText("Export", { exact: true }).click();
    for (const kind of ["predictions", "metrics", "spec", "events", "manifest", "model", "environment", "split_indices"]) {
      const [download] = await Promise.all([
        page.waitForEvent("download"), page.getByRole("link", { name: kind, exact: true }).click(),
      ]);
      expect(await download.failure()).toBeNull();
      await download.saveAs(testInfo.outputPath(download.suggestedFilename()));
    }
    await page.screenshot({ path: testInfo.outputPath(`${dataset}-results.png`), fullPage: true });
    await page.getByRole("link", { name: /History/ }).click();
    const row = page.getByRole("row").filter({ hasText: runUrl.split("/").pop()!.slice(0, 12) });
    await expect(row).toContainText("completed");
    await row.getByRole("button", { name: "Open results" }).click();
    await expect(page).toHaveURL(/\/results\//);
    expect(errors).toEqual([]);
  });
}

test("cancel, persisted terminal status and explicit retry", async ({ page }) => {
  await prepare(page, "diabetes");
  await page.getByRole("button", { name: "Start experiment run" }).click();
  await expect(page).toHaveURL(/\/live\/[\w-]+$/);
  const original = page.url();
  await page.getByRole("button", { name: "Cancel", exact: true }).click();
  await expect(page.locator(".statusBadge.cancelled")).toBeVisible();
  await page.reload();
  await page.getByRole("button", { name: "Retry", exact: true }).click();
  await expect(page).not.toHaveURL(original);
  await expect(page.getByText("Retry of", { exact: true })).toBeVisible();
  await expect(page.getByRole("button", { name: "View results" })).toBeVisible();
});
