import { expect, test } from "@playwright/test";

test("welcome page introduces the workflow and opens Data setup", async ({ page }) => {
  await page.goto("/");
  const navigation = page.getByRole("navigation", { name: "Experiment workflow" });
  await expect(navigation.getByRole("link").first()).toHaveText("Overview");
  await expect(navigation.getByRole("link", { name: "Overview", exact: true })).toHaveAttribute("aria-current", "page");
  await expect(navigation.getByRole("link").nth(1)).toHaveText("1Data setup");
  await expect(page.getByRole("heading", { name: /Better model settings/ })).toBeVisible();
  await expect(page.getByText("Choose a fair evaluation", { exact: true })).toBeVisible();
  await page.getByRole("link", { name: "Set up an experiment" }).click();
  await expect(page).toHaveURL(/\/data$/);
  await expect(page.getByLabel("Evaluation method")).toHaveValue("cross_validation");
  await expect(page.getByLabel("Folds", { exact: true })).toHaveValue("5");
  await page.getByLabel("Evaluation method").selectOption("holdout");
  await expect(page.getByLabel("Validation share", { exact: true })).toHaveValue("0.2");
  await expect(page.getByLabel("Folds", { exact: true })).toHaveCount(0);
  await page.reload();
  await expect(page.getByLabel("Evaluation method")).toHaveValue("holdout");
  await navigation.getByRole("link", { name: "Overview", exact: true }).click();
  await expect(page).toHaveURL(/\/$/);
});

test("example catalogue applies each dataset's standard task, target and metric", async ({ page }) => {
  await page.goto("/data");
  const dataset = page.getByLabel("Example dataset");

  await dataset.selectOption("banknote_authentication");
  await expect(page.getByLabel("Task", { exact: true })).toHaveValue("binary_classification");
  await expect(page.getByLabel("Target column")).toHaveValue("class");
  await expect(page.locator(".datasetPreset")).toContainText("UCI Machine Learning Repository");
  await expect(page.locator(".datasetPreset")).toContainText("ROC AUC");

  await dataset.selectOption("auto_mpg");
  await expect(page.getByLabel("Task", { exact: true })).toHaveValue("regression");
  await expect(page.getByLabel("Target column")).toHaveValue("mpg");
  await expect(page.locator(".datasetPreset")).toContainText("CMU StatLib via UCI");
  await expect(page.locator(".datasetPreset")).toContainText("RMSE");
  await page.getByRole("button", { name: "Inspect data", exact: true }).click();
  const rowMetric = page.locator(".metricStrip > div").filter({ has: page.getByText("Rows", { exact: true }) });
  await expect(rowMetric).toContainText("398");
  await expect(page.getByLabel("Feature type for origin")).toBeVisible();
  await page.goto("/search");
  await expect(page.getByLabel("Optimization metric")).toHaveValue("rmse");

  await page.goto("/data");
  await dataset.selectOption("palmer_penguins");
  await expect(page.getByLabel("Task", { exact: true })).toHaveValue("multiclass_classification");
  await expect(page.getByLabel("Target column")).toHaveValue("species");
  await expect(page.locator(".datasetPreset")).toContainText("Palmer Station LTER");
  await expect(page.locator(".datasetPreset")).toContainText("Accuracy");
  await page.goto("/search");
  await expect(page.getByLabel("Optimization metric")).toHaveValue("accuracy");
});

test("chronological holdout with nominal, ordinal, missing and outlier preparation produces saved predictions", async ({ page }, testInfo) => {
  page.setDefaultTimeout(15000);
  const errors: string[] = [];
  page.on("pageerror", (error) => errors.push(error.message));
  const rows = Array.from({ length: 50 }, (_, index) => [
    new Date(Date.UTC(2024, 0, index + 1)).toISOString().slice(0, 10),
    index % 7 === 0 ? "" : index === 45 ? "9999" : String(index),
    String(index % 2 ? 10 : 20), ["low", "medium", "high"][index % 3], String(index * 2 + 5),
  ].join(",")).reverse();
  await page.goto("/data");
  await page.getByRole("button", { name: "Import CSV", exact: true }).click();
  await page.getByLabel("CSV contents").fill("date,value,region_code,grade,target\n" + rows.join("\n"));
  await page.getByLabel("Task", { exact: true }).selectOption("regression");
  await page.getByLabel("Evaluation method").selectOption("holdout");
  await page.getByLabel("Row order", { exact: true }).selectOption("chronological");
  await page.getByRole("button", { name: "Inspect data", exact: true }).click();
  await expect(page.getByLabel("Feature type for grade")).toBeVisible();
  await page.getByLabel("Time column", { exact: true }).selectOption("date");
  await expect(page.getByLabel("Feature type for grade")).toBeVisible();
  await page.getByLabel("Gap between partitions (rows)").fill("1");
  await page.getByLabel("Numeric outliers", { exact: true }).selectOption("quantile");
  await page.getByLabel("Feature type for region_code").selectOption("nominal");
  await page.getByLabel("Feature type for grade").selectOption("ordinal");
  await page.getByLabel("Ordered categories for grade").fill("low\nmedium\nhigh");
  await expect(page.getByLabel("Use date", { exact: true })).not.toBeChecked();
  await expect(page.getByLabel("Use date", { exact: true })).toBeDisabled();
  await page.getByRole("button", { name: "Update split preview" }).click();
  await expect(page.getByText("Chronological · date · gap 1 rows")).toBeVisible();
  await page.screenshot({ path: testInfo.outputPath("feature-preparation.png"), fullPage: true });
  await page.getByRole("button", { name: "Validate and continue" }).click();
  await expect(page).toHaveURL(/\/model$/);
  await page.getByRole("button", { name: /^Random Forest/ }).click();
  await page.getByRole("button", { name: "Show advanced specification" }).click();
  await page.locator("textarea.codeEditor").fill(JSON.stringify({ fixed_params: { n_jobs: 1, random_state: 42 }, search_space: { n_estimators: { type: "choice", values: [5] }, max_depth: { type: "choice", values: [3] } } }));
  await page.locator("textarea.codeEditor").blur();
  await page.getByRole("button", { name: "Validate and continue" }).click();
  await page.getByLabel("Number of trials").fill("1");
  await page.getByLabel("Optimization metric").selectOption("mae");
  await expect(page.getByText("1 candidate model fit + 1 winner refit planned per model")).toBeVisible();
  await page.getByRole("button", { name: "Start experiment run" }).click();
  await page.getByRole("button", { name: "View results" }).click({ timeout: 30000 });
  await expect(page.getByRole("heading", { name: "Metric summary", exact: true })).toBeVisible();
  const predictionTool = page.getByRole("button", { name: /^Prediction table/ });
  if (!(await predictionTool.getAttribute("class"))?.includes("selected")) await predictionTool.click();
  await page.getByRole("button", { name: "Load rows" }).click();
  await expect(page.getByRole("columnheader", { name: "Actual", exact: true })).toBeVisible();
  await expect(page.getByRole("cell", { name: "85", exact: true })).toBeVisible();
  await expect(page.getByRole("cell", { name: "103", exact: true })).toBeVisible();
  expect(errors).toEqual([]);
});
