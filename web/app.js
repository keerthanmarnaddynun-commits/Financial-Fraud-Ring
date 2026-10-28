"use strict";

const MODEL_NAMES = {
  tabular_baseline: "Tabular baseline",
  graph_features: "Graph features",
  community_ring: "Community / ring",
};

const $ = (selector) => document.querySelector(selector);
let functionCatalog = [];

function formatNumber(value) {
  if (value === null || value === undefined || !Number.isFinite(Number(value))) return "—";
  return new Intl.NumberFormat("en-US").format(Number(value));
}

function formatScore(value) {
  return value === null || value === undefined || !Number.isFinite(Number(value))
    ? "—"
    : Number(value).toFixed(3);
}

function formatDuration(seconds) {
  if (seconds === null || seconds === undefined || !Number.isFinite(Number(seconds))) return "—";
  const total = Math.round(Number(seconds));
  if (total < 60) return `${total}s`;
  const minutes = Math.floor(total / 60);
  const remaining = total % 60;
  return remaining ? `${minutes}m ${remaining}s` : `${minutes}m`;
}

function formatBytes(bytes) {
  if (bytes === null || bytes === undefined || !Number.isFinite(Number(bytes))) return "—";
  const value = Number(bytes);
  if (value < 1024) return `${formatNumber(value)} B`;
  const units = ["KB", "MB", "GB", "TB"];
  let size = value / 1024;
  let unitIndex = 0;
  while (size >= 1024 && unitIndex < units.length - 1) {
    size /= 1024;
    unitIndex += 1;
  }
  return `${size.toFixed(2)} ${units[unitIndex]}`;
}

function setText(selector, value) {
  const element = $(selector);
  if (element) element.textContent = value;
}

function createIoItem(item) {
  const wrapper = document.createElement("div");
  wrapper.className = "io-item";
  const name = document.createElement("strong");
  name.textContent = item.name;
  const type = document.createElement("code");
  type.textContent = item.type;
  const description = document.createElement("p");
  description.textContent = item.description;
  wrapper.append(name, type, description);
  return wrapper;
}

function renderFunctionCard(item) {
  const details = document.createElement("details");
  details.className = "function-card";
  details.dataset.search = [
    item.name,
    item.file,
    item.purpose,
    item.signature,
    ...item.inputs.map((input) => `${input.name} ${input.type} ${input.description}`),
    ...item.outputs.map((output) => `${output.name} ${output.type} ${output.description}`),
  ].join(" ").toLowerCase();

  const summary = document.createElement("summary");
  const identity = document.createElement("span");
  const name = document.createElement("span");
  name.className = "function-name";
  name.textContent = item.name;
  const file = document.createElement("span");
  file.className = "function-file";
  file.textContent = item.file;
  identity.append(name, file);

  const purpose = document.createElement("span");
  purpose.className = "function-purpose";
  purpose.textContent = item.purpose;
  summary.append(identity, purpose);

  const panel = document.createElement("div");
  panel.className = "function-details";
  const signatureLabel = document.createElement("span");
  signatureLabel.className = "signature-label";
  signatureLabel.textContent = "FUNCTION SIGNATURE";
  const signature = document.createElement("pre");
  signature.className = "signature";
  signature.textContent = item.signature;
  panel.append(signatureLabel, signature);

  const ioGrid = document.createElement("div");
  ioGrid.className = "io-grid";
  for (const [label, entries] of [["INPUTS", item.inputs], ["OUTPUTS", item.outputs]]) {
    const column = document.createElement("div");
    column.className = "io-column";
    const heading = document.createElement("span");
    heading.className = "io-label";
    heading.textContent = label;
    column.append(heading);
    for (const entry of entries) column.append(createIoItem(entry));
    ioGrid.append(column);
  }
  panel.append(ioGrid);
  details.append(summary, panel);
  return details;
}

function renderFunctionCatalog() {
  const grid = $("#function-grid");
  if (!grid) return;
  grid.replaceChildren();
  for (const item of functionCatalog) grid.append(renderFunctionCard(item));
  setText("#function-count", `${functionCatalog.length} functions`);
  filterFunctions();
}

function filterFunctions() {
  const query = ($("#function-search")?.value || "").trim().toLowerCase();
  let visible = 0;
  for (const card of document.querySelectorAll(".function-card")) {
    const matches = !query || card.dataset.search.includes(query);
    card.hidden = !matches;
    if (matches) visible += 1;
  }
  const empty = $("#function-empty");
  if (empty) empty.remove();
  if (!visible) {
    const message = document.createElement("div");
    message.id = "function-empty";
    message.className = "loading-card";
    message.textContent = "No functions match that search. Try another function name, input, or file.";
    $("#function-grid")?.append(message);
  }
}

function addResultRow(body, modelKey, model, taskKey, metrics, sampleSize) {
  const row = document.createElement("tr");
  const modelCell = document.createElement("td");
  modelCell.className = "model-name";
  modelCell.textContent = MODEL_NAMES[modelKey] || modelKey;
  if (taskKey === "transaction" && modelKey === "tabular_baseline") {
    const chip = document.createElement("span");
    chip.className = "model-chip";
    chip.textContent = "BASELINE";
    modelCell.append(chip);
  }
  const taskCell = document.createElement("td");
  taskCell.className = "task-label";
  taskCell.textContent = taskKey === "transaction" ? "Transactions" : "Fraud rings";
  row.append(modelCell, taskCell);
  for (const key of ["precision", "recall", "f1", "roc_auc"]) {
    const cell = document.createElement("td");
    cell.textContent = formatScore(metrics[key]);
    if (key === "roc_auc" || key === "f1") cell.className = "metric-good";
    row.append(cell);
  }
  const sizeCell = document.createElement("td");
  sizeCell.textContent = taskKey === "transaction"
    ? formatNumber(metrics.rows ?? sampleSize)
    : formatNumber(metrics.rings_evaluated);
  row.append(sizeCell);
  body.append(row);
}

function renderResults(evaluation, scalability) {
  const body = $("#results-body");
  if (!body) return;
  body.replaceChildren();
  const modelEntries = Object.entries(evaluation || {});
  const aucs = modelEntries
    .map(([key, result]) => ({ key, auc: Number(result.transaction?.roc_auc) }))
    .filter((entry) => Number.isFinite(entry.auc));
  if (!modelEntries.length) {
    body.innerHTML = '<tr><td colspan="7" class="empty-state">No model results were found in evaluation.json.</td></tr>';
    return;
  }
  for (const [key, result] of modelEntries) {
    addResultRow(body, key, result, "transaction", result.transaction || {}, scalability.test_rows);
    addResultRow(body, key, result, "ring", result.ring || {}, scalability.test_rows);
  }
  if (aucs.length) {
    const best = aucs.reduce((current, entry) => entry.auc > current.auc ? entry : current);
    setText("#best-auc", best.auc.toFixed(3));
    setText("#best-model", `${MODEL_NAMES[best.key] || best.key} · transaction AUC`);
  }
  setText("#report-status", `${modelEntries.length} models · transaction + ring metrics`);
}

function renderScalability(metrics) {
  const timings = metrics.timings_seconds || {};
  const graphFeatureTime = timings.graph_feature_engineering;
  const graphBuildTime = timings.graph_construction;
  const maxTime = Math.max(Number(graphFeatureTime) || 0, Number(graphBuildTime) || 0, 1);
  setText("#transaction-count", formatNumber(metrics.transaction_count));
  setText("#account-count", formatNumber(metrics.account_count));
  setText("#pipeline-runtime", formatDuration(timings.pipeline_total));
  setText("#graph-runtime", formatDuration(graphFeatureTime));
  setText("#graph-construction-runtime", formatDuration(graphBuildTime));
  setText("#executor-memory", formatBytes(metrics.executor_memory_bytes));
  setText("#shuffle-read", formatBytes(metrics.shuffle_read_bytes));
  setText("#shuffle-write", formatBytes(metrics.shuffle_write_bytes));
  setText("#stage-count", formatNumber(metrics.stage_count));
  setText("#sample-count", formatNumber(metrics.sampled_training_rows));
  setText("#spark-app", `${metrics.spark_master || "Spark"} · ${metrics.spark_application_id || "application ID unavailable"}`);
  const bar = $("#graph-runtime-bar");
  if (bar) bar.style.width = `${Math.max(3, (Number(graphFeatureTime) / maxTime) * 100)}%`;
}

async function loadHealth() {
  const response = await fetch("/api/health", { cache: "no-store" });
  if (!response.ok) throw new Error(`Health endpoint returned HTTP ${response.status}`);
  const health = await response.json();
  const statusElement = $(".sidebar-bottom .status-dot");
  if (statusElement) statusElement.classList.toggle("offline", health.status !== "online");
  setText("#sidebar-status", health.status === "online" ? "Project is online" : "Project unavailable");
  const dataset = health.dataset || {};
  setText("#dataset-status", dataset.available
    ? `${dataset.filename} · ${formatBytes(dataset.size_bytes)}`
    : "Dataset CSV not found");
  setText("#dataset-name", dataset.available ? dataset.filename : "PaySim dataset not found");
  setText("#hero-status", health.reports_available ? "Latest run reports available" : "Run pipeline to generate reports");
}

async function loadReports() {
  const response = await fetch("/api/reports", { cache: "no-store" });
  const payload = await response.json();
  if (!response.ok) throw new Error(payload.error || `Report endpoint returned HTTP ${response.status}`);
  renderResults(payload.evaluation, payload.scalability);
  renderScalability(payload.scalability);
  const generatedAt = payload.generated_at
    ? new Date(payload.generated_at * 1000).toLocaleString()
    : "timestamp unavailable";
  setText("#last-updated", `Reports generated · ${generatedAt}`);
}

async function loadFunctions() {
  const response = await fetch("/api/functions", { cache: "no-store" });
  if (!response.ok) throw new Error(`Function catalog endpoint returned HTTP ${response.status}`);
  const payload = await response.json();
  functionCatalog = payload.functions;
  renderFunctionCatalog();
}

function showLoadError(error) {
  console.error(error);
  const body = $("#results-body");
  if (body) {
    body.replaceChildren();
    const row = document.createElement("tr");
    const cell = document.createElement("td");
    cell.colSpan = 7;
    cell.className = "empty-state";
    cell.textContent = error.message;
    row.append(cell);
    body.append(row);
  }
  setText("#report-status", "Report unavailable");
  setText("#hero-status", "Report data could not be loaded");
}

async function refreshDashboard() {
  const button = $("#refresh-button");
  if (button) button.disabled = true;
  try {
    await Promise.all([loadHealth(), loadReports()]);
  } catch (error) {
    showLoadError(error);
  } finally {
    if (button) button.disabled = false;
  }
}

function setupNavigation() {
  const links = [...document.querySelectorAll(".nav-link")];
  const sections = links
    .map((link) => document.querySelector(link.getAttribute("href")))
    .filter(Boolean);
  const observer = new IntersectionObserver((entries) => {
    for (const entry of entries) {
      if (!entry.isIntersecting) continue;
      for (const link of links) {
        link.classList.toggle("active", link.getAttribute("href") === `#${entry.target.id}`);
      }
    }
  }, { rootMargin: "-25% 0px -65% 0px" });
  sections.forEach((section) => observer.observe(section));
}

document.addEventListener("DOMContentLoaded", () => {
  $("#function-search")?.addEventListener("input", filterFunctions);
  $("#refresh-button")?.addEventListener("click", refreshDashboard);
  Promise.all([loadFunctions(), refreshDashboard()]).catch(showLoadError);
  setupNavigation();
});
