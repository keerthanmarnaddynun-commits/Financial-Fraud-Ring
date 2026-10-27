const form = document.querySelector("#run-form");
const sourceSelect = document.querySelector("#source");
const existingField = document.querySelector("#existing-dataset");
const uploadField = document.querySelector("#upload-dataset");
const datasetSelect = document.querySelector("#dataset");
const recordCountField = document.querySelector("#record-count-field");
const runButton = document.querySelector("#run-button");
const formError = document.querySelector("#form-error");

const states = {
  empty: document.querySelector("#empty-state"),
  running: document.querySelector("#running-state"),
  results: document.querySelector("#results"),
  failed: document.querySelector("#failure-state"),
};

function showState(state) {
  Object.entries(states).forEach(([name, element]) => {
    element.hidden = name !== state;
  });
}

async function loadDatasets() {
  try {
    const response = await fetch("/api/datasets");
    const body = await response.json();
    datasetSelect.replaceChildren();
    for (const filename of body.datasets) {
      const option = document.createElement("option");
      option.value = filename;
      option.textContent = filename;
      datasetSelect.append(option);
    }
    if (!body.datasets.length) {
      const option = document.createElement("option");
      option.value = "";
      option.textContent = "No project CSV found";
      option.disabled = true;
      option.selected = true;
      datasetSelect.append(option);
    }
  } catch {
    datasetSelect.replaceChildren();
    const option = document.createElement("option");
    option.value = "";
    option.textContent = "Could not load datasets";
    datasetSelect.append(option);
  }
}

function updateSourceFields() {
  const source = sourceSelect.value;
  existingField.hidden = source !== "existing";
  uploadField.hidden = source !== "upload";
  recordCountField.hidden = source !== "synthetic";
}

function showError(message) {
  formError.textContent = message;
  formError.hidden = false;
}

function renderTable(table, rows, preferredColumns, format = (value) => value) {
  table.replaceChildren();
  if (!rows || rows.length === 0) {
    const body = document.createElement("tbody");
    const row = body.insertRow();
    const cell = row.insertCell();
    cell.colSpan = preferredColumns.length;
    cell.textContent = "No data available.";
    table.append(body);
    return;
  }

  const columns = preferredColumns.filter((column) =>
    rows.some((row) => Object.hasOwn(row, column))
  );
  const head = table.createTHead().insertRow();
  for (const column of columns) {
    const cell = document.createElement("th");
    cell.textContent = column;
    head.append(cell);
  }
  const body = table.createTBody();
  for (const item of rows.slice(0, 12)) {
    const row = body.insertRow();
    for (const column of columns) {
      const cell = row.insertCell();
      const value = item[column];
      cell.textContent = format(value, column);
      if (typeof value === "number" && /F1-Score|Precision|Recall|AUC|Purity|Rate|Completeness|V-Measure|probability/.test(column)) {
        cell.classList.add("metric-value");
      }
    }
  }
}

function parseCsv(text) {
  const rows = [];
  let row = [];
  let value = "";
  let quoted = false;
  for (let index = 0; index < text.length; index += 1) {
    const character = text[index];
    if (quoted && character === '"' && text[index + 1] === '"') {
      value += '"';
      index += 1;
    } else if (character === '"') {
      quoted = !quoted;
    } else if (!quoted && character === ",") {
      row.push(value);
      value = "";
    } else if (!quoted && (character === "\n" || character === "\r")) {
      if (character === "\r" && text[index + 1] === "\n") index += 1;
      row.push(value);
      if (row.some((cell) => cell !== "")) rows.push(row);
      row = [];
      value = "";
    } else {
      value += character;
    }
  }
  if (value !== "" || row.length) {
    row.push(value);
    rows.push(row);
  }
  const [headers, ...data] = rows;
  if (!headers) return [];
  return data.map((cells) =>
    Object.fromEntries(headers.map((header, index) => [header, cells[index] ?? ""]))
  );
}

function setDownloadLink(id, artifacts, key) {
  const link = document.querySelector(id);
  const url = artifacts?.[key];
  link.hidden = !url;
  if (url) link.href = url;
}

function renderReport(job) {
  const report = job.report;
  const transactionRows = report.transaction_level || [];
  const ringRows = report.ring_level || [];
  const scalingRows = report.spark_scaling || [];
  const firstModel = transactionRows[0] || {};
  const testCount = ["True Positive", "False Positive", "False Negative", "True Negative"]
    .reduce((sum, key) => sum + (firstModel[key] || 0), 0);
  const cards = [
    ["Models evaluated", transactionRows.length],
    ["Test transactions", testCount.toLocaleString()],
    ["Fraud rings found", Math.max(0, ...ringRows.map((row) => row["Detected Fraud Rings"] || 0))],
  ];
  const cardContainer = document.querySelector("#summary-cards");
  cardContainer.replaceChildren();
  for (const [label, value] of cards) {
    const card = document.createElement("div");
    card.className = "summary-card";
    const title = document.createElement("span");
    title.textContent = label;
    const number = document.createElement("strong");
    number.textContent = value;
    card.append(title, number);
    cardContainer.append(card);
  }

  renderTable(document.querySelector("#transactions-table"), transactionRows, [
    "Model", "Precision", "Recall", "F1-Score", "AUC-ROC", "PR-AUC",
  ], (value) => typeof value === "number" ? value.toFixed(4) : value);
  renderTable(document.querySelector("#rings-table"), ringRows, [
    "Model", "Total Fraud Rings", "Detected Fraud Rings", "Ring Detection Rate",
    "Cluster Purity", "Cluster Completeness", "V-Measure",
  ], (value, column) => typeof value === "number" && /Rate|Purity|Completeness|V-Measure/.test(column) ? value.toFixed(4) : value);

  document.querySelector("#scaling-section").hidden = scalingRows.length === 0;
  renderTable(document.querySelector("#scaling-table"), scalingRows, [
    "Graph Scale", "Nodes", "Edges", "PageRank Time (s)",
    "Connected Comp Time (s)", "Triangle Count Time (s)", "LPA Time (s)", "Total Time (s)",
  ], (value) => typeof value === "number" ? value.toLocaleString(undefined, { maximumFractionDigits: 3 }) : value);

  const predictionLink = job.artifacts.predictions;
  if (predictionLink) {
    fetch(predictionLink)
      .then((response) => {
        if (!response.ok) throw new Error("Could not load the prediction preview.");
        return response.text();
      })
      .then((csv) => {
        const rows = parseCsv(csv);
        const probabilityColumns = Object.keys(rows[0] || {}).filter((column) => column.endsWith("_probability"));
        rows.sort((left, right) => {
          const leftRisk = Math.max(0, ...probabilityColumns.map((column) => Number(left[column]) || 0));
          const rightRisk = Math.max(0, ...probabilityColumns.map((column) => Number(right[column]) || 0));
          return rightRisk - leftRisk;
        });
        renderTable(document.querySelector("#predictions-table"), rows, [
          "step", "relationship", "src", "dst", "amount", "isFraud",
          "baseline_prediction", "baseline_probability",
          "model_1_node2vec_prediction", "model_1_node2vec_probability",
          "model_2_structural_graphx_prediction", "model_2_structural_graphx_probability",
          "model_3_community_rings_prediction", "model_3_community_rings_probability",
        ]);
      })
      .catch(() => renderTable(document.querySelector("#predictions-table"), [], ["Prediction output"]));
  }

  setDownloadLink("#transactions-download", job.artifacts, "transactions");
  setDownloadLink("#rings-download", job.artifacts, "rings");
  setDownloadLink("#scaling-download", job.artifacts, "scaling");
  setDownloadLink("#predictions-download", job.artifacts, "predictions");
  setDownloadLink("#report-download", job.artifacts, "report");
}

async function watchRun(jobId) {
  document.querySelector("#run-id").textContent = jobId.slice(0, 12);
  try {
    while (true) {
      const response = await fetch(`/api/runs/${jobId}`);
      const job = await response.json();
      if (!response.ok) throw new Error(job.error || "Could not retrieve run status.");
      if (job.status === "completed") {
        renderReport(job);
        showState("results");
        return;
      }
      if (job.status === "failed") {
        document.querySelector("#failure-message").textContent = job.error || "The pipeline returned an unspecified error.";
        showState("failed");
        return;
      }
      await new Promise((resolve) => window.setTimeout(resolve, 2000));
    }
  } catch (error) {
    document.querySelector("#failure-message").textContent = error.message;
    showState("failed");
  } finally {
    runButton.disabled = false;
    runButton.querySelector("span:first-child").textContent = "Run fraud analysis";
  }
}

sourceSelect.addEventListener("change", updateSourceFields);
form.addEventListener("submit", async (event) => {
  event.preventDefault();
  formError.hidden = true;
  if (sourceSelect.value === "existing" && !datasetSelect.value) {
    showError("There is no project CSV to analyze. Upload one or generate synthetic data instead.");
    return;
  }
  if (sourceSelect.value === "upload" && !document.querySelector("#dataset-file").files.length) {
    showError("Choose a CSV file to upload.");
    return;
  }

  runButton.disabled = true;
  runButton.querySelector("span:first-child").textContent = "Starting analysis…";
  const formData = new FormData(form);
  formData.set("run_scaling_benchmark", document.querySelector("#scaling").checked ? "true" : "false");
  showState("running");

  try {
    const response = await fetch("/api/runs", { method: "POST", body: formData });
    const result = await response.json();
    if (!response.ok) throw new Error(result.error || "Could not start the analysis.");
    runButton.querySelector("span:first-child").textContent = "Analysis running…";
    await watchRun(result.job_id);
  } catch (error) {
    showError(error.message);
    showState("empty");
    runButton.disabled = false;
    runButton.querySelector("span:first-child").textContent = "Run fraud analysis";
  }
});

updateSourceFields();
loadDatasets();
