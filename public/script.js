// Frontend logic: every number on the page comes from the Python API.
const $ = (id) => document.getElementById(id);

function esc(value) {
  const div = document.createElement("div");
  div.textContent = String(value);
  return div.innerHTML;
}

async function getJSON(url, options) {
  const res = await fetch(url, options);
  let body = null;
  try { body = await res.json(); } catch (e) { /* not JSON */ }
  if (!res.ok) {
    const err = new Error((body && body.error) || `Request failed (${res.status})`);
    err.details = (body && body.details) || [];
    throw err;
  }
  return body;
}

/* ---------- GET /api/health ---------- */
async function loadHealth() {
  const box = $("status");
  try {
    const h = await getJSON("/api/health");
    box.className = "status status-ok";
    $("status-text").textContent = `Backend online · ${h.dataset_rows} rows · scikit-learn ${h.scikit_learn_version}`;
  } catch (e) {
    box.className = "status status-error";
    $("status-text").textContent = "Backend unreachable";
  }
}

/* ---------- GET /api/metrics ---------- */
function barRow(label, value, max, text, cls) {
  const pct = max > 0 ? (value / max) * 100 : 0;
  return `<div class="bar-row"><span>${esc(label)}</span>
    <div class="bar-track"><div class="bar-fill ${esc(cls || "")}" style="width:${pct}%"></div></div>
    <span class="bar-value">${esc(text)}</span></div>`;
}

async function loadMetrics() {
  try {
    const m = await getJSON("/api/metrics");
    $("stat-total").textContent = m.dataset_size;
    $("stat-split").textContent = `${m.train_size} / ${m.test_size}`;
    $("stat-accuracy").textContent = (m.accuracy * 100).toFixed(1) + "%";
    $("stat-depth").textContent = m.tree_depth;

    const dist = m.class_distribution;
    const maxCount = Math.max(...m.labels.map((l) => dist[l]));
    $("distribution").innerHTML = m.labels
      .map((l) => barRow(l, dist[l], maxCount, `${dist[l]} (${((dist[l] / m.dataset_size) * 100).toFixed(0)}%)`, l))
      .join("");

    const cm = m.confusion_matrix;
    let html = `<table class="cm"><tr><th>Actual ↓ / Predicted →</th>${m.labels.map((l) => `<th>${esc(l)}</th>`).join("")}</tr>`;
    cm.forEach((row, i) => {
      const rowMax = Math.max(...row, 1);
      html += `<tr><th>${esc(m.labels[i])}</th>`;
      row.forEach((n, j) => {
        const alpha = (n / rowMax) * 0.55;
        const colour = i === j ? `rgba(16,185,129,${alpha})` : `rgba(239,68,68,${alpha})`;
        html += `<td class="cell" style="background:${colour}">${n}</td>`;
      });
      html += "</tr>";
    });
    $("confusion").innerHTML = html + "</table>";

    const imp = Object.entries(m.feature_importances).sort((a, b) => b[1] - a[1]);
    const top = imp[0][1] || 1;
    $("importance").innerHTML = imp.map(([name, v]) => barRow(name, v, top, (v * 100).toFixed(1) + "%")).join("");
  } catch (e) {
    ["distribution", "confusion", "importance"].forEach((id) => {
      $(id).innerHTML = `<p class="muted">Could not load metrics: ${esc(e.message)}</p>`;
    });
  }
}

/* ---------- GET /api/students ---------- */
async function loadPreview() {
  try {
    const d = await getJSON("/api/students?limit=10");
    let html = `<table><tr>${d.columns.map((c) => `<th>${esc(c.replaceAll("_", " "))}</th>`).join("")}</tr>`;
    d.rows.forEach((r) => {
      html += "<tr>" + d.columns.map((c) =>
        c === "Performance" ? `<td><span class="badge ${esc(r[c])}">${esc(r[c])}</span></td>` : `<td>${esc(r[c])}</td>`
      ).join("") + "</tr>";
    });
    $("preview").innerHTML = html + "</table>";
  } catch (e) {
    $("preview").innerHTML = `<p class="muted">Could not load data: ${esc(e.message)}</p>`;
  }
}

/* ---------- POST /api/predict ---------- */
$("predict-form").addEventListener("submit", async (event) => {
  event.preventDefault();
  const form = event.target;
  const payload = {};
  new FormData(form).forEach((value, key) => {
    payload[key] = key === "Participation_Level" ? value : (value === "" ? null : Number(value));
  });

  const box = $("result");
  const btn = $("predict-btn");
  btn.disabled = true;
  box.className = "result";
  box.innerHTML = `<p class="muted">Asking the Python backend…</p>`;
  try {
    const r = await getJSON("/api/predict", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });
    const probs = Object.entries(r.probabilities)
      .map(([label, p]) => barRow(label, p, 1, (p * 100).toFixed(0) + "%", label)).join("");
    box.innerHTML = `
      <div class="muted small">Predicted performance</div>
      <div class="big ${esc(r.prediction)}">${esc(r.prediction)}</div>
      <div class="muted small">Leaf confidence: ${(r.confidence * 100).toFixed(0)}%</div>
      <div class="bars">${probs}</div>
      <p><b>Why?</b> ${esc(r.explanation)}</p>`;
  } catch (e) {
    box.className = "result error";
    const list = e.details.length ? `<ul>${e.details.map((d) => `<li>${esc(d)}</li>`).join("")}</ul>` : "";
    box.innerHTML = `<b>${esc(e.message)}</b>${list}`;
  } finally {
    btn.disabled = false;
  }
});

/* ---------- Example buttons (only fill the form; the prediction still comes from the API) ---------- */
const PRESETS = {
  low:  { Attendance_Percentage: 58, Study_Hours_Per_Day: 1.2, Previous_Marks: 42, Assignment_Completion_Percentage: 45, Sleep_Hours: 5.5, Participation_Level: "Low" },
  avg:  { Attendance_Percentage: 76, Study_Hours_Per_Day: 3.2, Previous_Marks: 62, Assignment_Completion_Percentage: 72, Sleep_Hours: 7, Participation_Level: "Medium" },
  high: { Attendance_Percentage: 93, Study_Hours_Per_Day: 5.5, Previous_Marks: 85, Assignment_Completion_Percentage: 92, Sleep_Hours: 7.5, Participation_Level: "High" },
};
document.querySelectorAll("[data-preset]").forEach((chip) => {
  chip.addEventListener("click", () => {
    const preset = PRESETS[chip.dataset.preset];
    Object.entries(preset).forEach(([name, value]) => { $("predict-form").elements[name].value = value; });
  });
});

loadHealth();
loadMetrics();
loadPreview();
