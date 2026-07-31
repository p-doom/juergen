"use strict";

const state = { index: null, traces: [], selected: null, step: 0 };
const filterNames = ["experiment", "model", "modality", "grammar", "scale", "prose", "typing_format", "rank", "seed", "outcome"];

function el(tag, className = "", text = null) {
  const node = document.createElement(tag);
  if (className) node.className = className;
  if (text !== null) node.textContent = text;
  return node;
}

function valueText(value) {
  if (value === null || value === undefined || value === "") return "—";
  if (typeof value === "boolean") return value ? "yes" : "no";
  if (typeof value === "number") return Number.isInteger(value) ? String(value) : value.toFixed(3);
  if (typeof value === "object") return JSON.stringify(value);
  return String(value);
}

function pct(value) {
  return value === null || value === undefined ? "—" : `${(100 * value).toFixed(1)}%`;
}

function metricValue(run) {
  const m = run.metrics;
  if (m.mean_endpoint_error_px !== null) return `${m.mean_endpoint_error_px.toFixed(1)} px`;
  if (m.mean_action_span !== null) return `${m.mean_action_span.toFixed(1)} events`;
  return "—";
}

function uniqueValues(key) {
  return [...new Set(state.index.traces.map(t => t[key]).filter(v => v !== null && v !== undefined && v !== ""))]
    .sort((a, b) => String(a).localeCompare(String(b), undefined, { numeric: true }));
}

function setupFilters() {
  document.querySelectorAll("select[data-filter]").forEach(select => {
    const key = select.dataset.filter;
    if (key === "outcome") return;
    uniqueValues(key).forEach(value => {
      const option = el("option", "", valueText(value));
      option.value = String(value);
      select.appendChild(option);
    });
    select.addEventListener("change", applyFilters);
  });
  document.querySelector('select[data-filter="outcome"]').addEventListener("change", applyFilters);
  document.getElementById("reset-filters").addEventListener("click", () => {
    document.querySelectorAll("select[data-filter]").forEach(select => { select.value = ""; });
    applyFilters();
  });
}

function currentFilters() {
  const result = {};
  filterNames.forEach(name => { result[name] = document.querySelector(`select[data-filter="${name}"]`).value; });
  return result;
}

function traceMatches(trace, filters) {
  for (const key of filterNames) {
    const wanted = filters[key];
    if (!wanted) continue;
    if (key === "outcome") {
      if ((wanted === "success") !== Boolean(trace.success)) return false;
    } else if (String(trace[key] ?? "") !== wanted) return false;
  }
  return true;
}

function applyFilters() {
  const filters = currentFilters();
  state.traces = state.index.traces.filter(trace => traceMatches(trace, filters));
  if (!state.selected || !state.traces.some(trace => trace.id === state.selected.id)) {
    state.selected = state.traces[0] || null;
    state.step = 0;
  }
  renderCards();
  renderOverview();
  renderTraceList();
  renderDetail();
}

function renderCards() {
  const cards = document.getElementById("cards");
  cards.replaceChildren();
  const n = state.traces.length;
  const steps = state.traces.reduce((sum, trace) => sum + trace.steps.length, 0);
  const rate = key => n ? state.traces.filter(trace => trace[key]).length / n : null;
  const specs = [
    [n, "visible traces"], [steps, "action steps"], [pct(rate("success")), "task success"],
    [pct(rate("parse_ok")), "parse validity"], [pct(rate("format_ok")), "format validity"],
  ];
  specs.forEach(([value, label]) => {
    const card = el("div", "card");
    card.append(el("div", "eyebrow", label), el("div", "value", valueText(value)), el("div", "label", "current filter slice"));
    cards.appendChild(card);
  });
}

function visibleRuns() {
  const ids = new Set(state.traces.map(trace => trace.run_id));
  return state.index.runs.filter(run => ids.has(run.run_id));
}

function renderOverview() {
  const body = document.getElementById("overview-body");
  body.replaceChildren();
  const runs = visibleRuns();
  document.getElementById("overview-count").textContent = `${runs.length} sealed runs`;
  runs.forEach(run => {
    const row = document.createElement("tr");
    const first = el("td", "primary-cell", run.experiment);
    first.appendChild(el("span", "subcell", run.arm));
    const cells = [first, el("td", "", run.checkpoint), el("td", "", valueText(run.rank)), el("td", "", valueText(run.seed)),
      el("td", "", String(run.n)), el("td", "", pct(run.metrics.task_success_rate)),
      el("td", "", pct(run.metrics.exact_typing_rate ?? run.metrics.in_box_rate)), el("td", "", pct(run.metrics.parse_rate)),
      el("td", "", pct(run.metrics.format_validity_rate)), el("td", "", metricValue(run))];
    cells.forEach(cell => row.appendChild(cell));
    row.addEventListener("click", () => {
      state.selected = state.traces.find(trace => trace.run_id === run.run_id) || null;
      state.step = 0;
      renderTraceList(); renderDetail();
      document.getElementById("trace-detail").scrollIntoView({ behavior: "smooth", block: "start" });
    });
    body.appendChild(row);
  });
}

function renderTraceList() {
  const list = document.getElementById("trace-list");
  list.replaceChildren();
  document.getElementById("trace-count").textContent = `${state.traces.length} traces`;
  state.traces.slice(0, 600).forEach(trace => {
    const button = el("button", `trace-item${state.selected?.id === trace.id ? " selected" : ""}`);
    button.type = "button";
    const top = el("div", "topline");
    top.append(el("span", "", `${trace.arm} · ${trace.scale}`), el("span", `dot ${trace.success ? "good" : ""}`));
    button.append(top, el("div", "meta", trace.id.split(":").slice(1).join(":")));
    button.addEventListener("click", () => { state.selected = trace; state.step = 0; renderTraceList(); renderDetail(); });
    list.appendChild(button);
  });
  if (state.traces.length > 600) list.appendChild(el("p", "muted", `Showing first 600 of ${state.traces.length} matching traces.`));
}

function addBadge(container, text, kind = "") { container.appendChild(el("span", `badge ${kind}`, text)); }

function renderDetail() {
  const detail = document.getElementById("trace-detail");
  detail.replaceChildren();
  const trace = state.selected;
  if (!trace) { detail.appendChild(el("div", "empty-state", "No trace matches the current filters.")); return; }
  const step = trace.steps[state.step];
  const run = state.index.runs.find(item => item.run_id === trace.run_id);

  const header = el("div", "detail-header");
  const title = el("div");
  title.append(el("p", "eyebrow", `${trace.experiment} · ${trace.modality}`), el("h2", "", trace.model));
  const badges = el("div", "badges");
  addBadge(badges, trace.success ? "success" : "failure", trace.success ? "good" : "bad");
  addBadge(badges, trace.parse_ok ? "parse valid" : "parse invalid", trace.parse_ok ? "good" : "bad");
  addBadge(badges, trace.format_ok ? "format valid" : "format invalid", trace.format_ok ? "good" : "bad");
  addBadge(badges, trace.grammar || "grammar —"); addBadge(badges, `rank ${valueText(trace.rank)}`);
  title.appendChild(badges);
  header.append(title, renderStepNav(trace));
  detail.appendChild(header);

  const grid = el("div", "detail-grid");
  const left = el("div");
  left.appendChild(renderVisual(step));
  const instruction = el("section", "evidence-card");
  instruction.append(el("h3", "", "Instruction"), el("div", "instruction", step.instruction || "Not sealed with this row."));
  left.appendChild(instruction);
  if (step.typing) left.appendChild(renderTyping(step.typing));

  const right = el("div");
  right.appendChild(renderMetrics(step));
  right.appendChild(renderCodePair(step));
  right.appendChild(renderOutcome(step));
  right.appendChild(renderProvenance(trace, run));
  grid.append(left, right);
  detail.appendChild(grid);
}

function renderStepNav(trace) {
  const nav = el("div", "step-nav");
  const previous = el("button", "", "←"); previous.type = "button"; previous.disabled = state.step === 0;
  const next = el("button", "", "→"); next.type = "button"; next.disabled = state.step >= trace.steps.length - 1;
  previous.addEventListener("click", () => { state.step -= 1; renderDetail(); });
  next.addEventListener("click", () => { state.step += 1; renderDetail(); });
  nav.append(previous, el("span", "muted", `step ${state.step + 1} / ${trace.steps.length}`), next);
  return nav;
}

function renderVisual(step) {
  const card = el("section", "visual-card");
  card.appendChild(el("h3", "", "Observation and movement overlay"));
  const stage = el("div", "visual-stage");
  if (step.screenshot) {
    const image = document.createElement("img"); image.alt = "Sealed evaluation screenshot"; image.src = step.screenshot;
    stage.appendChild(image);
    image.addEventListener("error", () => { image.replaceWith(el("div", "no-visual", "Screenshot link is unavailable. Regenerate the bundle to refresh sealed asset links.")); });
  } else {
    stage.append(el("div", "schematic"), el("div", "no-visual", step.overlay ? "No screenshot bytes were sealed for this row. Geometry is shown on a schematic 1440×900 canvas." : "No screenshot or geometry was sealed for this row."));
  }
  if (step.overlay) stage.appendChild(overlaySvg(step.overlay, step.image_size || [1440, 900]));
  card.appendChild(stage);
  const legend = el("div", "legend");
  [["#5ccfe6", "cursor"], ["#70e1b2", "ideal vector"], ["#ffbd69", "predicted vector"], ["#ff7a83", "target / error"]].forEach(([color, label]) => {
    const span = el("span"); const swatch = el("i"); swatch.style.background = color; span.append(swatch, document.createTextNode(label)); legend.appendChild(span);
  });
  card.appendChild(legend);
  return card;
}

function overlaySvg(overlay, size) {
  const ns = "http://www.w3.org/2000/svg";
  const svg = document.createElementNS(ns, "svg");
  svg.setAttribute("viewBox", `0 0 ${size[0]} ${size[1]}`); svg.setAttribute("preserveAspectRatio", "xMidYMid meet");
  const defs = document.createElementNS(ns, "defs");
  [["ideal-arrow", "#70e1b2"], ["pred-arrow", "#ffbd69"]].forEach(([id, color]) => {
    const marker = document.createElementNS(ns, "marker"); marker.setAttribute("id", id); marker.setAttribute("viewBox", "0 0 10 10"); marker.setAttribute("refX", "8"); marker.setAttribute("refY", "5"); marker.setAttribute("markerWidth", "7"); marker.setAttribute("markerHeight", "7"); marker.setAttribute("orient", "auto-start-reverse");
    const path = document.createElementNS(ns, "path"); path.setAttribute("d", "M 0 0 L 10 5 L 0 10 z"); path.setAttribute("fill", color); marker.appendChild(path); defs.appendChild(marker);
  });
  svg.appendChild(defs);
  function shape(tag, attrs) { const node = document.createElementNS(ns, tag); Object.entries(attrs).forEach(([k, v]) => node.setAttribute(k, String(v))); svg.appendChild(node); }
  if (Array.isArray(overlay.bbox)) shape("rect", { x: overlay.bbox[0], y: overlay.bbox[1], width: overlay.bbox[2] - overlay.bbox[0], height: overlay.bbox[3] - overlay.bbox[1], fill: "rgba(255,122,131,.12)", stroke: "#ff7a83", "stroke-width": 5 });
  const cursor = overlay.cursor;
  if (Array.isArray(cursor)) shape("circle", { cx: cursor[0], cy: cursor[1], r: 11, fill: "#5ccfe6", stroke: "#07100d", "stroke-width": 3 });
  if (Array.isArray(cursor) && Array.isArray(overlay.ideal_landing)) shape("line", { x1: cursor[0], y1: cursor[1], x2: overlay.ideal_landing[0], y2: overlay.ideal_landing[1], stroke: "#70e1b2", "stroke-width": 6, "stroke-dasharray": "12 9", "marker-end": "url(#ideal-arrow)" });
  if (Array.isArray(cursor) && Array.isArray(overlay.predicted_landing)) shape("line", { x1: cursor[0], y1: cursor[1], x2: overlay.predicted_landing[0], y2: overlay.predicted_landing[1], stroke: "#ffbd69", "stroke-width": 7, "marker-end": "url(#pred-arrow)" });
  if (Array.isArray(overlay.predicted_landing)) shape("circle", { cx: overlay.predicted_landing[0], cy: overlay.predicted_landing[1], r: 10, fill: "#ffbd69", stroke: "#07100d", "stroke-width": 3 });
  if (Array.isArray(overlay.target)) shape("circle", { cx: overlay.target[0], cy: overlay.target[1], r: 7, fill: "#ff7a83" });
  return svg;
}

function renderMetrics(step) {
  const card = el("section", "evidence-card"); card.appendChild(el("h3", "", "Step metrics"));
  const strip = el("div", "metric-strip");
  const entries = Object.entries(step.metrics || {});
  if (!entries.length) strip.appendChild(el("span", "muted", "No numeric step metrics."));
  entries.forEach(([key, value]) => { const item = el("div", "metric"); item.append(el("b", "", valueText(value)), el("span", "", key.replaceAll("_", " "))); strip.appendChild(item); });
  card.appendChild(strip); return card;
}

function renderCodePair(step) {
  const card = el("section", "evidence-card"); card.appendChild(el("h3", "", "Model output and action comparison"));
  const raw = el("pre", "", step.raw_output || "(empty output)"); card.appendChild(raw);
  const pair = el("div", "two-col"); pair.style.marginTop = ".7rem";
  [["Parsed action", step.parsed_action], ["Gold action", step.gold_action]].forEach(([title, value]) => { const box = el("div"); box.append(el("h3", "", title), el("pre", "", JSON.stringify(value, null, 2))); pair.appendChild(box); });
  card.appendChild(pair); return card;
}

function renderOutcome(step) {
  const card = el("section", "evidence-card"); card.append(el("h3", "", "Reward / outcome"), el("pre", "", JSON.stringify(step.outcome, null, 2))); return card;
}

function renderProvenance(trace, run) {
  const card = el("section", "evidence-card"); card.appendChild(el("h3", "", "Sealed provenance"));
  const dl = el("dl", "kv");
  const fields = { "run ID": trace.run_id, "Slurm job": trace.job_id, "artifact ID": trace.artifact_id, recipe: trace.recipe, manifest: run?.manifest, "manifest SHA-256": run?.manifest_sha256 };
  Object.entries(fields).forEach(([key, value]) => { dl.append(el("dt", "", key), el("dd", "", valueText(value))); });
  card.appendChild(dl); return card;
}

function renderTyping(typing) {
  const card = el("section", "evidence-card typing-panel"); card.appendChild(el("h3", "", "Typing execution"));
  [["Requested text", typing.requested], ["Executed text", typing.executed]].forEach(([title, text]) => { card.append(el("p", "eyebrow", title), el("div", "typing-text", text)); });
  card.append(el("p", "eyebrow", "Character-level diff"), charDiff(typing.requested, typing.executed));
  card.appendChild(el("p", "eyebrow", "Raw coalesced / per-key events"));
  const events = el("div", "events"); (typing.events || []).forEach(event => events.appendChild(el("span", "event", event))); card.appendChild(events);
  return card;
}

function charDiff(a, b) {
  const n = a.length, m = b.length; const dp = Array.from({ length: n + 1 }, () => new Uint16Array(m + 1));
  for (let i = n - 1; i >= 0; i--) for (let j = m - 1; j >= 0; j--) dp[i][j] = a[i] === b[j] ? dp[i + 1][j + 1] + 1 : Math.max(dp[i + 1][j], dp[i][j + 1]);
  const box = el("div", "typing-text char-diff"); let i = 0, j = 0;
  while (i < n || j < m) {
    if (i < n && j < m && a[i] === b[j]) { box.appendChild(el("span", "equal", a[i])); i++; j++; }
    else if (j < m && (i === n || dp[i][j + 1] >= dp[i + 1][j])) { box.appendChild(el("span", "insert", b[j])); j++; }
    else { box.appendChild(el("span", "delete", a[i])); i++; }
  }
  return box;
}

async function start() {
  const status = document.getElementById("status");
  try {
    const response = await fetch("data/index.json", { cache: "no-store" });
    if (!response.ok) throw new Error(`HTTP ${response.status} loading data/index.json`);
    state.index = await response.json();
    if (state.index.status !== "complete") {
      status.textContent = "Audit failed"; status.classList.add("bad");
      const panel = document.getElementById("error-panel"); panel.classList.remove("hidden"); panel.textContent = `The generator refused to publish partial evidence:\n\n${(state.index.errors || []).join("\n")}`;
      return;
    }
    status.textContent = `${state.index.runs.length} sealed runs`; status.classList.add("ok");
    document.getElementById("generated-at").textContent = `Generated ${new Date(state.index.generated_at).toLocaleString()}`;
    setupFilters(); applyFilters();
  } catch (error) {
    status.textContent = "Index unavailable"; status.classList.add("bad");
    const panel = document.getElementById("error-panel"); panel.classList.remove("hidden"); panel.textContent = `Could not open audited trace index: ${error.message}\n\nServe this folder over HTTP; browsers do not allow fetch() from file://.`;
  }
}

start();
