"use strict";

const state = {
  traces: [], trace: null, run: null, graph: null, a: null, b: null,
  hardware: null, hardwareSelection: {type: "link", id: "hbm_l2"},
  dataflowSelection: "a_global_load",
  comparisonDataflowSelection: {a: null, b: null},
  scale: 1, x: 30, y: 30, iteration: 0, playTimer: null,
};
const $ = id => document.getElementById(id);
const query = values => new URLSearchParams(values).toString();
const short = value => value && value.length > 12 ? value.slice(0, 8) : value;
const number = (value, digits = 4) => value == null ? "—" : Number(value).toFixed(digits);
const escapeHtml = value => String(value ?? "").replace(
  /[&<>"']/g,
  character => ({"&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;"})[character],
);

async function api(path) {
  const response = await fetch(path);
  const value = await response.json();
  if (!response.ok) throw new Error(value.error || `HTTP ${response.status}`);
  return value;
}

async function loadCatalog() {
  const value = await api("/api/traces");
  state.traces = value.traces;
  $("trace-select").innerHTML = state.traces.map(item =>
    `<option value="${item.trace_id}" ${item.error ? "disabled" : ""}>` +
    `${escapeHtml(item.name)}${item.error ? " (invalid)" : ""}</option>`
  ).join("");
  $("compare-trace").innerHTML = state.traces.map(item =>
    `<option value="${item.trace_id}" ${item.error ? "disabled" : ""}>` +
    `${escapeHtml(item.name)}${item.error ? " (invalid)" : ""}</option>`
  ).join("");
  const first = state.traces.find(item => !item.error && item.runs.length);
  if (!first) {
    $("run-summary").textContent = "No supported SQLite traces found.";
    return;
  }
  $("trace-select").value = first.trace_id;
  $("compare-trace").value = first.trace_id;
  chooseCompareTrace();
  chooseTrace();
}

function chooseCompareTrace() {
  const trace = state.traces.find(item => item.trace_id === $("compare-trace").value);
  $("compare-run").innerHTML = (trace?.runs || []).map(run =>
    `<option value="${run.run_id}">${escapeHtml(run.started_at)} · ` +
    `${escapeHtml(run.benchmark_id)} · ${short(run.run_id)}</option>`
  ).join("");
}

function chooseTrace() {
  state.trace = state.traces.find(item => item.trace_id === $("trace-select").value);
  $("run-select").innerHTML = (state.trace?.runs || []).map(run =>
    `<option value="${run.run_id}">${escapeHtml(run.started_at)} · ` +
    `${escapeHtml(run.benchmark_id)} · ${short(run.run_id)}</option>`
  ).join("");
  if (state.trace?.runs.length) {
    $("run-select").value = state.trace.runs[0].run_id;
    loadRun();
  }
}

async function loadRun() {
  if (state.playTimer) {
    clearInterval(state.playTimer);
    state.playTimer = null;
    $("play-iterations").textContent = "Play";
  }
  state.run = $("run-select").value;
  state.a = null;
  state.b = null;
  state.hardware = null;
  $("hardware-visual").hidden = true;
  $("hardware-constraints").hidden = true;
  $("dataflow-panel").hidden = true;
  $("hardware-summary").textContent = "Loading GPU resources…";
  try {
    state.graph = await api(`/api/graph?${query({trace: state.trace.trace_id, run: state.run})}`);
    const run = state.graph.run;
    $("run-summary").textContent = `${run.status} · ${run.benchmark_id} · model ` +
      `${run.model_name || "—"} · B_gen ${run.b_gen}/${run.generation_budget ?? "?"} · ` +
      `B_mut ${run.b_mut || 0}/${run.mutation_budget ?? 0} · ` +
      `${run.node_count} nodes · best ${number(run.best_speedup, 3)}×`;
    $("score-notice").textContent = state.graph.historical_scores_available
      ? "Decision-time PUCT/UCB scores available."
      : "Edge values shown are final statistics. Existing traces do not contain exact historical PUCT/UCB candidate scores.";
    clearSelections();
    loadAnalysis();
    loadDecisions();
    renderGraph();
    fitGraph();
    await loadHardware();
  } catch (error) {
    showError(error);
  }
}

async function loadHardware() {
  const traceId = state.trace.trace_id, runId = state.run;
  try {
    const value = await api(`/api/hardware?${query({trace: traceId, run: runId})}`);
    if (state.trace?.trace_id !== traceId || state.run !== runId) return;
    state.hardware = value;
    const select = $("hardware-node");
    select.innerHTML = '<option value="">No kernel selected</option>' + value.kernels.map(row =>
      `<option value="${escapeHtml(row.node_id)}">${escapeHtml(short(row.node_id))}` +
      `${value.observations.some(item => item.node_id === row.node_id) ? " · NCU" : ""}</option>`
    ).join("");
    select.value = value.observations[0]?.node_id || value.kernels[0]?.node_id || "";
    renderHardware();
    if (state.a) renderNodeDataflow("a", state.a.node_id);
    if (state.b) renderNodeDataflow("b", state.b.node_id);
  } catch (error) {
    $("hardware-summary").textContent = `GPU resources unavailable: ${error.message}`;
  }
}

const hardwarePositions = {
  hbm: {x: 30, y: 165, w: 140, h: 75, label: "HBM"},
  l2: {x: 230, y: 165, w: 140, h: 75, label: "L2 cache"},
  sm: {x: 420, y: 20, w: 160, h: 34, label: "SMs"},
  l1: {x: 450, y: 80, w: 135, h: 70, label: "L1 cache"},
  shared: {x: 450, y: 250, w: 135, h: 70, label: "Shared memory"},
  registers: {x: 675, y: 165, w: 135, h: 75, label: "Registers"},
  scalar_compute: {x: 875, y: 80, w: 150, h: 70, label: "CUDA cores"},
  tensor_compute: {x: 875, y: 250, w: 150, h: 70, label: "Tensor cores"},
};
const hardwareRoutes = {
  hbm_l2: [170, 202, 230, 202], l2_l1: [370, 185, 450, 115],
  l1_registers: [585, 115, 675, 190], l1_shared: [517, 150, 517, 250],
  l2_shared: [370, 220, 450, 280], shared_registers: [585, 280, 675, 218],
  registers_scalar: [810, 190, 875, 115], shared_tensor: [585, 300, 875, 300],
  tensor_registers: [875, 265, 810, 220],
};
function hardwareValue(evidence) {
  if (!evidence || evidence.value == null) return "unknown";
  const value = Number(evidence.value);
  if (evidence.unit === "bytes/s") return `${(value / 1e12).toFixed(2)} TB/s`;
  if (evidence.unit === "bytes") {
    if (value >= 1e9) return `${(value / 2 ** 30).toFixed(1)} GiB`;
    if (value >= 1e6) return `${(value / 1e6).toFixed(0)} MB`;
    return `${(value / 1024).toFixed(0)} KiB`;
  }
  return `${value.toLocaleString()} ${evidence.unit}`;
}
function hardwareFacts(resource) {
  const fields = ["capacity", "maximum_capacity", "count", "units"];
  return fields.filter(key => resource[key]).map(key =>
    `${key.replaceAll("_", " ")}: ${hardwareValue(resource[key])} (${resource[key].basis})` +
    (resource[key].source ? `\nsource: ${resource[key].source}` : "")
  );
}
function renderHardware() {
  const value = state.hardware;
  const graph = value?.graph;
  const svg = $("hardware-graph");
  svg.replaceChildren();
  if (!graph || !Array.isArray(graph.nodes) || !Array.isArray(graph.links)) {
    $("hardware-visual").hidden = true;
    $("hardware-constraints").hidden = true;
    $("dataflow-panel").hidden = true;
    $("hardware-summary").textContent = "No supported H100 SXM 80 GB resource graph is available for this run.";
    return;
  }
  $("hardware-visual").hidden = false;
  $("hardware-constraints").hidden = false;
  const selectedNodeId = $("hardware-node").value;
  const profile = value.observations.find(row => row.node_id === selectedNodeId);
  const kernel = value.kernels.find(row => row.node_id === selectedNodeId);
  const selectedDataflowEdge = kernel?.dataflow?.edges.find(row => row.id === state.dataflowSelection);
  const mappedLinks = new Set(selectedDataflowEdge?.hardware_links || []);
  const metricsByResource = new Map();
  (profile?.observations || []).forEach(row => {
    if (!metricsByResource.has(row.resource_id)) metricsByResource.set(row.resource_id, []);
    metricsByResource.get(row.resource_id).push(row);
  });
  $("hardware-summary").textContent = `${graph.model} · graph v${graph.schema_version} · ` +
    `${value.source === "persisted" ? "stored in trace" : "derived from stored manifest"} · ` +
    `${kernel ? `kernel ${short(kernel.node_id)}` : "no kernel selected"} · ` +
    `${profile ? "NCU profile available" : "no NCU profile"}. ` +
    "Paths are conceptual; link costs are not calibrated.";

  const defs = svgElement("defs");
  defs.innerHTML = '<marker id="hardware-arrow" markerWidth="8" markerHeight="8" refX="7" refY="3" orient="auto"><path d="M0,0 L0,6 L8,3 z" fill="#6e7681"/></marker>';
  svg.appendChild(defs);
  svg.appendChild(svgElement("rect", {x: 405, y: 8, width: 665, height: 365, rx: 12, class: "hardware-sm-region"}));
  graph.links.forEach(link => {
    const route = hardwareRoutes[link.id];
    if (!route) return;
    const selected = state.hardwareSelection.type === "link" && state.hardwareSelection.id === link.id;
    const group = svgElement("g", {class: `hardware-link${selected ? " selected" : ""}${mappedLinks.has(link.id) ? " mapped" : ""}`, tabindex: "0", role: "button", "aria-label": `${link.source} to ${link.target}: ${link.pathway}`});
    const [x1, y1, x2, y2] = route;
    group.appendChild(svgElement("line", {x1, y1, x2, y2, class: "hardware-link-line"}));
    group.appendChild(svgElement("line", {x1, y1, x2, y2, class: "hardware-link-hit"}));
    activateHardware(group, "link", link.id);
    svg.appendChild(group);
  });
  graph.nodes.forEach(resource => {
    const position = hardwarePositions[resource.id];
    if (!position) return;
    const selected = state.hardwareSelection.type === "node" && state.hardwareSelection.id === resource.id;
    const group = svgElement("g", {class: `hardware-node${selected ? " selected" : ""}`, tabindex: "0", role: "button", "aria-label": position.label});
    group.appendChild(svgElement("rect", {x: position.x, y: position.y, width: position.w, height: position.h, rx: 8}));
    text(group, position.x + 10, position.y + (resource.id === "sm" ? 22 : 23), position.label, "hardware-label");
    if (resource.id !== "sm") {
      const fact = resource.capacity || resource.maximum_capacity || resource.units;
      text(group, position.x + 10, position.y + 43, hardwareValue(fact), "hardware-sub");
      const metrics = metricsByResource.get(resource.id) || [];
      text(group, position.x + 10, position.y + 60,
        metrics.length ? `NCU ${number(metrics[0].value, 2)}%` : "", "hardware-profile");
    } else {
      text(group, position.x + 75, position.y + 22, `× ${resource.count?.value ?? "?"}`, "hardware-sub");
    }
    activateHardware(group, "node", resource.id);
    svg.appendChild(group);
  });
  showHardwareDetail(graph, metricsByResource);
  renderHardwareConstraints(graph, kernel);
  renderDataflow(kernel);
}
function dataflowSummary(kernel) {
  const graph = kernel.dataflow;
  const summary = graph.summary;
  const provenance = kernel.dataflow_source === "persisted" ? "stored in trace" : "derived from stored node";
  const compiled = graph.compiled_ptx;
  const compiledText = compiled
    ? ` PTX dependency evidence: ${compiled.status}, ${compiled.instruction_count ?? 0} instruction sites, ${compiled.edges.length} proven local edges; unresolved paths are recorded separately.`
    : " PTX dependency evidence unavailable.";
  if (graph.kind === "logical_and_static_sites") {
    const sites = summary.static_sass_instruction_sites;
    const siteText = sites ? Object.entries(sites).map(([name, count]) => `${name.replaceAll("_", " ")} ${count}`).join(" · ") : "SASS evidence unavailable";
    return `Logical workload graph and static SASS inventory · ${provenance} · ${siteText}. ` +
      "Arrows show only workload inputs and output. SASS sites may include inactive branches; physical routes and runtime bytes are unknown." + compiledText;
  }
  return `${graph.template.replaceAll("_", " ")} · ${provenance} · ` +
    `source-level global loads ${hardwareValue({value: summary.source_global_load_bytes, unit: "bytes"})} · ` +
    `stores ${hardwareValue({value: summary.source_global_store_bytes, unit: "bytes"})} · ` +
    `${summary.source_multiply_accumulates.toLocaleString()} multiply-accumulates · ` +
    `${summary.barriers_per_cta} CTA barriers per block. ` +
    "Edge values are source-level references, not measured HBM or L2 traffic." + compiledText;
}

function dataflowEdgeLabel(graph, edge) {
  if (edge.operation.startsWith("logical_")) return "logical";
  const quantity = edge.source_level_quantity;
  if (quantity?.value != null) {
    if (quantity.unit === "bytes") return hardwareValue(quantity);
    const count = new Intl.NumberFormat("en", {notation: "compact", maximumFractionDigits: 1}).format(quantity.value);
    return `${count} ${quantity.unit === "source operations" ? "ops" : quantity.unit}`;
  }
  const sites = graph.summary?.static_sass_instruction_sites;
  if (sites && Number.isInteger(sites[edge.operation])) return `${sites[edge.operation]} SASS sites`;
  return "unknown";
}

function dataflowEdgeDetail(graph, edge) {
  if (!edge) return "No dataflow edges.";
  const quantity = edge.source_level_quantity;
  const evidence = edge.operation.startsWith("logical_")
    ? "Workload-level relationship; compiled route and runtime quantity unknown"
    : quantity?.value != null
    ? `Source-level quantity: ${hardwareValue(quantity)} (${quantity.basis})`
    : `Static evidence: ${dataflowEdgeLabel(graph, edge)}; runtime quantity unknown`;
  return `${edge.operation}: ${edge.source} → ${edge.target}\n${evidence}\n` +
    `Mapped GPU links: ${edge.hardware_links.join(" → ") || "none"}\n` +
    `Measured bytes on those links: unknown\n${edge.note || ""}`;
}

function drawDataflowDiagram(svg, graph, selectedId, choose) {
  svg.replaceChildren();
  svg.setAttribute("viewBox", graph.kind === "logical_and_static_sites" ? "0 0 1240 390" : "0 0 1240 290");
  const defs = svgElement("defs");
  const markerId = `${svg.id}-arrow`;
  defs.innerHTML = `<marker id="${markerId}" markerWidth="8" markerHeight="8" refX="7" refY="3" orient="auto"><path d="M0,0 L0,6 L8,3 z" fill="#6e7681"/></marker>`;
  svg.appendChild(defs);
  const positions = new Map(graph.nodes.map(node => [node.id, {
    x: 25 + node.column * 205, y: 15 + node.lane * 95,
  }]));
  if (graph.kind === "logical_and_static_sites" && graph.nodes.some(node => node.id.startsWith("sass_"))) {
    svg.appendChild(svgElement("line", {x1: 25, y1: 275, x2: 1125, y2: 275, class: "dataflow-divider"}));
    text(svg, 25, 293, "Static SASS sites (possibly inactive branches)", "dataflow-section-label");
  }
  graph.edges.forEach(edge => {
    const from = positions.get(edge.source), to = positions.get(edge.target);
    if (!from || !to) return;
    const selected = edge.id === selectedId;
    const group = svgElement("g", {
      class: `dataflow-edge${selected ? " selected" : ""}${edge.operation.startsWith("logical_") ? " logical" : ""}`, tabindex: "0", role: "button",
      "aria-label": `${edge.operation}: ${edge.source} to ${edge.target}; ${dataflowEdgeLabel(graph, edge)}`,
    });
    const x1 = from.x + 165, y1 = from.y + 27, x2 = to.x, y2 = to.y + 27;
    group.appendChild(svgElement("path", {d: `M${x1},${y1} C${(x1+x2)/2},${y1} ${(x1+x2)/2},${y2} ${x2},${y2}`, class: "dataflow-edge-line", "marker-end": `url(#${markerId})`}));
    group.appendChild(svgElement("path", {d: `M${x1},${y1} C${(x1+x2)/2},${y1} ${(x1+x2)/2},${y2} ${x2},${y2}`, class: "dataflow-edge-hit"}));
    const labelY = Math.abs(y1 - y2) < 5 ? y1 + 52 : (y1 + y2) / 2 - 9;
    text(group, (x1 + x2) / 2, labelY, dataflowEdgeLabel(graph, edge), "dataflow-edge-value");
    group.addEventListener("click", () => choose(edge.id));
    group.addEventListener("keydown", event => {
      if (event.key === "Enter" || event.key === " ") { event.preventDefault(); choose(edge.id); }
    });
    svg.appendChild(group);
  });
  graph.nodes.forEach(node => {
    const position = positions.get(node.id);
    const group = svgElement("g", {class: `dataflow-node dataflow-${node.memory_space}`});
    group.appendChild(svgElement("rect", {x: position.x, y: position.y, width: 165, height: 55, rx: 7}));
    text(group, position.x + 8, position.y + 22, node.label, "dataflow-label");
    text(group, position.x + 8, position.y + 42, node.memory_space, "dataflow-sub");
    svg.appendChild(group);
  });
}

function renderDataflow(kernel) {
  const graph = kernel?.dataflow;
  $("dataflow-panel").hidden = false;
  $("dataflow-graph").replaceChildren();
  if (!graph) {
    $("dataflow-visual").hidden = true;
    $("dataflow-summary").textContent = kernel
      ? "No bounded dataflow model for this source and workload."
      : "Select a kernel to view its dataflow.";
    return;
  }
  $("dataflow-visual").hidden = false;
  $("dataflow-summary").textContent = dataflowSummary(kernel);
  drawDataflowDiagram($("dataflow-graph"), graph, state.dataflowSelection, edgeId => {
    state.dataflowSelection = edgeId;
    renderHardware();
  });
  const edge = graph.edges.find(row => row.id === state.dataflowSelection) || graph.edges[0];
  $("dataflow-detail").textContent = dataflowEdgeDetail(graph, edge);
}

function renderNodeDataflow(side, nodeId) {
  const panel = $(`node-${side}-dataflow`);
  const kernel = state.hardware?.kernels.find(row => row.node_id === nodeId);
  panel.hidden = !kernel;
  if (!kernel) return;
  const graph = kernel.dataflow;
  const svg = $(`node-${side}-dataflow-graph`);
  svg.replaceChildren();
  if (!graph) {
    $(`node-${side}-dataflow-summary`).textContent = "No dataflow graph for this node.";
    $(`node-${side}-dataflow-detail`).textContent = "";
    return;
  }
  $(`node-${side}-dataflow-summary`).textContent = dataflowSummary(kernel);
  const selectedId = state.comparisonDataflowSelection[side];
  drawDataflowDiagram(svg, graph, selectedId, edgeId => {
    state.comparisonDataflowSelection[side] = edgeId;
    renderNodeDataflow(side, nodeId);
  });
  const edge = graph.edges.find(row => row.id === selectedId) || graph.edges[0];
  $(`node-${side}-dataflow-detail`).textContent = dataflowEdgeDetail(graph, edge);
}
function renderHardwareConstraints(graph, kernel) {
  const checks = new Map((kernel?.constraint_checks || []).map(row => [row.constraint_id, row]));
  $("hardware-constraint-table").querySelector("tbody").innerHTML = (graph.constraints || []).map(row => {
    const limit = row.maximum || row.combined_maximum;
    const check = checks.get(row.id);
    const observed = check?.observed?.value == null ? "—" : hardwareValue(check.observed) +
      (check.observed.basis === "lower_bound" ? " (minimum)" : "");
    const status = {
      exceeds_architectural_limit: "Exceeds limit",
      within_architectural_limit: "Within maximum",
      lower_bound_within_limit: "Minimum within; total unknown",
      unknown: "Evidence missing",
    }[check?.status] || "Context only";
    const klass = check?.status === "exceeds_architectural_limit" ? "constraint-exceeds" : "";
    return `<tr class="${klass}"><td>${escapeHtml(row.id.replaceAll("_", " "))}</td>` +
      `<td>${escapeHtml(row.scope || "per SM")}</td><td>${escapeHtml(hardwareValue(limit))}</td>` +
      `<td>${escapeHtml(observed)}</td><td>${escapeHtml(status)}</td></tr>`;
  }).join("");
}
function activateHardware(element, type, id) {
  const choose = () => { state.hardwareSelection = {type, id}; renderHardware(); };
  element.addEventListener("click", choose);
  element.addEventListener("keydown", event => {
    if (event.key === "Enter" || event.key === " ") { event.preventDefault(); choose(); }
  });
}
function showHardwareDetail(graph, metricsByResource) {
  const selection = state.hardwareSelection;
  const detail = $("hardware-detail");
  if (selection.type === "node") {
    const resource = graph.nodes.find(row => row.id === selection.id);
    if (!resource) return;
    const metrics = metricsByResource.get(resource.id) || [];
    const facts = hardwareFacts(resource);
    (graph.constraints || []).filter(row => row.resource_id === resource.id).forEach(row => {
      facts.push(`${row.id.replaceAll("_", " ")}: max ${hardwareValue(row.maximum)} (${row.scope})` +
        (row.maximum?.source ? `\nsource: ${row.maximum.source}` : ""));
    });
    if (resource.id === "l1") facts.push("Capacity depends on the L1/shared carveout.");
    if (resource.id === "shared") facts.push("Shares a configurable 256 KiB pool with L1 per SM.");
    detail.textContent = `${hardwarePositions[resource.id]?.label || resource.id} · ${resource.scope}\n` +
      `${facts.join("\n") || "Capacity unknown"}\n` +
      (metrics.length ? `\nSelected kernel NCU metrics:\n${metrics.map(row => `${row.metric}: ${number(row.value, 2)}${row.unit}`).join("\n")}` : "\nNo NCU metric mapped to this resource for the selected kernel.");
  } else {
    const link = graph.links.find(row => row.id === selection.id);
    if (!link) return;
    detail.textContent = `${link.source} ${link.direction === "forward" ? "→" : "↔"} ${link.target} · ${link.pathway}\n` +
      `Published peak: ${hardwareValue(link.published_peak_bandwidth)}\n` +
      `Sustainable bandwidth: ${hardwareValue(link.sustainable_bandwidth)}\n` +
      `Latency: ${hardwareValue(link.latency)}\n` +
      (link.published_peak_bandwidth?.source ? `Peak source: ${link.published_peak_bandwidth.source}` : "No published peak recorded.");
  }
}

function loadAnalysis() {
  const analysis = state.graph.analysis || {timeline: [], strategies: []};
  const maximum = analysis.timeline.length
    ? Math.max(...analysis.timeline.map(item => item.iteration))
    : 0;
  state.iteration = maximum;
  $("iteration-slider").max = maximum;
  $("iteration-slider").value = maximum;
  $("iteration-value").textContent = maximum;
  showIteration();
  drawTimeline();
  $("strategy-table").querySelector("tbody").innerHTML = analysis.strategies.map(row => {
    const callRate = row.calls ? `${row.valid_calls}/${row.calls}` : "—";
    const proposalRate = row.completed_proposals ? `${row.valid_outcomes}/${row.completed_proposals}` : "—";
    return `<tr><td>${escapeHtml(row.strategy_id)}</td><td>${row.visits ?? 0}</td>` +
      `<td>${row.calls ?? 0}</td><td>${row.repair_calls ?? 0}</td><td>${callRate}</td>` +
      `<td>${proposalRate}</td><td>${number(row.mean_valid_reward)}</td>` +
      `<td>${number(row.max_valid_reward)}</td></tr>`;
  }).join("");
}

function showIteration() {
  const timeline = state.graph?.analysis?.timeline || [];
  const point = timeline.find(item => item.iteration === state.iteration);
  $("iteration-value").textContent = state.iteration;
  if (!point) {
    $("iteration-summary").textContent = state.iteration === 0
      ? "Root state before the first iteration."
      : "No completed iteration record.";
  } else {
    $("iteration-summary").textContent = `status ${point.status} · B_gen ${point.b_gen} · B_mut ${point.b_mut || 0} · ` +
      `strategy ${point.selected_strategy_id || "—"} · backed-up reward ${number(point.backed_up_reward)} · ` +
      `best ${number(point.cumulative_best_speedup, 3)}× (${short(point.cumulative_best_node_id)})`;
    const decisions = state.graph.selection_decisions || [];
    const decisionIndex = decisions.findIndex(item => item.iteration === state.iteration);
    if (decisionIndex >= 0) {
      $("decision-select").value = String(decisionIndex);
      showDecision(decisionIndex);
    }
  }
  renderGraph();
  drawTimeline();
}

function drawTimeline() {
  const svg = $("reward-timeline");
  const timeline = state.graph?.analysis?.timeline || [];
  svg.innerHTML = "";
  if (!timeline.length) return;
  const width = 600, height = 170, left = 42, right = 12, top = 12, bottom = 28;
  svg.setAttribute("viewBox", `0 0 ${width} ${height}`);
  const maximumX = Math.max(1, ...timeline.map(item => item.b_gen));
  const rewards = [0, ...timeline.map(item => item.cumulative_best_reward)];
  const minimumY = Math.min(...rewards), maximumY = Math.max(...rewards);
  const spanY = Math.max(0.1, maximumY - minimumY);
  const x = value => left + value / maximumX * (width - left - right);
  const y = value => top + (maximumY - value) / spanY * (height - top - bottom);
  svg.appendChild(svgElement("path", {d: `M${left},${top} V${height-bottom} H${width-right}`, class: "timeline-axis"}));
  const path = timeline.map((item, index) => `${index ? "L" : "M"}${x(item.b_gen)},${y(item.cumulative_best_reward)}`).join(" ");
  svg.appendChild(svgElement("path", {d: path, class: "timeline-line"}));
  timeline.forEach(item => {
    const point = svgElement("circle", {cx: x(item.b_gen), cy: y(item.cumulative_best_reward), r: 4, class: "timeline-point"});
    point.addEventListener("click", () => setIteration(item.iteration));
    hover(point, `iteration ${item.iteration}\nB_gen ${item.b_gen}\nbest reward ${number(item.cumulative_best_reward)}\n${number(item.cumulative_best_speedup, 3)}×`);
    svg.appendChild(point);
  });
  const current = timeline.find(item => item.iteration === state.iteration);
  if (current) svg.appendChild(svgElement("line", {x1: x(current.b_gen), x2: x(current.b_gen), y1: top, y2: height-bottom, class: "timeline-current"}));
  const xLabel = svgElement("text", {x: width-right, y: height-8, "text-anchor": "end", class: "graph-label graph-sub"});
  xLabel.textContent = `B_gen ${maximumX}`; svg.appendChild(xLabel);
  const yLabel = svgElement("text", {x: 5, y: top+8, class: "graph-label graph-sub"});
  yLabel.textContent = `r ${number(maximumY, 2)}`; svg.appendChild(yLabel);
}

function setIteration(value) {
  state.iteration = Number(value);
  $("iteration-slider").value = state.iteration;
  showIteration();
}

function togglePlayback() {
  if (state.playTimer) {
    clearInterval(state.playTimer); state.playTimer = null; $("play-iterations").textContent = "Play"; return;
  }
  const maximum = Number($("iteration-slider").max);
  if (!maximum) return;
  if (state.iteration >= maximum) setIteration(0);
  $("play-iterations").textContent = "Pause";
  state.playTimer = setInterval(() => {
    if (state.iteration >= maximum) { togglePlayback(); return; }
    setIteration(state.iteration + 1);
  }, 700);
}

async function compareRuns() {
  const traceB = $("compare-trace").value;
  const runB = $("compare-run").value;
  if (!state.trace || !state.run || !traceB || !runB) return;
  try {
    const value = await api(`/api/compare-runs?${query({
      trace_a: state.trace.trace_id, run_a: state.run, trace_b: traceB, run_b: runB,
    })}`);
    const a = value.run_a, b = value.run_b;
    const warnings = [];
    if (!value.comparable_workload) warnings.push("workloads differ");
    if (!value.comparable_hardware) warnings.push("hardware differs");
    const warning = warnings.length ? ` Warning: ${warnings.join(" and ")}.` : "";
    $("run-comparison-summary").textContent = `A: ${short(a.run_id)} · best ${number(a.best_speedup, 3)}× · ` +
      `B_gen ${a.b_gen}. B: ${short(b.run_id)} · best ${number(b.best_speedup, 3)}× · ` +
      `B_gen ${b.b_gen}. Δbest=${number(value.summary_deltas.best_speedup, 3)}×.${warning}`;
    drawRunComparison(value.timeline_a, value.timeline_b);
    $("run-difference-table").querySelector("tbody").innerHTML = value.differences.map(row =>
      `<tr><td>${escapeHtml(row.field)}</td><td>${escapeHtml(formatValue(row.a))}</td>` +
      `<td>${escapeHtml(formatValue(row.b))}</td></tr>`
    ).join("") || '<tr><td colspan="3">No recorded differences.</td></tr>';
    $("best-profile-table").querySelector("tbody").innerHTML = value.best_profile_comparison.map(row =>
      `<tr><td>${escapeHtml(row.metric)}</td><td>${number(row.a)}</td><td>${number(row.b)}</td>` +
      `<td>${number(row.delta)}</td><td>${number(row.percent_change, 2)}</td></tr>`
    ).join("") || '<tr><td colspan="5">One or both best nodes were not profiled.</td></tr>';
    $("run-strategy-table").querySelector("tbody").innerHTML = value.strategies.map(row =>
      `<tr><td>${escapeHtml(row.strategy_id)}</td>` +
      `<td>${pair(row.visits_a, row.visits_b)}</td><td>${pair(row.calls_a, row.calls_b)}</td>` +
      `<td>${pair(row.repair_calls_a, row.repair_calls_b)}</td>` +
      `<td>${ratioPair(row.valid_calls_a, row.calls_a, row.valid_calls_b, row.calls_b)}</td>` +
      `<td>${ratioPair(row.valid_outcomes_a, row.completed_proposals_a, row.valid_outcomes_b, row.completed_proposals_b)}</td>` +
      `<td>${number(row.max_valid_reward_a)} / ${number(row.max_valid_reward_b)}</td></tr>`
    ).join("");
  } catch (error) {
    showError(error);
  }
}

function pair(a, b) { return `${a ?? 0} / ${b ?? 0}`; }
function ratioPair(va, ta, vb, tb) {
  const left = ta ? `${va}/${ta}` : "—";
  const right = tb ? `${vb}/${tb}` : "—";
  return `${left} / ${right}`;
}
function formatValue(value) {
  if (value == null) return "—";
  return typeof value === "object" ? JSON.stringify(value) : String(value);
}
function drawRunComparison(timelineA, timelineB) {
  const svg = $("run-comparison-chart");
  svg.innerHTML = "";
  if (!timelineA.length && !timelineB.length) return;
  const width = 900, height = 190, left = 55, right = 15, top = 14, bottom = 32;
  svg.setAttribute("viewBox", `0 0 ${width} ${height}`);
  const seriesA = [{b_gen: 0, cumulative_best_reward: 0}, ...timelineA];
  const seriesB = [{b_gen: 0, cumulative_best_reward: 0}, ...timelineB];
  const maximumX = Math.max(1, ...seriesA.map(item => item.b_gen), ...seriesB.map(item => item.b_gen));
  const rewards = [...seriesA, ...seriesB].map(item => item.cumulative_best_reward);
  const minimumY = Math.min(...rewards), maximumY = Math.max(...rewards);
  const plotMaximumY = maximumY - minimumY < 0.1 ? minimumY + 0.1 : maximumY;
  const spanY = plotMaximumY - minimumY;
  const x = value => left + value / maximumX * (width-left-right);
  const y = value => top + (plotMaximumY-value) / spanY * (height-top-bottom);
  svg.appendChild(svgElement("path", {d: `M${left},${top} V${height-bottom} H${width-right}`, class: "timeline-axis"}));
  const xTicks = [...new Set(Array.from({length: 6}, (_, index) => Math.round(maximumX * index / 5)))];
  xTicks.forEach(value => {
    svg.appendChild(svgElement("line", {x1: x(value), x2: x(value), y1: height-bottom, y2: height-bottom+5, class: "timeline-tick"}));
    const label = svgElement("text", {x: x(value), y: height-10, "text-anchor": "middle", class: "graph-label graph-sub"});
    label.textContent = value; svg.appendChild(label);
  });
  Array.from({length: 6}, (_, index) => minimumY + spanY * index / 5).forEach(value => {
    svg.appendChild(svgElement("line", {x1: left-5, x2: left, y1: y(value), y2: y(value), class: "timeline-tick"}));
    const label = svgElement("text", {x: left-8, y: y(value)+3, "text-anchor": "end", class: "graph-label graph-sub"});
    label.textContent = number(value, 2); svg.appendChild(label);
  });
  const addSeries = (series, klass, label, labelY, labelClass) => {
    const path = series.map((item,index) => `${index ? "L" : "M"}${x(item.b_gen)},${y(item.cumulative_best_reward)}`).join(" ");
    svg.appendChild(svgElement("path", {d: path, class: klass}));
    const textNode = svgElement("text", {x: width-80, y: labelY, class: `graph-label ${labelClass}`});
    textNode.textContent = label; svg.appendChild(textNode);
  };
  addSeries(seriesA, "timeline-line", "A", 22, "timeline-label-a");
  addSeries(seriesB, "timeline-line-b", "B", 40, "timeline-label-b");
  const xLabel = svgElement("text", {x: width-right, y: height-10, "text-anchor": "end", class: "graph-label graph-sub"});
  xLabel.textContent = "B_gen"; svg.appendChild(xLabel);
  const yLabel = svgElement("text", {x: 6, y: top+8, class: "graph-label graph-sub"});
  yLabel.textContent = "reward"; svg.appendChild(yLabel);
}

function loadDecisions() {
  const decisions = state.graph.selection_decisions || [];
  $("decision-select").innerHTML = decisions.map((decision, index) =>
    `<option value="${index}">iteration ${decision.iteration}, step ${decision.step} · ` +
    `${escapeHtml(short(decision.node_id))} → ${escapeHtml(decision.selected_strategy_id)}</option>`
  ).join("");
  if (decisions.length) showDecision(0);
  else {
    $("decision-summary").textContent = "This trace has no exact decision snapshots.";
    $("puct-table").querySelector("tbody").innerHTML = "";
    $("ucb-table").querySelector("tbody").innerHTML = "";
  }
}

function showDecision(index) {
  const decision = state.graph.selection_decisions[Number(index)];
  if (!decision) return;
  const widening = `${decision.existing_children}/${decision.allowed_children} existing/allowed`;
  $("decision-summary").textContent = `node ${short(decision.node_id)} · selected ` +
    `${decision.selected_strategy_id} · ${decision.selection_mode} · total action visits ` +
    `${decision.total_action_visits} · progressive widening ${widening} · ` +
    `c_puct=${decision.c_puct}, c_ucb=${decision.c_ucb}`;
  $("puct-table").querySelector("tbody").innerHTML = decision.puct_candidates.map(row =>
    `<tr class="${row.selected ? "selected-row" : ""}"><td>${escapeHtml(row.strategy_id)}</td>` +
    `<td>${number(row.prior, 3)}</td><td>${row.visits}</td><td>${number(row.exploit_term)}</td>` +
    `<td>${number(row.explore_term)}</td><td>${number(row.total_score)}</td></tr>`
  ).join("");
  $("ucb-table").querySelector("tbody").innerHTML = decision.ucb_candidates.map(row =>
    `<tr class="${row.selected ? "selected-row" : ""}"><td>${escapeHtml(short(row.child_node_id))}</td>` +
    `<td>${row.descents}</td><td>${number(row.exploit_term)}</td>` +
    `<td>${number(row.explore_term)}</td><td>${number(row.total_score)}</td></tr>`
  ).join("");
}

function clearSelections() {
  state.comparisonDataflowSelection = {a: null, b: null};
  for (const side of ["a", "b"]) {
    $(`node-${side}-title`).textContent = side === "a" ? "not selected" : "Shift-click a node";
    $(`node-${side}-meta`).textContent = "";
    $(`node-${side}-source`).textContent = "";
    $(`node-${side}-profile`).textContent = "";
    $(`node-${side}-dataflow`).hidden = true;
    $(`node-${side}-dataflow-graph`).replaceChildren();
  }
  $("relationship").textContent = "Select two nodes.";
  $("relationship").className = "relationship empty";
  $("source-diff").textContent = "Select two nodes.";
  $("profile-table").querySelector("tbody").innerHTML = "";
  $("export-bundle").disabled = true;
  $("export-status").textContent = "";
}

function visibleGraph() {
  const maxDepth = $("max-depth").value === "" ? Infinity : Number($("max-depth").value);
  const minVisits = Number($("min-visits").value || 0);
  const nodes = state.graph.nodes.filter(node =>
    (node.depth == null || node.depth <= maxDepth) &&
    (node.created_iteration == null || node.created_iteration <= state.iteration)
  );
  const nodeIds = new Set(nodes.map(node => node.node_id));
  const realizations = state.graph.realizations.filter(edge =>
    nodeIds.has(edge.parent_node_id) && nodeIds.has(edge.child_node_id) && edge.descents >= minVisits &&
    (edge.created_iteration == null || edge.created_iteration <= state.iteration)
  );
  const keys = new Set(realizations.map(edge => `${edge.parent_node_id}\u0000${edge.strategy_id}`));
  const failures = $("show-failures").checked && minVisits === 0
    ? state.graph.failures.filter(item =>
      nodeIds.has(item.parent_node_id) && (item.iteration == null || item.iteration <= state.iteration)
    )
    : [];
  failures.forEach(item => keys.add(`${item.parent_node_id}\u0000${item.strategy_id}`));
  const strategies = state.graph.strategies.filter(item =>
    keys.has(`${item.parent_node_id}\u0000${item.strategy_id}`)
  );
  return {nodes, realizations, strategies, failures};
}

function renderGraph() {
  if (!state.graph) return;
  const data = visibleGraph();
  const viewport = $("viewport");
  viewport.innerHTML = "";
  const defs = svgElement("defs");
  defs.innerHTML = '<marker id="arrow" markerWidth="8" markerHeight="8" refX="7" refY="3" orient="auto"><path d="M0,0 L0,6 L8,3 z" fill="#52606d"/></marker>';
  viewport.appendChild(defs);

  const byDepth = new Map();
  data.nodes.forEach(node => {
    const depth = node.depth ?? 0;
    if (!byDepth.has(depth)) byDepth.set(depth, []);
    byDepth.get(depth).push(node);
  });
  const positions = new Map();
  [...byDepth.entries()].sort((a, b) => a[0] - b[0]).forEach(([depth, nodes]) =>
    nodes.sort((a, b) => b.reward - a.reward).forEach((node, index) =>
      positions.set(node.node_id, {x: depth * 390, y: index * 130})
    )
  );
  const strategyPositions = new Map();
  data.strategies.forEach(strategy => {
    const parent = positions.get(strategy.parent_node_id);
    if (!parent) return;
    const siblings = data.strategies.filter(item => item.parent_node_id === strategy.parent_node_id);
    const offset = siblings.indexOf(strategy) - (siblings.length - 1) / 2;
    strategyPositions.set(`${strategy.parent_node_id}\u0000${strategy.strategy_id}`, {
      x: parent.x + 190, y: parent.y + offset * 76,
    });
  });
  const failurePositions = new Map();
  data.failures.forEach((failure, index) => {
    const parent = strategyPositions.get(`${failure.parent_node_id}\u0000${failure.strategy_id}`);
    if (parent) failurePositions.set(failure.generation_id, {x: parent.x + 170, y: parent.y + 65 + index * 42});
  });

  data.strategies.forEach(strategy => {
    const parent = positions.get(strategy.parent_node_id);
    const child = strategyPositions.get(`${strategy.parent_node_id}\u0000${strategy.strategy_id}`);
    if (parent && child) drawEdge(parent.x + 150, parent.y + 40, child.x - 10, child.y + 35, "graph-edge");
  });
  data.realizations.forEach(edge => {
    const parent = strategyPositions.get(`${edge.parent_node_id}\u0000${edge.strategy_id}`);
    const child = positions.get(edge.child_node_id);
    if (parent && child) drawEdge(parent.x + 110, parent.y + 35, child.x, child.y + 40, "graph-edge realization");
  });
  data.failures.forEach(failure => {
    const parent = strategyPositions.get(`${failure.parent_node_id}\u0000${failure.strategy_id}`);
    const child = failurePositions.get(failure.generation_id);
    if (parent && child) drawEdge(parent.x + 110, parent.y + 35, child.x, child.y + 22, "graph-edge failure");
  });
  data.nodes.forEach(node => drawNode(node, positions.get(node.node_id)));
  data.strategies.forEach(item => drawStrategy(item, strategyPositions.get(`${item.parent_node_id}\u0000${item.strategy_id}`)));
  data.failures.forEach(item => drawFailure(item, failurePositions.get(item.generation_id)));
  applyTransform();
}

function svgElement(name, attributes = {}) {
  const element = document.createElementNS("http://www.w3.org/2000/svg", name);
  Object.entries(attributes).forEach(([key, value]) => element.setAttribute(key, value));
  return element;
}
function drawEdge(x1, y1, x2, y2, klass) {
  const bend = (x1 + x2) / 2;
  $("viewport").appendChild(svgElement("path", {
    d: `M${x1},${y1} C${bend},${y1} ${bend},${y2} ${x2},${y2}`, class: klass,
  }));
}
function text(group, x, y, value, klass = "graph-label") {
  const node = svgElement("text", {x, y, class: klass});
  node.textContent = value;
  group.appendChild(node);
}
function drawNode(node, position) {
  if (!position) return;
  const group = svgElement("g");
  let klass = "node-box";
  if (node.node_id === state.graph.root_node_id) klass += " node-root";
  if (node.node_id === state.graph.best_node_id) klass += " node-best";
  if (state.a?.node_id === node.node_id) klass += " node-selected-a";
  if (state.b?.node_id === node.node_id) klass += " node-selected-b";
  const box = svgElement("rect", {x: position.x, y: position.y, width: 150, height: 80, rx: 7, class: klass});
  box.addEventListener("click", event => selectNode(node.node_id, event.shiftKey));
  hover(box, `node ${node.node_id}\nreward ${number(node.reward)} · ${number(node.speedup, 3)}×\n` +
    `median ${number(node.median_us, 3)} us\n${node.profiled ? "profiled" : "not profiled"}`);
  group.appendChild(box);
  text(group, position.x + 10, position.y + 22, `node ${short(node.node_id)}`);
  text(group, position.x + 10, position.y + 43, `${number(node.speedup, 3)}× · r=${number(node.reward, 3)}`, "graph-label graph-sub");
  text(group, position.x + 10, position.y + 63, `${number(node.median_us, 2)} us · visits ${node.action_visits}`, "graph-label graph-sub");
  $("viewport").appendChild(group);
}
function drawStrategy(item, position) {
  if (!position) return;
  const group = svgElement("g");
  const box = svgElement("rect", {x: position.x, y: position.y, width: 110, height: 70, rx: 18, class: "node-box strategy-box"});
  hover(box, `${item.strategy_id}\nprior ${number(item.prior, 3)} · visits ${item.visits}\n` +
    `final Q_mean ${number(item.q_mean)} · Q_max ${number(item.q_max)}\n` +
    `proposals ${item.proposal_count} (${item.valid_proposal_count} valid, ${item.invalid_proposal_count} invalid)`);
  group.appendChild(box);
  text(group, position.x + 8, position.y + 22, short(item.strategy_id));
  text(group, position.x + 8, position.y + 42, `Q=${number(item.q_mean, 3)} N=${item.visits}`, "graph-label graph-sub");
  text(group, position.x + 8, position.y + 58, `P=${number(item.prior, 3)}`, "graph-label graph-sub");
  $("viewport").appendChild(group);
}
function drawFailure(item, position) {
  if (!position) return;
  const group = svgElement("g");
  const box = svgElement("rect", {x: position.x, y: position.y, width: 135, height: 44, rx: 5, class: "node-box failure-box"});
  hover(box, `B_gen ${item.b_gen} · ${item.proposal_status}\n${item.invalid_reason || "unknown reason"}\n` +
    `compile ${item.compile_status} · correctness ${item.correctness_status}`);
  group.appendChild(box);
  text(group, position.x + 7, position.y + 18, `invalid B${item.b_gen}`);
  text(group, position.x + 7, position.y + 34, short(item.invalid_reason || "unknown"), "graph-label graph-sub");
  $("viewport").appendChild(group);
}

async function selectNode(nodeId, selectB) {
  try {
    const detail = await api(`/api/node?${query({trace: state.trace.trace_id, run: state.run, node: nodeId})}`);
    if (selectB) state.b = detail;
    else {
      state.a = detail;
      if (state.hardware?.kernels.some(row => row.node_id === nodeId)) {
        $("hardware-node").value = nodeId;
        renderHardware();
      }
    }
    showNode(selectB ? "b" : "a", detail);
    renderNodeDataflow(selectB ? "b" : "a", detail.node_id);
    renderGraph();
    if (state.a && state.b) {
      $("export-bundle").disabled = false;
      await compare();
    }
  } catch (error) {
    showError(error);
  }
}
function showNode(side, node) {
  $(`node-${side}-title`).textContent = short(node.node_id);
  $(`node-${side}-meta`).innerHTML = `reward <b>${number(node.reward)}</b> · speedup ` +
    `<b>${number(node.speedup, 3)}×</b> · median <b>${number(node.benchmark?.median_us, 3)} us</b> · ` +
    `backend ${escapeHtml(node.backend_type)}<br>launch ${escapeHtml(JSON.stringify(node.launch_config))}`;
  $(`node-${side}-source`).textContent = node.program_text;
  $(`node-${side}-profile`).textContent = node.profile ? JSON.stringify(node.profile, null, 2) : "Not profiled";
}
async function compare() {
  const value = await api(`/api/compare?${query({
    trace: state.trace.trace_id, run: state.run, a: state.a.node_id, b: state.b.node_id,
  })}`);
  showRelationship(value.relationship);
  showDiff(value.source_diff);
  $("profile-table").querySelector("tbody").innerHTML = value.profile_comparison.map(row =>
    `<tr><td>${escapeHtml(row.metric)}</td><td>${number(row.a)}</td><td>${number(row.b)}</td>` +
    `<td>${number(row.delta)}</td><td>${number(row.percent_change, 2)}</td></tr>`
  ).join("");
}
function hops(path) {
  return (path || []).map(edge =>
    `<div class="hop"><b>${escapeHtml(short(edge.parent_node_id))}</b> → ` +
    `${escapeHtml(edge.strategy_id)} → <b>${escapeHtml(short(edge.child_node_id))}</b><br>` +
    `<span class="hint">final descents ${edge.descents}; Q ${number(edge.q_mean)}</span></div>`
  ).join("") || '<div class="hop">same node</div>';
}
function showRelationship(relationship) {
  const element = $("relationship");
  element.className = "relationship";
  if (relationship.kind === "A_ANCESTOR_OF_B") {
    element.innerHTML = `<p>A is an ancestor of B (shortest path).</p>${hops(relationship.path)}`;
  } else if (relationship.kind === "B_ANCESTOR_OF_A") {
    element.innerHTML = `<p>B is an ancestor of A (shortest path shown B → A).</p>${hops(relationship.path)}`;
  } else if (relationship.kind === "COMMON_ANCESTOR") {
    element.innerHTML = `<p>Nearest common ancestor: <b>${escapeHtml(short(relationship.common_ancestor))}</b></p>` +
      `<h3>ancestor → A</h3>${hops(relationship.path_to_a)}` +
      `<h3>ancestor → B</h3>${hops(relationship.path_to_b)}`;
  } else {
    element.textContent = "The nodes are unrelated in the persisted graph.";
  }
}
function showDiff(diff) {
  $("source-diff").innerHTML = (diff || "Sources are identical.").split("\n").map(line => {
    const klass = line.startsWith("+") && !line.startsWith("+++")
      ? "diff-add" : line.startsWith("-") && !line.startsWith("---") ? "diff-remove" : "";
    return `<span class="${klass}">${escapeHtml(line)}</span>`;
  }).join("\n");
}

async function exportBundle() {
  if (!state.a || !state.b) return;
  const button = $("export-bundle");
  button.disabled = true;
  $("export-status").textContent = "Building analysis bundle…";
  try {
    const response = await fetch(`/api/export?${query({
      trace: state.trace.trace_id, run: state.run, a: state.a.node_id, b: state.b.node_id,
    })}`);
    if (!response.ok) {
      const error = await response.json();
      throw new Error(error.error || `HTTP ${response.status}`);
    }
    const disposition = response.headers.get("Content-Disposition") || "";
    const match = disposition.match(/filename="([^"]+)"/);
    const filename = match ? match[1] : "mcts-analysis.zip";
    const url = URL.createObjectURL(await response.blob());
    const link = document.createElement("a");
    link.href = url; link.download = filename; document.body.appendChild(link); link.click(); link.remove();
    URL.revokeObjectURL(url);
    $("export-status").textContent = `Downloaded ${filename}`;
  } catch (error) {
    $("export-status").textContent = error.message;
  } finally {
    button.disabled = false;
  }
}

function hover(element, message) {
  const tooltip = $("tooltip");
  element.addEventListener("mousemove", event => {
    tooltip.hidden = false;
    tooltip.textContent = message;
    tooltip.style.left = `${event.clientX + 12}px`;
    tooltip.style.top = `${event.clientY + 12}px`;
  });
  element.addEventListener("mouseleave", () => { tooltip.hidden = true; });
}
function applyTransform() {
  $("viewport").setAttribute("transform", `translate(${state.x} ${state.y}) scale(${state.scale})`);
}
function fitGraph() {
  state.scale = 0.8;
  state.x = 30;
  state.y = 35;
  applyTransform();
}
function installPanZoom() {
  const svg = $("graph");
  let drag = null;
  svg.addEventListener("wheel", event => {
    event.preventDefault();
    state.scale = Math.max(0.2, Math.min(2.5, state.scale * (event.deltaY < 0 ? 1.1 : 0.9)));
    applyTransform();
  }, {passive: false});
  svg.addEventListener("mousedown", event => {
    drag = {x: event.clientX, y: event.clientY, startX: state.x, startY: state.y};
    svg.classList.add("dragging");
  });
  window.addEventListener("mousemove", event => {
    if (!drag) return;
    state.x = drag.startX + event.clientX - drag.x;
    state.y = drag.startY + event.clientY - drag.y;
    applyTransform();
  });
  window.addEventListener("mouseup", () => {
    drag = null;
    svg.classList.remove("dragging");
  });
}
function showError(error) {
  $("run-summary").innerHTML = `<span class="error">${escapeHtml(error.message)}</span>`;
}

$("trace-select").addEventListener("change", chooseTrace);
$("compare-trace").addEventListener("change", chooseCompareTrace);
$("run-select").addEventListener("change", loadRun);
$("reload").addEventListener("click", loadCatalog);
$("fit-graph").addEventListener("click", fitGraph);
$("decision-select").addEventListener("change", event => showDecision(event.target.value));
$("iteration-slider").addEventListener("input", event => setIteration(event.target.value));
$("play-iterations").addEventListener("click", togglePlayback);
$("compare-runs").addEventListener("click", compareRuns);
$("export-bundle").addEventListener("click", exportBundle);
$("hardware-node").addEventListener("change", renderHardware);
["max-depth", "min-visits", "show-failures"].forEach(id => $(id).addEventListener("change", renderGraph));
installPanZoom();
loadCatalog().catch(showError);
