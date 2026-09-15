"use strict";

const state = {
  traces: [], trace: null, run: null, graph: null, a: null, b: null,
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
  const first = state.traces.find(item => !item.error && item.runs.length);
  if (!first) {
    $("run-summary").textContent = "No supported SQLite traces found.";
    return;
  }
  $("trace-select").value = first.trace_id;
  chooseTrace();
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
  try {
    state.graph = await api(`/api/graph?${query({trace: state.trace.trace_id, run: state.run})}`);
    const run = state.graph.run;
    $("run-summary").textContent = `${run.status} · ${run.benchmark_id} · model ` +
      `${run.model_name || "—"} · B_gen ${run.b_gen}/${run.generation_budget ?? "?"} · ` +
      `${run.node_count} nodes · best ${number(run.best_speedup, 3)}×`;
    $("score-notice").textContent = state.graph.historical_scores_available
      ? "Decision-time PUCT/UCB scores available."
      : "Edge values shown are final statistics. Existing traces do not contain exact historical PUCT/UCB candidate scores.";
    clearSelections();
    loadAnalysis();
    loadDecisions();
    renderGraph();
    fitGraph();
  } catch (error) {
    showError(error);
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
    $("iteration-summary").textContent = `status ${point.status} · B_gen ${point.b_gen} · ` +
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
  for (const side of ["a", "b"]) {
    $(`node-${side}-title`).textContent = side === "a" ? "not selected" : "Shift-click a node";
    $(`node-${side}-meta`).textContent = "";
    $(`node-${side}-source`).textContent = "";
    $(`node-${side}-profile`).textContent = "";
  }
  $("relationship").textContent = "Select two nodes.";
  $("relationship").className = "relationship empty";
  $("source-diff").textContent = "Select two nodes.";
  $("profile-table").querySelector("tbody").innerHTML = "";
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
    else state.a = detail;
    showNode(selectB ? "b" : "a", detail);
    renderGraph();
    if (state.a && state.b) await compare();
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
$("run-select").addEventListener("change", loadRun);
$("reload").addEventListener("click", loadCatalog);
$("fit-graph").addEventListener("click", fitGraph);
$("decision-select").addEventListener("change", event => showDecision(event.target.value));
$("iteration-slider").addEventListener("input", event => setIteration(event.target.value));
$("play-iterations").addEventListener("click", togglePlayback);
["max-depth", "min-visits", "show-failures"].forEach(id => $(id).addEventListener("change", renderGraph));
installPanZoom();
loadCatalog().catch(showError);
