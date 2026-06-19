function stepTypeBadge(type) {
  const color = STEP_TYPE_COLORS[type] || "var(--text3)";
  return `<span class="badge" style="background:transparent;border:1px solid ${color};color:${color};">${type}</span>`;
}

function stepSummary(step) {
  const type = step.type || "?";
  const parts = [];
  if (step.is_output) parts.push("output");
  if (step.required === false) parts.push("optional");
  if (step.only_if_route) {
    const r = Array.isArray(step.only_if_route)
      ? step.only_if_route.join("|")
      : step.only_if_route;
    parts.push(`route: ${r}`);
  }
  switch (type) {
    case "router_match":
      parts.push(
        `${(step.rules || []).length} rules -> default: ${step.default || "?"}`,
      );
      break;
    case "router_llm":
    case "llm_extract":
    case "llm_decide":
    case "llm_summarize":
    case "llm_analyze":
      if (step.prompt)
        parts.push(
          step.prompt.slice(0, 80) + (step.prompt.length > 80 ? "..." : ""),
        );
      if (step.search_query) parts.push(`search: ${step.search_query}`);
      break;
    case "web_search":
      parts.push(`query: ${step.query_template || "?"}`);
      if (step.time_range) parts.push(step.time_range);
      break;
    case "finance":
      parts.push(`ticker_key: ${step.ticker_key || "selected_ticker"}`);
      break;
    case "finance_search":
      parts.push(`query_key: ${step.query_key || "selected_isin"}`);
      break;
    case "finance_search":
      parts.push(`query_key: ${step.query_key || "selected_isin"}`);
      break;
    case "http_fetch":
      parts.push(step.url || step.url_template || "?");
      if (step.method && step.method !== "GET") parts.push(step.method);
      break;
    case "xlsx_fetch": {
      parts.push(step.url || step.url_template || "?");
      if (step.sheet !== undefined && step.sheet !== null)
        parts.push(`sheet: ${step.sheet}`);
      if (step.columns && step.columns.length)
        parts.push(
          `cols: ${step.columns.slice(0, 3).join(", ")}${step.columns.length > 3 ? "…" : ""}`,
        );
      if (step.filter)
        parts.push(`filter: ${step.filter.column}=${step.filter.value}`);
      break;
    }
    case "state_read":
    case "state_write":
      parts.push(`key: ${step.key || "?"}`);
      if (step.source_key) parts.push(`<- ${step.source_key}`);
      break;
    case "state_read_external":
    case "state_write_external":
    case "data_read_external":
    case "data_write_external":
      parts.push(`agent: ${step.agent_name || "?"}`);
      if (step.namespace) parts.push(`ns: ${step.namespace}`);
      if (step.key) parts.push(`key: ${step.key}`);
      break;
    case "data_read":
    case "data_write":
      parts.push(`${step.namespace || "?"}/${step.key_template || "?"}`);
      if (step.source_key) parts.push(`<- ${step.source_key}`);
      break;
    case "transform": {
      const op = step.operation || "?";
      parts.push(`op: ${op}`);
      if (step.source_key) parts.push(`<- ${step.source_key}`);
      if (step.field) parts.push(`field: ${step.field}`);
      if (step.group_field) parts.push(`group: ${step.group_field}`);
      if (step.group_key) parts.push(`group: ${step.group_key}`);
      if (step.subtract_key) parts.push(`- ${step.subtract_key}`);
      if (step.other_key) parts.push(`∪ ${step.other_key}`);
      if (step.target_key) parts.push(`-> ${step.target_key}`);
      break;
    }
    case "trigger_agent":
      parts.push(`-> ${step.target_agent_name || "?"}`);
      break;
    case "notify_user":
      if (step.source_key) parts.push(`<- ${step.source_key}`);
      break;
  }
  if (step.output_key) parts.push(`out: ${step.output_key}`);
  return parts.join(" . ");
}

function stepTypeOptions(selected) {
  let html = "";
  for (const [group, types] of Object.entries(STEP_TYPE_GROUPS)) {
    html += `<optgroup label="${group}">`;
    for (const t of types)
      html += `<option value="${t}" ${t === selected ? "selected" : ""}>${t}</option>`;
    html += "</optgroup>";
  }
  return html;
}

function timeRangeOptions(selected) {
  return ["", "day", "week", "month", "year"]
    .map(
      (t) =>
        `<option value="${t}" ${t === (selected || "") ? "selected" : ""}>${t || "-- kein Filter --"}</option>`,
    )
    .join("");
}
function categoryOptions(selected) {
  return ["", "general", "news", "finance", "it", "science", "social media"]
    .map(
      (c) =>
        `<option value="${c}" ${c === (selected || "") ? "selected" : ""}>${c || "-- general --"}</option>`,
    )
    .join("");
}
function transformOpOptions(selected) {
  return [
    "array_push",
    "map_field",
    "filter",
    "first",
    "slice",
    "diff",
    "intersect",
    "union",
    "list_append",
    "count",
    "group_by",
    "flatten",
    "sort",
    "statistics",
    "json_path",
    "xml_extract",
    "regex_extract",
    "arithmetic",
    "compare",
  ]
    .map(
      (o) =>
        `<option value="${o}" ${o === (selected || "") ? "selected" : ""}>${o}</option>`,
    )
    .join("");
}

function contextDropdown(id, value, placeholder) {
  const opts = _stepContextKeys
    .map(
      (k) =>
        `<option value="${k}" ${k === (value || "") ? "selected" : ""}>${k}</option>`,
    )
    .join("");
  return `<div style="position:relative;">
    <input class="modal-input" id="${id}" value="${(value || "").replace(/"/g, "&quot;")}" placeholder="${placeholder || ""}" list="${id}-list" autocomplete="off" />
    <datalist id="${id}-list">${opts}</datalist>
  </div>`;
}

function routeDropdown(id, value, placeholder) {
  const opts = _stepRouteKeys
    .map(
      (k) =>
        `<option value="${k}" ${k === (value || "") ? "selected" : ""}>${k}</option>`,
    )
    .join("");
  return `<div style="position:relative;">
    <input class="modal-input" id="${id}" value="${(value || "").replace(/"/g, "&quot;")}" placeholder="${placeholder || ""}" list="${id}-list" autocomplete="off" />
    <datalist id="${id}-list">${opts}</datalist>
  </div>`;
}

function field(label, inputHtml, id) {
  return `<div class="modal-field" id="${id || ""}"><div class="modal-label">${label}</div>${inputHtml}</div>`;
}
function textInput(id, value, placeholder) {
  return `<input class="modal-input" id="${id}" value="${String(value ?? "").replace(/"/g, "&quot;")}" placeholder="${placeholder || ""}" />`;
}
function textarea(id, value, minHeight) {
  return `<textarea class="modal-input" id="${id}" style="min-height:${minHeight || 120}px;font-family:var(--mono);font-size:12px;">${value || ""}</textarea>`;
}
function checkbox(id, checked, label) {
  return `<label style="display:flex;align-items:center;gap:6px;font-size:13px;color:var(--text2);cursor:pointer;"><input type="checkbox" id="${id}" ${checked ? "checked" : ""} /> ${label}</label>`;
}
