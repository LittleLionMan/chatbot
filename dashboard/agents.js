function stepTypeBadge(type) {
  const color = STEP_TYPE_COLORS[type] || "var(--text3)";
  return `<span class="badge" style="background:transparent;border:1px solid ${color};color:${color};">${type}</span>`;
}

function stepSummary(step) {
  const type = step.type || "?";
  const parts = [];
  if (step.is_output) parts.push("output");
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
  return `<input class="modal-input" id="${id}" value="${(value || "").replace(/"/g, "&quot;")}" placeholder="${placeholder || ""}" />`;
}
function textarea(id, value, minHeight) {
  return `<textarea class="modal-input" id="${id}" style="min-height:${minHeight || 120}px;font-family:var(--mono);font-size:12px;">${value || ""}</textarea>`;
}
function checkbox(id, checked, label) {
  return `<label style="display:flex;align-items:center;gap:6px;font-size:13px;color:var(--text2);cursor:pointer;"><input type="checkbox" id="${id}" ${checked ? "checked" : ""} /> ${label}</label>`;
}

function buildStepFormBody(step) {
  const type = step.type || "llm_extract";
  const onlyIfRoute = Array.isArray(step.only_if_route)
    ? step.only_if_route.join(", ")
    : step.only_if_route || "";
  const onlyIfKeyKey = step.only_if_key?.key || "";
  const onlyIfKeyVal = step.only_if_key?.value ?? "";
  const commonTop = `
    ${field("ID", textInput("sf-id", step.id, "z.B. extract_gpu"))}
    ${field("Type", `<select class="modal-select" id="sf-type" onchange="onStepTypeChange()">${stepTypeOptions(type)}</select>`)}
    ${field("only_if_route", routeDropdown("sf-route", onlyIfRoute, "z.B. evaluate"))}
    <div class="modal-field">
      <div class="modal-label">only_if_key</div>
      <div style="display:grid;grid-template-columns:1fr 1fr;gap:8px;">
        <input class="modal-input" id="sf-only-if-key" value="${onlyIfKeyKey}" placeholder="Key z.B. decision.approved" oninput="document.getElementById('sf-only-if-val-wrap').style.display=this.value?'':'none'" />
        <div id="sf-only-if-val-wrap" style="display:${onlyIfKeyKey ? "" : "none"}">
          <input class="modal-input" id="sf-only-if-val" value="${onlyIfKeyVal}" placeholder="Wert z.B. true" />
        </div>
      </div>
    </div>
  `;
  const commonBottom = `<div class="modal-field" style="display:flex;gap:16px;">${checkbox("sf-output", step.is_output, "is_output")}</div>`;
  const outputKey = field(
    "output_key",
    textInput("sf-output-key", step.output_key, "z.B. extracted"),
  );
  const defaultVal = field(
    "default",
    textInput("sf-default", step.default, ""),
  );
  const sourceKey = field(
    "source_key (Context-Key: woher kommt der Wert)",
    contextDropdown("sf-source-key", step.source_key, "z.B. extracted_isins"),
  );
  const targetKey = field(
    "target_key",
    contextDropdown("sf-target-key", step.target_key, "z.B. baselines"),
  );
  const agentName = field(
    "agent_name",
    textInput("sf-agent-name", step.agent_name, "z.B. Gordon"),
  );
  const namespace = field(
    "namespace",
    textInput("sf-namespace", step.namespace, "z.B. analyses"),
  );
  const keyField = field(
    "key (State-Key: Name unter dem gespeichert wird)",
    textInput("sf-key", step.key, "z.B. known_isins"),
  );
  const keyTemplate = field(
    "key_template",
    textInput("sf-key-template", step.key_template, "z.B. {{ticker}}"),
  );
  const promptField = field("prompt", textarea("sf-prompt", step.prompt, 160));
  let typeSpecific = "";
  switch (type) {
    case "router_match":
      typeSpecific =
        field(
          "rules (JSON)",
          textarea("sf-rules", JSON.stringify(step.rules || [], null, 2), 140),
        ) +
        field(
          "default route",
          textInput("sf-default-route", step.default, "idle"),
        );
      break;
    case "router_llm":
      typeSpecific = promptField + outputKey;
      break;
    case "llm_extract":
    case "llm_decide":
    case "llm_analyze":
      return commonTop + promptField + outputKey + commonBottom;
    case "llm_summarize":
      return (
        commonTop +
        promptField +
        field(
          "search_query",
          textInput(
            "sf-search-query",
            step.search_query,
            "z.B. {{ticker}} news",
          ),
        ) +
        field(
          "time_range",
          `<select class="modal-select" id="sf-time-range">${timeRangeOptions(step.time_range)}</select>`,
        ) +
        field(
          "categories",
          `<select class="modal-select" id="sf-categories">${categoryOptions(step.categories)}</select>`,
        ) +
        outputKey +
        commonBottom
      );
    case "web_search":
      typeSpecific =
        field(
          "query_template",
          textInput(
            "sf-query-template",
            step.query_template,
            "z.B. {{ticker}} earnings",
          ),
        ) +
        field("prompt", textarea("sf-prompt", step.prompt, 80)) +
        field(
          "time_range",
          `<select class="modal-select" id="sf-time-range">${timeRangeOptions(step.time_range)}</select>`,
        ) +
        field(
          "categories",
          `<select class="modal-select" id="sf-categories">${categoryOptions(step.categories)}</select>`,
        ) +
        outputKey;
      break;
    case "finance":
      typeSpecific =
        field(
          "ticker_key",
          contextDropdown("sf-ticker-key", step.ticker_key, "selected_ticker"),
        ) + outputKey;
      break;
    case "finance_search":
      typeSpecific =
        field(
          "query_key (ISIN oder Suchbegriff)",
          contextDropdown("sf-query-key", step.query_key, "selected_isin"),
        ) +
        outputKey +
        defaultVal;
      break;
    case "http_fetch":
      typeSpecific =
        field(
          "url",
          textInput(
            "sf-url",
            step.url || step.url_template,
            "https://example.com/api",
          ),
        ) +
        field(
          "method",
          `<select class="modal-select" id="sf-method"><option value="GET" ${(step.method || "GET") === "GET" ? "selected" : ""}>GET</option><option value="POST" ${step.method === "POST" ? "selected" : ""}>POST</option></select>`,
        ) +
        field(
          "headers (JSON)",
          textInput(
            "sf-headers",
            step.headers ? JSON.stringify(step.headers) : "",
            "",
          ),
        ) +
        field("timeout", textInput("sf-timeout", step.timeout, "15")) +
        defaultVal +
        outputKey;
      break;
    case "xlsx_fetch": {
      const columnsStr = Array.isArray(step.columns)
        ? step.columns.join(", ")
        : step.columns || "";
      const existingFilters = Array.isArray(step.filters)
        ? step.filters
        : step.filter
          ? [
              {
                column: step.filter.column,
                operator: "equals",
                value: step.filter.value,
              },
            ]
          : [];
      const filtersStr = JSON.stringify(existingFilters, null, 2);
      typeSpecific =
        field(
          "url",
          textInput(
            "sf-url",
            step.url || step.url_template,
            "https://example.com/data.xlsx",
          ),
        ) +
        field(
          "sheet (Index oder Name)",
          textInput(
            "sf-sheet",
            step.sheet !== undefined ? String(step.sheet) : "0",
            "0",
          ),
        ) +
        field(
          "columns (kommasepariert, leer = alle)",
          textInput(
            "sf-xlsx-columns",
            columnsStr,
            "company_name, isin, near_term_status, sector",
          ),
        ) +
        field(
          "filters (JSON-Array) — Operatoren: equals, not_equals, contains, not_contains, not_empty, empty, starts_with, ends_with",
          textarea("sf-xlsx-filters", filtersStr, 120),
        ) +
        field("timeout", textInput("sf-timeout", step.timeout, "30")) +
        defaultVal +
        outputKey;
      break;
    }
    case "state_read":
      typeSpecific = keyField + outputKey + defaultVal;
      break;
    case "state_write":
      typeSpecific = keyField + sourceKey;
      break;
    case "state_read_external":
      typeSpecific = agentName + keyField + outputKey + defaultVal;
      break;
    case "state_write_external":
      typeSpecific = agentName + keyField + sourceKey;
      break;
    case "data_read":
      typeSpecific = namespace + keyTemplate + outputKey + defaultVal;
      break;
    case "data_write":
      typeSpecific = namespace + keyTemplate + sourceKey;
      break;
    case "data_read_external":
      typeSpecific =
        agentName + namespace + keyTemplate + outputKey + defaultVal;
      break;
    case "data_write_external":
      typeSpecific = agentName + namespace + keyTemplate + sourceKey;
      break;
    case "transform": {
      const op = step.operation || "map_field";
      typeSpecific =
        field(
          "operation",
          `<select class="modal-select" id="sf-operation" onchange="onTransformOpChange()">${transformOpOptions(op)}</select>`,
        ) +
        sourceKey +
        `<div id="sf-value-key-wrap">${field("value_key (list_append)", textInput("sf-value-key", step.value_key, "z.B. selected_isin"))}</div>` +
        `<div id="sf-field-wrap">${field("field (map_field / sort)", textInput("sf-field", step.field, "z.B. isin"))}</div>` +
        `<div id="sf-group-field-wrap">${field("group_field (group_by)", textInput("sf-group-field", step.group_field, "z.B. model"))}</div>` +
        `<div id="sf-value-field-wrap">${field("value_field (group_by, optional)", textInput("sf-value-field", step.value_field, "z.B. price"))}</div>` +
        `<div id="sf-filters-json-wrap">${field("filters (JSON-Array)", textarea("sf-filters-json", JSON.stringify(step.filters || [], null, 2), 100))}</div>` +
        `<div id="sf-subtract-key-wrap">${field("subtract_key (diff)", contextDropdown("sf-subtract-key", step.subtract_key, "z.B. known_list"))}</div>` +
        `<div id="sf-other-key-wrap">${field("other_key (intersect / union)", contextDropdown("sf-other-key", step.other_key, "z.B. list_b"))}</div>` +
        `<div id="sf-start-wrap">${field("start (slice)", textInput("sf-start", step.start !== undefined ? String(step.start) : "0", "0"))}</div>` +
        `<div id="sf-end-wrap">${field("end (slice, leer = bis Ende)", textInput("sf-end", step.end !== undefined ? String(step.end) : "", ""))}</div>` +
        `<div id="sf-reverse-wrap">${field("reverse (sort)", `<select class=\"modal-select\" id=\"sf-reverse\"><option value=\"false\" ${!step.reverse ? "selected" : ""}>false</option><option value=\"true\" ${step.reverse ? "selected" : ""}>true</option></select>`)}</div>` +
        `<div id="sf-output-format-wrap">${field("output_format", `<select class=\"modal-select\" id=\"sf-output-format\"><option value=\"json_array\" ${(step.output_format || "json_array") === "json_array" ? "selected" : ""}>json_array</option><option value=\"comma_list\" ${step.output_format === "comma_list" ? "selected" : ""}>comma_list</option></select>`)}</div>` +
        `<div id="sf-group-key-wrap">${field("group_key", textInput("sf-group-key", step.group_key, "z.B. extracted_model"))}</div>` +
        `<div id="sf-max-items-wrap">${field("max_items", textInput("sf-max-items", step.max_items, "500"))}</div>` +
        targetKey +
        `<div id="sf-multiplier-wrap">${field("multiplier", textInput("sf-multiplier", step.multiplier, "1.5"))}</div>` +
        `<div id="sf-model-key-wrap">${field("model_key", textInput("sf-model-key", step.model_key, "z.B. extracted_model"))}</div>` +
        `<div id="sf-functions-wrap">${field("functions", textInput("sf-functions", (step.functions || []).join(", "), "q1, q3, iqr, lower_bound"))}</div>` +
        `<div id="sf-path-wrap">${field("path", textInput("sf-path", step.path, ""))}</div>` +
        `<div id="sf-xpath-wrap">${field("xpath", textInput("sf-xpath", step.xpath, ""))}</div>` +
        `<div id="sf-attribute-wrap">${field("attribute", textInput("sf-attribute", step.attribute, ""))}</div>` +
        `<div id="sf-pattern-wrap">${field("pattern", textInput("sf-pattern", step.pattern, ""))}</div>` +
        `<div id="sf-group-num-wrap">${field("group", textInput("sf-group-num", step.group, "1"))}</div>` +
        `<div id="sf-expression-wrap">${field("expression", textInput("sf-expression", step.expression, ""))}</div>` +
        `<div id="sf-round-wrap">${field("round", textInput("sf-round", step.round, "2"))}</div>` +
        `<div id="sf-left-key-wrap">${field("left_key", textInput("sf-left-key", step.left_key, ""))}</div>` +
        `<div id="sf-right-key-wrap">${field("right_key", textInput("sf-right-key", step.right_key, ""))}</div>` +
        `<div id="sf-operator-wrap">${field("operator", `<select class="modal-select" id="sf-operator"><option ${(step.operator || "<=") === "<=" ? "selected" : ""}"><=</option><option ${step.operator === "<" ? "selected" : ""}><</option><option ${step.operator === ">=" ? "selected" : ""}>>=</option><option ${step.operator === ">" ? "selected" : ""}>>></option><option ${step.operator === "==" ? "selected" : ""}>==</option><option ${step.operator === "!=" ? "selected" : ""}>!=</option></select>`)}</div>` +
        `<div id="sf-output-true-wrap">${field("output_true", textInput("sf-output-true", step.output_true, "true"))}</div>` +
        `<div id="sf-output-false-wrap">${field("output_false", textInput("sf-output-false", step.output_false, "false"))}</div>` +
        outputKey;
      break;
    }
    case "trigger_agent":
      typeSpecific =
        field(
          "target_agent_name",
          textInput("sf-target-agent", step.target_agent_name, "z.B. Gordon"),
        ) +
        field(
          "payload (JSON)",
          textarea(
            "sf-payload",
            JSON.stringify(step.payload || {}, null, 2),
            80,
          ),
        ) +
        field("delay_minutes", textInput("sf-delay", step.delay_minutes, "0"));
      break;
    case "notify_user":
      typeSpecific =
        field(
          "condition_key",
          textInput("sf-condition-key", step.condition_key, ""),
        ) +
        field(
          "source_key",
          textInput("sf-source-key", step.source_key, "z.B. report"),
        ) +
        field(
          "message_template",
          textInput("sf-message-template", step.message_template, ""),
        );
      break;
  }
  return commonTop + typeSpecific + commonBottom;
}

function onStepTypeChange() {
  const type = document.getElementById("sf-type")?.value;
  if (!type) return;
  const currentStep = _readStepFromForm();
  currentStep.type = type;
  const posField = document.getElementById("sf-position");
  const posParent = posField ? posField.closest(".modal-field") : null;
  const posHtml = posParent ? posParent.outerHTML : "";
  document.getElementById("modal-body").innerHTML =
    posHtml + buildStepFormBody(currentStep);
  if (type === "transform") onTransformOpChange();
}

function onTransformOpChange() {
  const op = document.getElementById("sf-operation")?.value;
  const show = (id, v) => {
    const el = document.getElementById(id);
    if (el) el.style.display = v ? "" : "none";
  };
  show("sf-value-key-wrap", op === "list_append");
  show("sf-group-key-wrap", op === "group_by");
  show("sf-group-field-wrap", op === "group_by");
  show("sf-value-field-wrap", op === "group_by");
  show("sf-max-items-wrap", op === "list_append" || op === "group_by");
  show("sf-field-wrap", op === "map_field" || op === "sort");
  show(
    "sf-output-format-wrap",
    op === "map_field" ||
      op === "diff" ||
      op === "intersect" ||
      op === "union" ||
      op === "list_append" ||
      op === "flatten",
  );
  show("sf-filters-json-wrap", op === "filter");
  show("sf-start-wrap", op === "slice");
  show("sf-end-wrap", op === "slice");
  show("sf-subtract-key-wrap", op === "diff");
  show("sf-other-key-wrap", op === "intersect" || op === "union");
  show("sf-reverse-wrap", op === "sort");
  show("sf-multiplier-wrap", op === "statistics");
  show("sf-model-key-wrap", op === "statistics");
  show("sf-functions-wrap", op === "statistics");
  show("sf-path-wrap", op === "json_path");
  show("sf-xpath-wrap", op === "xml_extract");
  show("sf-attribute-wrap", op === "xml_extract");
  show("sf-pattern-wrap", op === "regex_extract");
  show("sf-group-num-wrap", op === "regex_extract");
  show("sf-expression-wrap", op === "arithmetic");
  show("sf-round-wrap", op === "arithmetic");
  show("sf-left-key-wrap", op === "compare");
  show("sf-right-key-wrap", op === "compare");
  show("sf-operator-wrap", op === "compare");
  show("sf-output-true-wrap", op === "compare");
  show("sf-output-false-wrap", op === "compare");
  show("sf-source-key", !["arithmetic", "compare"].includes(op));
  show("sf-target-key", op === "list_append");
}

function _val(id) {
  const el = document.getElementById(id);
  if (!el) return undefined;
  if (el.type === "checkbox") return el.checked;
  return el.value.trim();
}

function _readStepFromForm() {
  const type = _val("sf-type") || "llm_extract";
  const id = _val("sf-id") || "";
  const routeRaw = _val("sf-route") || "";
  let onlyIfRoute = undefined;
  if (routeRaw) {
    const parts = routeRaw
      .split(",")
      .map((s) => s.trim())
      .filter(Boolean);
    onlyIfRoute = parts.length === 1 ? parts[0] : parts;
  }
  const onlyIfKeyKey = _val("sf-only-if-key") || "";
  const onlyIfKeyVal = _val("sf-only-if-val") ?? "";
  const step = { id, type };
  if (onlyIfRoute !== undefined) step.only_if_route = onlyIfRoute;
  if (onlyIfKeyKey)
    step.only_if_key = { key: onlyIfKeyKey, value: onlyIfKeyVal };
  if (_val("sf-output")) step.is_output = true;
  const outputKey = _val("sf-output-key");
  if (outputKey) step.output_key = outputKey;
  switch (type) {
    case "router_match":
      try {
        step.rules = JSON.parse(_val("sf-rules") || "[]");
      } catch {
        step.rules = [];
      }
      step.default = _val("sf-default-route") || "idle";
      break;
    case "router_llm":
      step.prompt = _val("sf-prompt") || "";
      break;
    case "llm_extract":
    case "llm_decide":
    case "llm_analyze":
      step.prompt = _val("sf-prompt") || "";
      break;
    case "llm_summarize": {
      step.prompt = _val("sf-prompt") || "";
      const sq = _val("sf-search-query");
      if (sq) step.search_query = sq;
      const tr = _val("sf-time-range");
      if (tr) step.time_range = tr;
      const cat = _val("sf-categories");
      if (cat) step.categories = cat;
      break;
    }
    case "web_search": {
      step.query_template = _val("sf-query-template") || "";
      step.prompt = _val("sf-prompt") || "";
      const tr = _val("sf-time-range");
      if (tr) step.time_range = tr;
      const cat = _val("sf-categories");
      if (cat) step.categories = cat;
      break;
    }
    case "finance":
      step.ticker_key = _val("sf-ticker-key") || "selected_ticker";
      break;
    case "finance_search":
      step.query_key = _val("sf-query-key") || "selected_isin";
      const fsDef = _val("sf-default");
      if (fsDef) step.default = fsDef;
      break;
    case "state_read": {
      step.key = _val("sf-key") || "";
      const def = _val("sf-default");
      if (def) step.default = def;
      break;
    }
    case "state_write":
      step.key = _val("sf-key") || "";
      step.source_key = _val("sf-source-key") || "";
      break;
    case "state_read_external": {
      step.agent_name = _val("sf-agent-name") || "";
      step.key = _val("sf-key") || "";
      const def = _val("sf-default");
      if (def) step.default = def;
      break;
    }
    case "state_write_external":
      step.agent_name = _val("sf-agent-name") || "";
      step.key = _val("sf-key") || "";
      step.source_key = _val("sf-source-key") || "";
      break;
    case "data_read": {
      step.namespace = _val("sf-namespace") || "";
      step.key_template = _val("sf-key-template") || "";
      const def = _val("sf-default");
      if (def) step.default = def;
      break;
    }
    case "data_write":
      step.namespace = _val("sf-namespace") || "";
      step.key_template = _val("sf-key-template") || "";
      step.source_key = _val("sf-source-key") || "";
      break;
    case "data_read_external": {
      step.agent_name = _val("sf-agent-name") || "";
      step.namespace = _val("sf-namespace") || "";
      step.key_template = _val("sf-key-template") || "";
      const def = _val("sf-default");
      if (def) step.default = def;
      break;
    }
    case "data_write_external":
      step.agent_name = _val("sf-agent-name") || "";
      step.namespace = _val("sf-namespace") || "";
      step.key_template = _val("sf-key-template") || "";
      step.source_key = _val("sf-source-key") || "";
      break;
    case "transform": {
      step.operation = _val("sf-operation") || "array_push";
      step.source_key = _val("sf-source-key") || "";
      const tk = _val("sf-target-key");
      if (tk) step.target_key = tk;
      if (step.operation === "map_field") {
        step.field = _val("sf-field") || "";
        const fmt = _val("sf-output-format");
        if (fmt && fmt !== "json_array") step.output_format = fmt;
      }
      if (step.operation === "filter") {
        try {
          step.filters = JSON.parse(_val("sf-filters-json") || "[]");
        } catch {
          step.filters = [];
        }
      }
      if (step.operation === "slice") {
        const s = _val("sf-start");
        if (s !== "") step.start = parseInt(s);
        const e = _val("sf-end");
        if (e !== "") step.end = parseInt(e);
      }
      if (step.operation === "diff") {
        step.subtract_key = _val("sf-subtract-key") || "";
        const fmt = _val("sf-output-format");
        if (fmt && fmt !== "json_array") step.output_format = fmt;
      }
      if (step.operation === "intersect" || step.operation === "union") {
        step.other_key = _val("sf-other-key") || "";
        const fmt = _val("sf-output-format");
        if (fmt && fmt !== "json_array") step.output_format = fmt;
      }
      if (step.operation === "list_append") {
        step.value_key = _val("sf-value-key") || "";
        const mi = _val("sf-max-items");
        if (mi) step.max_items = parseInt(mi);
        const fmt = _val("sf-output-format");
        if (fmt && fmt !== "json_array") step.output_format = fmt;
      }
      if (step.operation === "group_by") {
        step.group_field = _val("sf-group-field") || "";
        const vf = _val("sf-value-field");
        if (vf) step.value_field = vf;
        const mi = _val("sf-max-items");
        if (mi) step.max_items = parseInt(mi);
      }
      if (step.operation === "sort") {
        const f = _val("sf-field");
        if (f) step.field = f;
        step.reverse = _val("sf-reverse") === "true";
      }
      if (step.operation === "flatten") {
        const fmt = _val("sf-output-format");
        if (fmt && fmt !== "json_array") step.output_format = fmt;
      }
      if (step.operation === "statistics") {
        const mult = _val("sf-multiplier");
        if (mult) step.multiplier = parseFloat(mult);
        const mk = _val("sf-model-key");
        if (mk) step.model_key = mk;
        const fns = _val("sf-functions");
        if (fns)
          step.functions = fns
            .split(",")
            .map((f) => f.trim())
            .filter(Boolean);
      }
      if (step.operation === "json_path") {
        step.path = _val("sf-path") || "";
        const def = _val("sf-default");
        if (def) step.default = def;
      }
      if (step.operation === "xml_extract") {
        step.xpath = _val("sf-xpath") || "";
        const attr = _val("sf-attribute");
        if (attr) step.attribute = attr;
        const def = _val("sf-default");
        if (def) step.default = def;
      }
      if (step.operation === "regex_extract") {
        step.pattern = _val("sf-pattern") || "";
        const grp = _val("sf-group-num");
        if (grp) step.group = parseInt(grp);
        const def = _val("sf-default");
        if (def) step.default = def;
      }
      if (step.operation === "arithmetic") {
        step.expression = _val("sf-expression") || "";
        const rnd = _val("sf-round");
        if (rnd !== "" && rnd !== undefined) step.round = parseInt(rnd);
        const def = _val("sf-default");
        if (def) step.default = def;
      }
      if (step.operation === "compare") {
        step.left_key = _val("sf-left-key") || "";
        step.right_key = _val("sf-right-key") || "";
        step.operator = _val("sf-operator") || "<=";
        const ot = _val("sf-output-true");
        if (ot) step.output_true = ot;
        const of_ = _val("sf-output-false");
        if (of_) step.output_false = of_;
      }
      break;
    }
    case "http_fetch": {
      const urlVal = _val("sf-url") || "";
      if (urlVal.includes("{{")) step.url_template = urlVal;
      else step.url = urlVal;
      step.method = _val("sf-method") || "GET";
      const hdrs = _val("sf-headers");
      if (hdrs) {
        try {
          step.headers = JSON.parse(hdrs);
        } catch {}
      }
      const to = _val("sf-timeout");
      if (to) step.timeout = parseFloat(to);
      const def = _val("sf-default");
      if (def) step.default = def;
      break;
    }
    case "xlsx_fetch": {
      const urlVal = _val("sf-url") || "";
      if (urlVal.includes("{{")) step.url_template = urlVal;
      else step.url = urlVal;
      const sheetRaw = _val("sf-sheet") || "0";
      step.sheet = isNaN(Number(sheetRaw)) ? sheetRaw : Number(sheetRaw);
      const colsRaw = _val("sf-xlsx-columns") || "";
      if (colsRaw) {
        step.columns = colsRaw
          .split(",")
          .map((c) => c.trim())
          .filter(Boolean);
      }
      const filtersRaw = _val("sf-xlsx-filters") || "[]";
      try {
        const parsed = JSON.parse(filtersRaw);
        if (Array.isArray(parsed) && parsed.length > 0) step.filters = parsed;
      } catch {
        // ungültiges JSON ignorieren
      }
      const to = _val("sf-timeout");
      if (to) step.timeout = parseFloat(to);
      const def = _val("sf-default");
      if (def) step.default = def;
      break;
    }
    case "trigger_agent": {
      step.target_agent_name = _val("sf-target-agent") || "";
      try {
        step.payload = JSON.parse(_val("sf-payload") || "{}");
      } catch {
        step.payload = {};
      }
      const delay = _val("sf-delay");
      if (delay) step.delay_minutes = parseInt(delay);
      break;
    }
    case "notify_user": {
      const ck = _val("sf-condition-key");
      if (ck) step.condition_key = ck;
      const sk = _val("sf-source-key");
      if (sk) step.source_key = sk;
      const mt = _val("sf-message-template");
      if (mt) step.message_template = mt;
      break;
    }
  }
  return step;
}

function renderStep(step, idx, agentId) {
  const summary = stepSummary(step);
  return `
    <div class="pipeline-step">
      <div class="pipeline-step-header">
        <div class="pipeline-step-info">
          <span class="pipeline-step-id">${step.id}</span>
          ${stepTypeBadge(step.type || "?")}
          ${step.is_output ? '<span class="badge badge-active">output</span>' : ""}
        </div>
        <div class="pipeline-step-actions">
          <button class="btn btn-sm" onclick="editStep(${agentId},${idx})">Bearbeiten</button>
          <button class="btn btn-sm btn-danger" onclick="deleteStep(${agentId},${idx})">Loeschen</button>
        </div>
      </div>
      ${summary ? `<div class="pipeline-step-prompt">${summary}</div>` : ""}
    </div>`;
}

function editStep(agentId, idx) {
  const a = agentsData.find((x) => x.id === agentId);
  const steps = a.steps || [];
  const step = steps[idx];
  _stepContextKeys = steps
    .slice(0, idx)
    .filter((s) => s.output_key)
    .map((s) => s.output_key);
  _stepRouteKeys = steps
    .slice(0, idx)
    .filter((s) => s.type === "router_match" || s.type === "router_llm")
    .flatMap((s) => {
      const routes = (s.rules || []).map((r) => r.then).filter(Boolean);
      if (s.default) routes.push(s.default);
      return routes;
    });
  const posField = `<div class="modal-field"><div class="modal-label">Position</div><input class="modal-input" id="sf-position" type="number" value="${idx}" /></div>`;
  openModal(
    "Step bearbeiten",
    posField + buildStepFormBody(step),
    `<button class="btn" onclick="closeModal()">Abbrechen</button><button class="btn btn-accent" onclick="saveStep(${agentId},${idx})">Speichern</button>`,
  );
  if (step.type === "transform") onTransformOpChange();
}

async function saveStep(agentId, idx) {
  const a = agentsData.find((x) => x.id === agentId);
  const steps = [...(a.steps || [])];
  const updatedStep = _readStepFromForm();
  const posRaw = _val("sf-position");
  const newPos =
    posRaw !== "" && posRaw !== null && posRaw !== undefined
      ? parseInt(posRaw, 10)
      : idx;
  steps.splice(idx, 1);
  if (isNaN(newPos) || newPos >= steps.length) steps.push(updatedStep);
  else steps.splice(Math.max(0, newPos), 0, updatedStep);
  try {
    await api("/api/agents/" + agentId, {
      method: "PATCH",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ steps }),
    });
    a.steps = steps;
    closeModal();
    toast("Gespeichert.");
    selectAgent(agentId);
  } catch {
    toast("Fehler.", true);
  }
}

function deleteStep(agentId, idx) {
  confirmModal("Step wirklich loeschen?", async () => {
    const a = agentsData.find((x) => x.id === agentId);
    const steps = [...(a.steps || [])];
    steps.splice(idx, 1);
    try {
      await api("/api/agents/" + agentId, {
        method: "PATCH",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ steps }),
      });
      a.steps = steps;
      toast("Geloescht.");
      selectAgent(agentId);
    } catch {
      toast("Fehler.", true);
    }
  });
}

function addStep(agentId) {
  const a = agentsData.find((x) => x.id === agentId);
  const steps = a.steps || [];
  _stepContextKeys = steps.filter((s) => s.output_key).map((s) => s.output_key);
  _stepRouteKeys = steps
    .filter((s) => s.type === "router_match" || s.type === "router_llm")
    .flatMap((s) => {
      const routes = (s.rules || []).map((r) => r.then).filter(Boolean);
      if (s.default) routes.push(s.default);
      return routes;
    });
  const posField = `<div class="modal-field"><div class="modal-label">Position (leer = ans Ende)</div><input class="modal-input" id="sf-position" type="number" placeholder="0 = Anfang" /></div>`;
  openModal(
    "Neuen Step hinzufuegen",
    posField + buildStepFormBody({ id: "", type: "llm_extract" }),
    `<button class="btn" onclick="closeModal()">Abbrechen</button><button class="btn btn-accent" onclick="saveNewStep(${agentId})">Hinzufuegen</button>`,
  );
}

async function saveNewStep(agentId) {
  const step = _readStepFromForm();
  if (!step.id) {
    toast("ID ist Pflichtfeld.", true);
    return;
  }
  const posRaw = _val("sf-position");
  const pos =
    posRaw !== "" && posRaw !== null && posRaw !== undefined
      ? parseInt(posRaw, 10)
      : null;
  const a = agentsData.find((x) => x.id === agentId);
  const steps = [...(a.steps || [])];
  if (pos === null || isNaN(pos) || pos >= steps.length) steps.push(step);
  else steps.splice(pos, 0, step);
  try {
    await api("/api/agents/" + agentId, {
      method: "PATCH",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ steps }),
    });
    a.steps = steps;
    closeModal();
    toast("Hinzugefuegt.");
    selectAgent(agentId);
  } catch {
    toast("Fehler.", true);
  }
}

document.querySelectorAll(".nav-btn[data-section]").forEach((btn) =>
  btn.addEventListener("click", function () {
    switchSection(this.dataset.section);
  }),
);
document.querySelectorAll(".bottom-nav-item").forEach((btn) =>
  btn.addEventListener("click", function () {
    switchSection(this.dataset.section);
  }),
);
document.getElementById("agents-search").addEventListener("input", function () {
  renderAgentsList(this.value);
});
document.getElementById("memory-search").addEventListener("input", function () {
  renderMemoryList(this.value);
});

async function loadAgents() {
  try {
    agentsData = await api("/api/agents");
    renderAgentsList();
  } catch {
    document.getElementById("agents-list").innerHTML =
      '<div style="font-size:13px;color:var(--red);padding:8px;">Backend nicht erreichbar.</div>';
  }
}

function renderAgentsList(filter) {
  const q = (filter || "").toLowerCase();
  const filtered = q
    ? agentsData.filter(
        (a) => a.name.toLowerCase().includes(q) || (a.type || "").includes(q),
      )
    : agentsData;
  if (!filtered.length) {
    document.getElementById("agents-list").innerHTML =
      '<div style="font-size:13px;color:var(--text3);padding:8px;">Keine Agenten.</div>';
    return;
  }
  document.getElementById("agents-list").innerHTML = filtered
    .map((a) => {
      const totalSteps = (a.steps || []).length;
      return `<div class="sidebar-item" data-id="${a.id}" onclick="selectAgent(${a.id})">
      <div class="si-name">${a.name}<span class="badge ${a.is_active ? "badge-active" : "badge-inactive"}">${a.is_active ? "aktiv" : "inaktiv"}</span>${totalSteps > 0 ? `<span class="badge badge-pipeline">${totalSteps} steps</span>` : ""}</div>
      <div class="si-meta">${a.type || "--"} . ${a.schedule || "trigger-only"}</div></div>`;
    })
    .join("");
}

async function selectAgent(id) {
  currentAgentId = id;
  document
    .querySelectorAll("#agents-list .sidebar-item")
    .forEach((el) =>
      el.classList.toggle("active", Number(el.dataset.id) === id),
    );
  const a = agentsData.find((x) => x.id === id);
  const detail = document.getElementById("agents-detail");
  detail.innerHTML = '<div class="empty-state">Lade...</div>';
  showDetail(detail, document.getElementById("agents-sidebar"), a.name);
  try {
    const [state, data, memories] = await Promise.all([
      api("/api/agents/" + id + "/state"),
      api("/api/agents/" + id + "/data"),
      api("/api/agents/" + id + "/memories"),
    ]);
    _agentMemCache[id] = memories;
    _agentDataCache[id] = data;
    const nsMap = {};
    data.forEach((d, i) => {
      if (!nsMap[d.namespace]) nsMap[d.namespace] = [];
      nsMap[d.namespace].push({ ...d, _idx: i });
    });
    const allSteps = a.steps || [];
    detail.innerHTML = `
      <div class="action-bar">
        <button class="btn btn-accent" onclick="triggerAgent(${id})">Jetzt ausfuehren</button>
        <button class="btn" onclick="editAgent(${id})">Bearbeiten</button>
        ${a.is_active ? `<button class="btn btn-danger" onclick="stopAgent(${id})">Stoppen</button>` : ""}
      </div>
      <div class="detail-block"><div class="detail-block-title">Anweisung</div><div class="detail-text">${a.instruction || "--"}</div></div>
      <div class="detail-block"><div class="detail-block-title">Konfiguration</div>
        <div class="detail-mono">Zeitplan: ${a.schedule || "nur auf Trigger"}</div>
        <div class="detail-mono">Letzter Lauf: ${fmt(a.last_run_at)}</div>
        <div class="detail-mono">Naechster Lauf: ${a.next_run_at ? fmt(a.next_run_at) : "--"}</div>
        <div class="detail-mono">Typ: ${a.type || "--"}</div></div>
      <div class="detail-block"><div class="detail-block-title">Pipeline (${allSteps.length} Steps)<button class="btn btn-sm btn-accent" onclick="addStep(${id})">+ Step</button></div>
        ${allSteps.length ? allSteps.map((s, i) => renderStep(s, i, id)).join("") : '<div style="font-size:13px;color:var(--text3);">Keine Pipeline.</div>'}</div>
      <hr class="divider">
      <div class="agent-tabs">
        <button class="agent-tab active" data-tab="state">State <span class="agent-tab-count">${state.length}</span></button>
        <button class="agent-tab" data-tab="data">Data <span class="agent-tab-count">${data.length}</span></button>
        <button class="agent-tab" data-tab="memory">Memory <span class="agent-tab-count">${memories.length}</span></button>
      </div>
      <div class="tab-panel active" id="tab-state-${id}">
        <div class="state-grid">
          ${state.map((s) => `<div class="state-card"><div class="state-card-header"><div class="state-key">${s.key}</div><div class="state-actions"><button class="btn btn-sm" data-agent="${id}" data-key="${s.key}" data-action="edit-state">Bearbeiten</button><button class="btn btn-sm btn-danger" data-agent="${id}" data-key="${s.key}" data-action="delete-state">Loeschen</button></div></div><div class="state-val">${s.value}</div></div>`).join("")}
          ${!state.length ? '<div style="font-size:13px;color:var(--text3);padding:8px 0;">Kein State.</div>' : ""}
        </div>
      </div>
      <div class="tab-panel" id="tab-data-${id}">
        <div style="display:flex;justify-content:flex-end;margin-bottom:10px;"><button class="btn btn-sm btn-accent" onclick="addAgentData(${id})">+ Neu</button></div>
        ${
          Object.keys(nsMap).length
            ? Object.entries(nsMap)
                .map(
                  ([ns, entries]) =>
                    `<div class="ns-block"><div class="ns-label">${ns}</div>${entries.map((d) => `<div class="data-card"><div class="data-card-header"><div class="data-key">${d.key} <span style="color:var(--text3);margin-left:6px;">${fmt(d.updated_at)}</span></div><div class="data-actions"><button class="btn btn-sm" data-agent="${id}" data-idx="${d._idx}" data-action="edit-data">Bearbeiten</button><button class="btn btn-sm btn-danger" data-agent="${id}" data-idx="${d._idx}" data-action="delete-data">Loeschen</button></div></div><div class="data-val">${d.value}</div></div>`).join("")}</div>`,
                )
                .join("")
            : '<div style="font-size:13px;color:var(--text3);padding:8px 0;">Keine Data-Eintraege.</div>'
        }
      </div>
      <div class="tab-panel" id="tab-memory-${id}">
        <div style="display:flex;justify-content:flex-end;margin-bottom:10px;"><button class="btn btn-sm btn-accent" onclick="addAgentMemory(${id})">+ Neu</button></div>
        <div class="mem-list">
          ${memories.map((m, i) => `<div class="mem-row"><div class="mem-text">${m.content}</div><div class="mem-date">${fmt(m.created_at)}</div><div class="mem-actions"><button class="btn btn-sm" data-agent="${id}" data-idx="${i}" data-action="edit-agent-mem">Edit</button><button class="btn btn-sm btn-danger" data-agent="${id}" data-idx="${i}" data-action="delete-agent-mem">Del</button></div></div>`).join("")}
          ${!memories.length ? '<div style="font-size:13px;color:var(--text3);padding:8px 0;">Keine Memories.</div>' : ""}
        </div>
      </div>`;
    detail.querySelectorAll(".agent-tab").forEach((tab) =>
      tab.addEventListener("click", function () {
        detail
          .querySelectorAll(".agent-tab")
          .forEach((t) => t.classList.remove("active"));
        detail
          .querySelectorAll(".tab-panel")
          .forEach((p) => p.classList.remove("active"));
        this.classList.add("active");
        detail
          .querySelector("#tab-" + this.dataset.tab + "-" + id)
          .classList.add("active");
      }),
    );
    detail.removeEventListener("click", handleAgentDetailClick);
    detail.addEventListener("click", handleAgentDetailClick);
  } catch {
    detail.innerHTML =
      '<div style="font-size:13px;color:var(--red);padding:1rem;">Fehler beim Laden.</div>';
  }
}

function handleAgentDetailClick(e) {
  const btn = e.target.closest("[data-action]");
  if (!btn) return;
  const action = btn.dataset.action;
  const agentId = Number(btn.dataset.agent);
  if (action === "edit-state") {
    const key = btn.dataset.key;
    const currentVal = btn
      .closest(".state-card")
      .querySelector(".state-val").textContent;
    openModal(
      "State bearbeiten",
      `<div class="modal-field"><div class="modal-label">Key</div><input class="modal-input" value="${key}" disabled /></div><div class="modal-field"><div class="modal-label">Value</div><textarea class="modal-input" id="edit-state-val" style="min-height:140px;">${currentVal}</textarea></div>`,
      `<button class="btn" onclick="closeModal()">Abbrechen</button><button class="btn btn-accent" onclick="saveStateEntry(${agentId},'${key}')">Speichern</button>`,
    );
  } else if (action === "delete-state") {
    const key = btn.dataset.key;
    confirmModal(`State-Eintrag "${key}" wirklich loeschen?`, () => {
      api("/api/agents/" + agentId + "/state/" + encodeURIComponent(key), {
        method: "DELETE",
      })
        .then(() => {
          toast("Geloescht.");
          selectAgent(agentId);
        })
        .catch(() => toast("Fehler.", true));
    });
  } else if (action === "edit-data") {
    const idx = Number(btn.dataset.idx);
    const d = _agentDataCache[agentId][idx];
    openModal(
      "Data bearbeiten",
      `<div class="modal-field"><div class="modal-label">Namespace</div><input class="modal-input" value="${d.namespace}" disabled /></div><div class="modal-field"><div class="modal-label">Key</div><input class="modal-input" value="${d.key}" disabled /></div><div class="modal-field"><div class="modal-label">Value</div><textarea class="modal-input" id="edit-data-val" style="min-height:200px;">${d.value}</textarea></div>`,
      `<button class="btn" onclick="closeModal()">Abbrechen</button><button class="btn btn-accent" onclick="saveAgentData(${agentId},${idx})">Speichern</button>`,
    );
  } else if (action === "delete-data") {
    const idx = Number(btn.dataset.idx);
    const d = _agentDataCache[agentId][idx];
    confirmModal(`Eintrag "${d.namespace}/${d.key}" loeschen?`, () => {
      api(
        `/api/agents/${agentId}/data/${encodeURIComponent(d.namespace)}/${encodeURIComponent(d.key)}`,
        { method: "DELETE" },
      )
        .then(() => {
          toast("Geloescht.");
          selectAgent(agentId);
        })
        .catch(() => toast("Fehler.", true));
    });
  } else if (action === "edit-agent-mem") {
    const idx = Number(btn.dataset.idx);
    const mem = _agentMemCache[agentId][idx];
    openModal(
      "Memory bearbeiten",
      `<div class="modal-field"><div class="modal-label">Inhalt</div><textarea class="modal-input" id="edit-mem-content" style="min-height:100px;">${mem.content}</textarea></div>`,
      `<button class="btn" onclick="closeModal()">Abbrechen</button><button class="btn btn-accent" onclick="saveAgentMemory(${agentId},${idx})">Speichern</button>`,
    );
  } else if (action === "delete-agent-mem") {
    const idx = Number(btn.dataset.idx);
    const mem = _agentMemCache[agentId][idx];
    confirmModal("Memory loeschen?", () => {
      api("/api/agents/" + agentId + "/memories", {
        method: "DELETE",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ content: mem.content, subject_type: "agent" }),
      })
        .then(() => {
          toast("Geloescht.");
          selectAgent(agentId);
        })
        .catch(() => toast("Fehler.", true));
    });
  }
}

async function saveStateEntry(agentId, key) {
  const value = document.getElementById("edit-state-val").value;
  try {
    await api("/api/agents/" + agentId + "/state/" + encodeURIComponent(key), {
      method: "PATCH",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ value }),
    });
    closeModal();
    toast("Gespeichert.");
    selectAgent(agentId);
  } catch {
    toast("Fehler.", true);
  }
}
async function saveAgentData(agentId, idx) {
  const d = _agentDataCache[agentId][idx];
  const value = document.getElementById("edit-data-val").value;
  try {
    await api(
      `/api/agents/${agentId}/data/${encodeURIComponent(d.namespace)}/${encodeURIComponent(d.key)}`,
      {
        method: "PATCH",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ value }),
      },
    );
    closeModal();
    toast("Gespeichert.");
    selectAgent(agentId);
  } catch {
    toast("Fehler.", true);
  }
}
function addAgentData(agentId) {
  const existing = _agentDataCache[agentId] || [];
  const namespaces = [...new Set(existing.map((d) => d.namespace))];
  openModal(
    "Data-Eintrag hinzufuegen",
    `<div class="modal-field"><div class="modal-label">Namespace</div><input class="modal-input" id="new-data-ns" list="ns-suggestions" placeholder="z.B. analyses" /><datalist id="ns-suggestions">${namespaces.map((ns) => `<option value="${ns}">`).join("")}</datalist></div><div class="modal-field"><div class="modal-label">Key</div><input class="modal-input" id="new-data-key" placeholder="z.B. EOSE" /></div><div class="modal-field"><div class="modal-label">Value</div><textarea class="modal-input" id="new-data-val" style="min-height:120px;"></textarea></div>`,
    `<button class="btn" onclick="closeModal()">Abbrechen</button><button class="btn btn-accent" onclick="saveNewAgentData(${agentId})">Speichern</button>`,
  );
}
async function saveNewAgentData(agentId) {
  const namespace = document.getElementById("new-data-ns").value.trim();
  const key = document.getElementById("new-data-key").value.trim();
  const value = document.getElementById("new-data-val").value.trim();
  if (!namespace || !key || !value) {
    toast("Alle Felder sind Pflicht.", true);
    return;
  }
  try {
    await api("/api/agents/" + agentId + "/data", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ namespace, key, value }),
    });
    closeModal();
    toast("Gespeichert.");
    selectAgent(agentId);
  } catch {
    toast("Fehler.", true);
  }
}
async function saveAgentMemory(agentId, idx) {
  const mem = _agentMemCache[agentId][idx];
  const newContent = document.getElementById("edit-mem-content").value.trim();
  try {
    await api("/api/agents/" + agentId + "/memories", {
      method: "PATCH",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        old_content: mem.content,
        new_content: newContent,
        subject_type: "agent",
      }),
    });
    closeModal();
    toast("Gespeichert.");
    selectAgent(agentId);
  } catch {
    toast("Fehler.", true);
  }
}
function addAgentMemory(agentId) {
  openModal(
    "Memory hinzufuegen",
    `<div class="modal-field"><div class="modal-label">Inhalt</div><textarea class="modal-input" id="new-mem-content" placeholder="Neue Beobachtung..."></textarea></div>`,
    `<button class="btn" onclick="closeModal()">Abbrechen</button><button class="btn btn-accent" onclick="saveNewAgentMemory(${agentId})">Speichern</button>`,
  );
}
async function saveNewAgentMemory(agentId) {
  const content = document.getElementById("new-mem-content").value.trim();
  if (!content) return;
  try {
    await api("/api/agents/" + agentId + "/memories", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ content, subject_type: "agent" }),
    });
    closeModal();
    toast("Gespeichert.");
    selectAgent(agentId);
  } catch {
    toast("Fehler.", true);
  }
}
async function triggerAgent(id) {
  try {
    await api("/api/agents/" + id + "/trigger", { method: "POST" });
    toast("Trigger gesetzt.");
  } catch {
    toast("Fehler.", true);
  }
}
async function stopAgent(id) {
  confirmModal("Agent wirklich stoppen?", async () => {
    try {
      await api("/api/agents/" + id, { method: "DELETE" });
      toast("Agent gestoppt.");
      goBack();
      await loadAgents();
    } catch {
      toast("Fehler.", true);
    }
  });
}
function editAgent(id) {
  const a = agentsData.find((x) => x.id === id);
  openModal(
    "Agent bearbeiten",
    `<div class="modal-field"><div class="modal-label">Name</div><input class="modal-input" id="edit-name" value="${a.name}" /></div><div class="modal-field"><div class="modal-label">Zeitplan (Cron, leer = nur Trigger)</div><input class="modal-input" id="edit-schedule" value="${a.schedule || ""}" /></div><div class="modal-field"><div class="modal-label">Anweisung</div><textarea class="modal-input" id="edit-instruction" style="min-height:140px;">${a.instruction}</textarea></div>`,
    `<button class="btn" onclick="closeModal()">Abbrechen</button><button class="btn btn-accent" onclick="saveAgent(${id})">Speichern</button>`,
  );
}
async function saveAgent(id) {
  const name = document.getElementById("edit-name").value.trim();
  const schedule =
    document.getElementById("edit-schedule").value.trim() || null;
  const instruction = document.getElementById("edit-instruction").value.trim();
  try {
    await api("/api/agents/" + id, {
      method: "PATCH",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ name, schedule, instruction }),
    });
    closeModal();
    toast("Gespeichert.");
    await loadAgents();
    selectAgent(id);
  } catch {
    toast("Fehler.", true);
  }
}
