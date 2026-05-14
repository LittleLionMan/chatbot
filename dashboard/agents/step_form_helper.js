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
      const op = step.operation || "map_field";
      const opSel = field(
        "operation",
        `<select class="modal-select" id="sf-operation" onchange="onStepTypeChange()">${transformOpOptions(op)}</select>`,
      );
      const outFmt = field(
        "output_format",
        `<select class="modal-select" id="sf-output-format"><option value="json_array" ${(step.output_format || "json_array") === "json_array" ? "selected" : ""}>json_array</option><option value="comma_list" ${step.output_format === "comma_list" ? "selected" : ""}>comma_list</option></select>`,
      );
      const srcKey = field(
        "source_key",
        contextDropdown("sf-source-key", step.source_key, "z.B. items"),
      );
      const tgtKey = field(
        "target_key",
        contextDropdown(
          "sf-target-key",
          step.target_key,
          "z.B. historical_prices",
        ),
      );
      const maxIt = field(
        "max_items",
        textInput("sf-max-items", step.max_items, "500"),
      );

      if (op === "array_push") {
        typeSpecific =
          opSel +
          field(
            "value_key",
            contextDropdown("sf-value-key", step.value_key, "z.B. price_eur"),
          ) +
          field(
            "group_key",
            contextDropdown(
              "sf-group-key",
              step.group_key,
              "z.B. extracted_model",
            ),
          ) +
          tgtKey +
          maxIt +
          outputKey;
      } else if (op === "map_field") {
        typeSpecific =
          opSel +
          srcKey +
          field("field", textInput("sf-field", step.field, "z.B. isin")) +
          outFmt +
          outputKey;
      } else if (op === "filter") {
        typeSpecific =
          opSel +
          srcKey +
          field(
            "filters (JSON-Array)",
            textarea(
              "sf-filters-json",
              JSON.stringify(step.filters || [], null, 2),
              100,
            ),
          ) +
          outputKey;
      } else if (op === "first") {
        typeSpecific =
          opSel +
          srcKey +
          field("default", textInput("sf-default", step.default, "")) +
          outputKey;
      } else if (op === "slice") {
        typeSpecific =
          opSel +
          srcKey +
          field(
            "start",
            textInput(
              "sf-start",
              step.start !== undefined ? String(step.start) : "0",
              "0",
            ),
          ) +
          field(
            "end (leer = bis Ende)",
            textInput(
              "sf-end",
              step.end !== undefined ? String(step.end) : "",
              "",
            ),
          ) +
          outputKey;
      } else if (op === "diff") {
        typeSpecific =
          opSel +
          srcKey +
          field(
            "subtract_key",
            contextDropdown(
              "sf-subtract-key",
              step.subtract_key,
              "z.B. known_list",
            ),
          ) +
          outFmt +
          outputKey;
      } else if (op === "intersect" || op === "union") {
        typeSpecific =
          opSel +
          srcKey +
          field(
            "other_key",
            contextDropdown("sf-other-key", step.other_key, "z.B. list_b"),
          ) +
          outFmt +
          outputKey;
      } else if (op === "list_append") {
        typeSpecific =
          opSel +
          field(
            "value_key",
            contextDropdown(
              "sf-value-key",
              step.value_key,
              "z.B. selected_isin",
            ),
          ) +
          tgtKey +
          maxIt +
          outFmt +
          outputKey;
      } else if (op === "count") {
        typeSpecific = opSel + srcKey + outputKey;
      } else if (op === "group_by") {
        typeSpecific =
          opSel +
          srcKey +
          field(
            "group_field",
            textInput("sf-group-field", step.group_field, "z.B. model"),
          ) +
          field(
            "value_field (optional)",
            textInput("sf-value-field", step.value_field, "z.B. price"),
          ) +
          tgtKey +
          maxIt +
          outputKey;
      } else if (op === "flatten") {
        typeSpecific = opSel + srcKey + outFmt + outputKey;
      } else if (op === "sort") {
        typeSpecific =
          opSel +
          srcKey +
          field(
            "field (optional)",
            textInput("sf-field", step.field, "z.B. price"),
          ) +
          field(
            "reverse",
            `<select class="modal-select" id="sf-reverse"><option value="false" ${!step.reverse ? "selected" : ""}>false</option><option value="true" ${step.reverse ? "selected" : ""}>true</option></select>`,
          ) +
          outputKey;
      } else if (op === "statistics") {
        typeSpecific =
          opSel +
          srcKey +
          field(
            "model_key",
            contextDropdown(
              "sf-model-key",
              step.model_key,
              "z.B. extracted_model",
            ),
          ) +
          field(
            "functions",
            textInput(
              "sf-functions",
              (step.functions || []).join(", "),
              "q1, q3, iqr, lower_bound",
            ),
          ) +
          field(
            "multiplier",
            textInput("sf-multiplier", step.multiplier, "1.5"),
          ) +
          outputKey;
      } else if (op === "json_path") {
        typeSpecific =
          opSel +
          srcKey +
          field("path", textInput("sf-path", step.path, "z.B. symbol")) +
          field("default", textInput("sf-default", step.default, "")) +
          outputKey;
      } else if (op === "xml_extract") {
        typeSpecific =
          opSel +
          srcKey +
          field("xpath", textInput("sf-xpath", step.xpath, "")) +
          field(
            "attribute (optional)",
            textInput("sf-attribute", step.attribute, ""),
          ) +
          field("default", textInput("sf-default", step.default, "")) +
          outputKey;
      } else if (op === "regex_extract") {
        typeSpecific =
          opSel +
          srcKey +
          field("pattern", textInput("sf-pattern", step.pattern, "")) +
          field("group", textInput("sf-group-num", step.group, "1")) +
          field("default", textInput("sf-default", step.default, "")) +
          outputKey;
      } else if (op === "arithmetic") {
        typeSpecific =
          opSel +
          field(
            "expression",
            textInput(
              "sf-expression",
              step.expression,
              "z.B. price / exchange_rate",
            ),
          ) +
          field("round", textInput("sf-round", step.round, "2")) +
          field("default", textInput("sf-default", step.default, "")) +
          outputKey;
      } else if (op === "compare") {
        typeSpecific =
          opSel +
          field(
            "left_key",
            contextDropdown("sf-left-key", step.left_key, "z.B. price_eur"),
          ) +
          field(
            "right_key",
            contextDropdown(
              "sf-right-key",
              step.right_key,
              "z.B. price_stats.lower_bound",
            ),
          ) +
          field(
            "operator",
            `<select class="modal-select" id="sf-operator"><option ${(step.operator || "<=") === "<=" ? "selected" : ""}><=</option><option ${step.operator === "<" ? "selected" : ""}><</option><option ${step.operator === ">=" ? "selected" : ""}>>=</option><option ${step.operator === ">" ? "selected" : ""}>></option><option ${step.operator === "==" ? "selected" : ""}>==</option><option ${step.operator === "!=" ? "selected" : ""}>!=</option></select>`,
          ) +
          field(
            "output_true",
            textInput("sf-output-true", step.output_true, "true"),
          ) +
          field(
            "output_false",
            textInput("sf-output-false", step.output_false, "false"),
          ) +
          outputKey;
      } else {
        typeSpecific = opSel + srcKey + outputKey;
      }
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
  return `<input class="modal-input" id="${id}" value="${String(value ?? "").replace(/"/g, "&quot;")}" placeholder="${placeholder || ""}" />`;
}
function textarea(id, value, minHeight) {
  return `<textarea class="modal-input" id="${id}" style="min-height:${minHeight || 120}px;font-family:var(--mono);font-size:12px;">${value || ""}</textarea>`;
}
function checkbox(id, checked, label) {
  return `<label style="display:flex;align-items:center;gap:6px;font-size:13px;color:var(--text2);cursor:pointer;"><input type="checkbox" id="${id}" ${checked ? "checked" : ""} /> ${label}</label>`;
}
