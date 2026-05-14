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
}
