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
      const op = (step.operation = _val("sf-operation") || "map_field");
      if (op === "array_push") {
        step.value_key = _val("sf-value-key") || "";
        step.group_key = _val("sf-group-key") || "";
        step.target_key = _val("sf-target-key") || "";
        const mi = _val("sf-max-items");
        if (mi) step.max_items = parseInt(mi);
      } else if (op === "map_field") {
        step.source_key = _val("sf-source-key") || "";
        step.field = _val("sf-field") || "";
        const fmt = _val("sf-output-format");
        if (fmt && fmt !== "json_array") step.output_format = fmt;
      } else if (op === "filter") {
        step.source_key = _val("sf-source-key") || "";
        try {
          step.filters = JSON.parse(_val("sf-filters-json") || "[]");
        } catch {
          step.filters = [];
        }
      } else if (op === "first") {
        step.source_key = _val("sf-source-key") || "";
        const def = _val("sf-default");
        if (def) step.default = def;
      } else if (op === "slice") {
        step.source_key = _val("sf-source-key") || "";
        const s = _val("sf-start");
        if (s !== "") step.start = parseInt(s);
        const e = _val("sf-end");
        if (e !== "") step.end = parseInt(e);
      } else if (op === "diff") {
        step.source_key = _val("sf-source-key") || "";
        step.subtract_key = _val("sf-subtract-key") || "";
        const fmt = _val("sf-output-format");
        if (fmt && fmt !== "json_array") step.output_format = fmt;
      } else if (op === "intersect" || op === "union") {
        step.source_key = _val("sf-source-key") || "";
        step.other_key = _val("sf-other-key") || "";
        const fmt = _val("sf-output-format");
        if (fmt && fmt !== "json_array") step.output_format = fmt;
      } else if (op === "list_append") {
        step.value_key = _val("sf-value-key") || "";
        step.target_key = _val("sf-target-key") || "";
        const mi = _val("sf-max-items");
        if (mi) step.max_items = parseInt(mi);
        const fmt = _val("sf-output-format");
        if (fmt && fmt !== "json_array") step.output_format = fmt;
      } else if (op === "count") {
        step.source_key = _val("sf-source-key") || "";
      } else if (op === "group_by") {
        step.source_key = _val("sf-source-key") || "";
        step.group_field = _val("sf-group-field") || "";
        const vf = _val("sf-value-field");
        if (vf) step.value_field = vf;
        const tk = _val("sf-target-key");
        if (tk) step.target_key = tk;
        const mi = _val("sf-max-items");
        if (mi) step.max_items = parseInt(mi);
      } else if (op === "flatten") {
        step.source_key = _val("sf-source-key") || "";
        const fmt = _val("sf-output-format");
        if (fmt && fmt !== "json_array") step.output_format = fmt;
      } else if (op === "sort") {
        step.source_key = _val("sf-source-key") || "";
        const f = _val("sf-field");
        if (f) step.field = f;
        step.reverse = _val("sf-reverse") === "true";
      } else if (op === "statistics") {
        step.source_key = _val("sf-source-key") || "";
        const mk = _val("sf-model-key");
        if (mk) step.model_key = mk;
        const fns = _val("sf-functions");
        if (fns)
          step.functions = fns
            .split(",")
            .map((f) => f.trim())
            .filter(Boolean);
        const mult = _val("sf-multiplier");
        if (mult) step.multiplier = parseFloat(mult);
      } else if (op === "json_path") {
        step.source_key = _val("sf-source-key") || "";
        step.path = _val("sf-path") || "";
        const def = _val("sf-default");
        if (def) step.default = def;
      } else if (op === "xml_extract") {
        step.source_key = _val("sf-source-key") || "";
        step.xpath = _val("sf-xpath") || "";
        const attr = _val("sf-attribute");
        if (attr) step.attribute = attr;
        const def = _val("sf-default");
        if (def) step.default = def;
      } else if (op === "regex_extract") {
        step.source_key = _val("sf-source-key") || "";
        step.pattern = _val("sf-pattern") || "";
        const grp = _val("sf-group-num");
        if (grp) step.group = parseInt(grp);
        const def = _val("sf-default");
        if (def) step.default = def;
      } else if (op === "arithmetic") {
        step.expression = _val("sf-expression") || "";
        const rnd = _val("sf-round");
        if (rnd !== "" && rnd !== undefined) step.round = parseInt(rnd);
        const def = _val("sf-default");
        if (def) step.default = def;
      } else if (op === "compare") {
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
