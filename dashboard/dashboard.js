function getBase() {
  return window.__API_URL__ || "";
}

let STEP_TYPES = [];
let agentsData = [];
let scrapersData = [];
let currentAgentId = null;
let currentScraperId = null;
let currentMemSubjectId = null;
let memMode = "users";
let memData = { users: [], groups: [] };
let _agentMemCache = {};
let _agentDataCache = {};
let _memCache = {};
let _confirmCallback = null;
let _usagePage = 0;
const _usageLimit = 10;

const _isMobile = () => window.innerWidth <= 768;

function api(path, opts) {
  return fetch(getBase() + path, opts).then((r) => {
    if (!r.ok) throw new Error(r.status);
    return r.json();
  });
}

function fmt(iso) {
  if (!iso) return "—";
  const d = new Date(iso);
  return (
    d.toLocaleDateString("de-DE", {
      day: "2-digit",
      month: "2-digit",
      year: "2-digit",
    }) +
    " " +
    d.toLocaleTimeString("de-DE", { hour: "2-digit", minute: "2-digit" })
  );
}

function fmtNum(n) {
  return Number(n).toLocaleString("de-DE");
}
function fmtCost(usd) {
  if (usd === 0) return "—";
  if (usd < 0.01) return `$${usd.toFixed(4)}`;
  return `$${usd.toFixed(2)}`;
}

function toast(msg, isError) {
  const el = document.getElementById("toast");
  el.textContent = msg;
  el.className = "toast show" + (isError ? " error" : "");
  setTimeout(() => (el.className = "toast"), 2500);
}

function openModal(title, bodyHtml, actions) {
  document.getElementById("modal-title").textContent = title;
  document.getElementById("modal-body").innerHTML = bodyHtml;
  document.getElementById("modal-actions").innerHTML = actions;
  document.getElementById("modal-overlay").classList.add("open");
}

function closeModal(e) {
  if (!e || e.target === document.getElementById("modal-overlay")) {
    document.getElementById("modal-overlay").classList.remove("open");
  }
}

function confirmModal(message, onConfirm) {
  _confirmCallback = onConfirm;
  openModal(
    "Bestätigen",
    `<div style="font-size:13px;color:var(--text);line-height:1.6;">${message}</div>`,
    `<button class="btn" onclick="closeModal()">Abbrechen</button>
     <button class="btn btn-danger" onclick="closeModal();_confirmCallback()">Loeschen</button>`,
  );
}

function showDetail(detailEl, sidebarEl, title) {
  if (_isMobile()) {
    sidebarEl.classList.add("hidden");
    detailEl.classList.add("visible");
    document.getElementById("topbar-back").classList.add("visible");
    document.getElementById("topbar-dot").style.display = "none";
    document.getElementById("topbar-name").textContent = title;
  }
}

function goBack() {
  const activeSection = document.querySelector(".section.active");
  if (!activeSection) return;
  const sidebar = activeSection.querySelector(".sidebar");
  const detail = activeSection.querySelector(".detail");
  if (sidebar) sidebar.classList.remove("hidden");
  if (detail) detail.classList.remove("visible");
  document.getElementById("topbar-back").classList.remove("visible");
  document.getElementById("topbar-dot").style.display = "";
  document.getElementById("topbar-name").textContent = "Bob Dashboard";
}

function switchSection(name) {
  document
    .querySelectorAll(".section")
    .forEach((s) => s.classList.remove("active"));
  document.getElementById("section-" + name).classList.add("active");
  document
    .querySelectorAll(".nav-btn[data-section]")
    .forEach((b) => b.classList.toggle("active", b.dataset.section === name));
  document
    .querySelectorAll(".bottom-nav-item")
    .forEach((b) => b.classList.toggle("active", b.dataset.section === name));
  goBack();
}

const STEP_TYPE_GROUPS = {
  Routing: ["router_match", "router_llm"],
  LLM: ["llm_extract", "llm_decide", "llm_summarize"],
  Datenzugriff: [
    "web_search",
    "finance",
    "http_fetch",
    "state_read",
    "state_write",
    "state_read_external",
    "state_write_external",
    "data_read",
    "data_write",
    "data_read_external",
    "data_write_external",
  ],
  Transformation: ["transform"],
  Koordination: ["trigger_agent", "notify_user"],
};

const STEP_TYPE_COLORS = {
  router_match: "var(--amber)",
  router_llm: "var(--amber)",
  llm_extract: "var(--accent)",
  llm_decide: "var(--accent)",
  llm_summarize: "var(--accent)",
  web_search: "var(--blue)",
  finance: "var(--blue)",
  http_fetch: "var(--blue)",
  state_read: "var(--text3)",
  state_write: "var(--text3)",
  state_read_external: "var(--text3)",
  state_write_external: "var(--text3)",
  data_read: "var(--text3)",
  data_write: "var(--text3)",
  data_read_external: "var(--text3)",
  data_write_external: "var(--text3)",
  transform: "var(--blue)",
  trigger_agent: "var(--red)",
  notify_user: "var(--red)",
};

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
    case "http_fetch":
      parts.push(step.url || step.url_template || "?");
      if (step.method && step.method !== "GET") parts.push(step.method);
      break;
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
    case "transform":
      parts.push(`op: ${step.operation || "?"}`);
      if (step.source_key) parts.push(`<- ${step.source_key}`);
      if (step.target_key) parts.push(`-> ${step.target_key}`);
      break;
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
    ${field("only_if_route", textInput("sf-route", onlyIfRoute, "z.B. new_listing"))}
    ${field("only_if_key.key", textInput("sf-only-if-key", onlyIfKeyKey, "z.B. is_bargain"))}
    ${field("only_if_key.value", textInput("sf-only-if-val", onlyIfKeyVal, "z.B. true"))}
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
    "source_key",
    textInput("sf-source-key", step.source_key, "z.B. baselines"),
  );
  const targetKey = field(
    "target_key",
    textInput("sf-target-key", step.target_key, "z.B. baselines"),
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
    "key",
    textInput("sf-key", step.key, "z.B. price_baselines"),
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
          textInput("sf-ticker-key", step.ticker_key, "selected_ticker"),
        ) + outputKey;
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
      const op = step.operation || "array_push";
      typeSpecific =
        field(
          "operation",
          `<select class="modal-select" id="sf-operation" onchange="onTransformOpChange()">${transformOpOptions(op)}</select>`,
        ) +
        sourceKey +
        `<div id="sf-value-key-wrap">${field("value_key", textInput("sf-value-key", step.value_key, "z.B. price_eur"))}</div>` +
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
        `<div id="sf-operator-wrap">${field("operator", `<select class="modal-select" id="sf-operator"><option ${(step.operator || "<=") === ">=" ? "selected" : ""}><=</option><option ${step.operator === "<" ? "selected" : ""}><</option><option ${step.operator === ">=" ? "selected" : ""}>>=</option><option ${step.operator === ">" ? "selected" : ""}>>></option><option ${step.operator === "==" ? "selected" : ""}>==</option><option ${step.operator === "!=" ? "selected" : ""}>!=</option></select>`)}</div>` +
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
  document.getElementById("modal-body").innerHTML =
    buildStepFormBody(currentStep);
  if (type === "transform") onTransformOpChange();
}

function onTransformOpChange() {
  const op = document.getElementById("sf-operation")?.value;
  const show = (id, v) => {
    const el = document.getElementById(id);
    if (el) el.style.display = v ? "" : "none";
  };
  show("sf-value-key-wrap", op === "array_push");
  show("sf-group-key-wrap", op === "array_push");
  show("sf-max-items-wrap", op === "array_push");
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
  show("sf-target-key", op === "array_push");
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
      if (step.operation === "array_push") {
        step.value_key = _val("sf-value-key") || "";
        step.group_key = _val("sf-group-key") || "";
        const mi = _val("sf-max-items");
        if (mi) step.max_items = parseInt(mi);
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
  const step = (a.steps || [])[idx];
  openModal(
    "Step bearbeiten",
    buildStepFormBody(step),
    `<button class="btn" onclick="closeModal()">Abbrechen</button><button class="btn btn-accent" onclick="saveStep(${agentId},${idx})">Speichern</button>`,
  );
  if (step.type === "transform") onTransformOpChange();
}

async function saveStep(agentId, idx) {
  const a = agentsData.find((x) => x.id === agentId);
  const steps = [...(a.steps || [])];
  steps[idx] = _readStepFromForm();
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
  openModal(
    "Neuen Step hinzufuegen",
    `<div class="modal-field"><div class="modal-label">Position (leer = ans Ende)</div><input class="modal-input" id="sf-position" type="number" placeholder="0 = Anfang" /></div>` +
      buildStepFormBody({ id: "", type: "llm_extract" }),
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

async function loadMemory() {
  try {
    const [users, groups] = await Promise.all([
      api("/api/users"),
      api("/api/groups"),
    ]);
    memData.users = users;
    memData.groups = groups;
    renderMemoryList();
  } catch {
    document.getElementById("memory-list").innerHTML =
      '<div style="font-size:13px;color:var(--red);padding:8px;">Fehler.</div>';
  }
}

function switchMemMode(mode) {
  memMode = mode;
  currentMemSubjectId = null;
  document
    .getElementById("mem-users-btn")
    .classList.toggle("active", mode === "users");
  document
    .getElementById("mem-groups-btn")
    .classList.toggle("active", mode === "groups");
  document.getElementById("memory-detail").innerHTML =
    '<div class="empty-state">Eintrag auswaehlen</div>';
  document.getElementById("memory-detail").classList.remove("visible");
  document.getElementById("memory-sidebar").classList.remove("hidden");
  document.getElementById("topbar-back").classList.remove("visible");
  document.getElementById("topbar-dot").style.display = "";
  document.getElementById("topbar-name").textContent = "Bob Dashboard";
  renderMemoryList();
}

function renderMemoryList(filter) {
  const items = memData[memMode] || [];
  const q = (
    filter ||
    document.getElementById("memory-search").value ||
    ""
  ).toLowerCase();
  const filtered = q
    ? items.filter((i) => {
        const name =
          memMode === "users"
            ? (
                (i.first_name || "") +
                " " +
                (i.last_name || "") +
                " " +
                (i.username || "")
              ).toLowerCase()
            : (i.title || "").toLowerCase();
        return name.includes(q) || String(i.id).includes(q);
      })
    : items;
  if (!filtered.length) {
    document.getElementById("memory-list").innerHTML =
      '<div style="font-size:13px;color:var(--text3);padding:8px;">Keine Eintraege.</div>';
    return;
  }
  document.getElementById("memory-list").innerHTML = filtered
    .map((i) => {
      const name =
        memMode === "users"
          ? [i.first_name, i.last_name].filter(Boolean).join(" ") ||
            i.username ||
            String(i.id)
          : i.title || String(i.id);
      const sub =
        memMode === "users"
          ? i.username
            ? "@" + i.username
            : i.timezone
          : fmt(i.first_seen_at);
      return `<div class="sidebar-item" data-id="${i.id}" onclick="selectMemItem(${i.id})"><div class="si-name">${name}</div><div class="si-meta">${sub || ""}</div></div>`;
    })
    .join("");
}

async function selectMemItem(id) {
  currentMemSubjectId = id;
  document
    .querySelectorAll("#memory-list .sidebar-item")
    .forEach((el) =>
      el.classList.toggle("active", Number(el.dataset.id) === id),
    );
  const item = memData[memMode].find((i) => i.id === id);
  const name =
    memMode === "users"
      ? [item.first_name, item.last_name].filter(Boolean).join(" ") ||
        item.username ||
        String(id)
      : item.title || String(id);
  showDetail(
    document.getElementById("memory-detail"),
    document.getElementById("memory-sidebar"),
    name,
  );
  await renderMemDetail(id);
}

async function renderMemDetail(id) {
  const detail = document.getElementById("memory-detail");
  detail.innerHTML = '<div class="empty-state">Lade...</div>';
  try {
    const endpoint =
      memMode === "users"
        ? "/api/users/" + id + "/memories"
        : "/api/groups/" + id + "/memories";
    const mems = await api(endpoint);
    _memCache[id] = mems;
    const availableTypes =
      memMode === "users"
        ? ["user", "reflection"]
        : ["group", "bot", "reflection"];
    const byType = {};
    availableTypes.forEach((t) => (byType[t] = []));
    mems.forEach((m, i) => {
      if (byType[m.type]) byType[m.type].push({ ...m, _globalIdx: i });
    });

    const observations =
      memMode === "groups"
        ? await api("/api/chats/" + id + "/observations").catch(() => [])
        : [];

    detail.innerHTML = `
      <div class="action-bar"><button class="btn btn-accent" onclick="addMemory(${id})">+ Neue Memory</button></div>
      ${availableTypes
        .map(
          (type) => `
        <div class="detail-block">
          <div class="detail-block-title">${type} (${byType[type].length})</div>
          <div class="mem-list">
            ${byType[type]
              .map(
                (m) => `
              <div class="mem-row">
                <span class="badge badge-${m.type}" style="flex-shrink:0;margin-top:2px;">${m.type}</span>
                <div class="mem-text">${m.content}</div>
                <div class="mem-date">${fmt(m.created_at)}</div>
                <div class="mem-actions">
                  <button class="btn btn-sm" data-subject="${id}" data-idx="${m._globalIdx}" data-action="edit-mem">Edit</button>
                  <button class="btn btn-sm btn-danger" data-subject="${id}" data-idx="${m._globalIdx}" data-action="delete-mem">Del</button>
                </div>
              </div>`,
              )
              .join("")}
            ${!byType[type].length ? '<div style="font-size:13px;color:var(--text3);padding:6px 0;">Keine Eintraege.</div>' : ""}
          </div>
        </div>`,
        )
        .join("")}
      ${
        memMode === "groups"
          ? `
        <hr class="divider">
        <div class="detail-block">
          <div class="detail-block-title">Observations (${observations.length})</div>
          <div class="mem-list">
            ${observations.length ? observations.map((o) => _renderObservation(o, id)).join("") : '<div style="font-size:13px;color:var(--text3);padding:6px 0;">Noch keine Observations.</div>'}
          </div>
        </div>`
          : ""
      }`;
    detail.removeEventListener("click", handleMemDetailClick);
    detail.addEventListener("click", handleMemDetailClick);
  } catch {
    detail.innerHTML =
      '<div style="font-size:13px;color:var(--red);padding:1rem;">Fehler.</div>';
  }
}

function _renderObservation(o, chatId) {
  const compressedBadge = o.is_compressed
    ? '<span class="badge badge-pipeline" style="flex-shrink:0;">verdichtet</span>'
    : '<span class="badge badge-active" style="flex-shrink:0;">aktuell</span>';
  const priorityChar = {
    red: "\uD83D\uDD34",
    yellow: "\uD83D\uDFE1",
    green: "\uD83D\uDFE2",
  };
  const lines = o.content
    .split("\n")
    .map((line) => {
      const color = line.startsWith(priorityChar.red)
        ? "var(--red)"
        : line.startsWith(priorityChar.yellow)
          ? "var(--amber)"
          : "var(--text3)";
      return `<div style="font-size:12px;color:${color};line-height:1.6;font-family:var(--mono);">${line}</div>`;
    })
    .join("");
  return `
    <div class="mem-row" style="flex-direction:column;align-items:flex-start;gap:6px;padding:12px 0;">
      <div style="display:flex;align-items:center;gap:8px;width:100%;">
        <div class="mem-date" style="flex-shrink:0;">${fmt(o.observed_at)}</div>
        ${compressedBadge}
        <div style="flex:1;"></div>
        <div class="mem-actions" style="opacity:1;">
          <button class="btn btn-sm btn-danger" data-obs-id="${o.id}" data-chat-id="${chatId}" data-action="delete-obs">Del</button>
        </div>
      </div>
      <div style="width:100%;">${lines}</div>
    </div>`;
}

function handleMemDetailClick(e) {
  const btn = e.target.closest("[data-action]");
  if (!btn) return;
  const action = btn.dataset.action;

  if (action === "delete-obs") {
    const obsId = Number(btn.dataset.obsId);
    const chatId = Number(btn.dataset.chatId);
    confirmModal("Observation loeschen?", () => {
      api(`/api/chats/${chatId}/observations/${obsId}`, { method: "DELETE" })
        .then(() => {
          toast("Geloescht.");
          renderMemDetail(chatId);
        })
        .catch(() => toast("Fehler.", true));
    });
    return;
  }

  const subjectId = Number(btn.dataset.subject);
  const idx = Number(btn.dataset.idx);
  const mem = _memCache[subjectId]?.[idx];
  if (!mem) return;
  const endpoint =
    memMode === "users"
      ? "/api/users/" + subjectId + "/memories"
      : "/api/groups/" + subjectId + "/memories";
  if (action === "edit-mem") {
    openModal(
      "Memory bearbeiten",
      `<div class="modal-field"><div class="modal-label">Typ</div><input class="modal-input" value="${mem.type}" disabled /></div><div class="modal-field"><div class="modal-label">Inhalt</div><textarea class="modal-input" id="edit-mem-content" style="min-height:100px;">${mem.content}</textarea></div>`,
      `<button class="btn" onclick="closeModal()">Abbrechen</button><button class="btn btn-accent" onclick="saveMemory(${subjectId},${idx})">Speichern</button>`,
    );
  } else if (action === "delete-mem") {
    confirmModal("Memory loeschen?", () => {
      api(endpoint, {
        method: "DELETE",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ content: mem.content, subject_type: mem.type }),
      })
        .then(() => {
          toast("Geloescht.");
          renderMemDetail(subjectId);
        })
        .catch(() => toast("Fehler.", true));
    });
  }
}

async function saveMemory(subjectId, idx) {
  const mem = _memCache[subjectId][idx];
  const newContent = document.getElementById("edit-mem-content").value.trim();
  const endpoint =
    memMode === "users"
      ? "/api/users/" + subjectId + "/memories"
      : "/api/groups/" + subjectId + "/memories";
  try {
    await api(endpoint, {
      method: "PATCH",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        old_content: mem.content,
        new_content: newContent,
        subject_type: mem.type,
      }),
    });
    closeModal();
    toast("Gespeichert.");
    renderMemDetail(subjectId);
  } catch {
    toast("Fehler.", true);
  }
}

function addMemory(subjectId) {
  const types =
    memMode === "users"
      ? ["user", "reflection"]
      : ["group", "bot", "reflection"];
  openModal(
    "Memory hinzufuegen",
    `<div class="modal-field"><div class="modal-label">Typ</div><select class="modal-select" id="new-mem-type">${types.map((t) => `<option value="${t}">${t}</option>`).join("")}</select></div><div class="modal-field"><div class="modal-label">Inhalt</div><textarea class="modal-input" id="new-mem-content" placeholder="Inhalt..."></textarea></div>`,
    `<button class="btn" onclick="closeModal()">Abbrechen</button><button class="btn btn-accent" onclick="saveNewMemory(${subjectId})">Speichern</button>`,
  );
}
async function saveNewMemory(subjectId) {
  const content = document.getElementById("new-mem-content").value.trim();
  const subject_type = document.getElementById("new-mem-type").value;
  if (!content) return;
  const endpoint =
    memMode === "users"
      ? "/api/users/" + subjectId + "/memories"
      : "/api/groups/" + subjectId + "/memories";
  try {
    await api(endpoint, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ content, subject_type }),
    });
    closeModal();
    toast("Gespeichert.");
    renderMemDetail(subjectId);
  } catch {
    toast("Fehler.", true);
  }
}

document
  .getElementById("usage-tabs")
  .querySelectorAll(".agent-tab")
  .forEach((tab) =>
    tab.addEventListener("click", function () {
      document
        .querySelectorAll("#usage-tabs .agent-tab")
        .forEach((t) => t.classList.remove("active"));
      ["models", "caller", "history"].forEach((n) =>
        document.getElementById("usage-tab-" + n).classList.remove("active"),
      );
      this.classList.add("active");
      document
        .getElementById("usage-tab-" + this.dataset.tab)
        .classList.add("active");
    }),
  );

async function loadUsage() {
  try {
    const u = await api("/api/usage");
    const totalCost = u.by_model.reduce((s, m) => s + m.estimated_cost_usd, 0);
    document.getElementById("usage-tab-models").innerHTML = `
      <div class="metrics-grid">
        <div class="metric-card"><div class="metric-label">input tokens</div><div class="metric-value">${fmtNum(u.total_input)}</div></div>
        <div class="metric-card"><div class="metric-label">output tokens</div><div class="metric-value">${fmtNum(u.total_output)}</div></div>
        <div class="metric-card"><div class="metric-label">geschaetzte kosten</div><div class="metric-value">${fmtCost(totalCost)}</div></div>
      </div>
      <div class="detail-block"><div class="detail-block-title">Nach Modell</div><div style="display:flex;flex-direction:column;">
        ${u.by_model.map((m) => `<div style="display:grid;grid-template-columns:1fr auto auto auto;align-items:baseline;gap:12px;padding:8px 0;border-bottom:0.5px solid var(--border);"><div style="font-family:var(--mono);font-size:11px;color:var(--text2);overflow:hidden;text-overflow:ellipsis;white-space:nowrap;">${m.model}</div><div style="font-family:var(--mono);font-size:11px;color:var(--text3);white-space:nowrap;">up${fmtNum(m.input)} down${fmtNum(m.output)}</div><div style="font-family:var(--mono);font-size:11px;color:var(--text3);white-space:nowrap;">${m.calls} calls</div><div style="font-family:var(--mono);font-size:11px;color:var(--accent);white-space:nowrap;">${fmtCost(m.estimated_cost_usd)}</div></div>`).join("")}
        ${!u.by_model.length ? '<div style="font-size:13px;color:var(--text3);padding:8px 0;">Noch keine Daten.</div>' : ""}
      </div></div>`;
    const maxTokens = u.by_caller.reduce(
      (m, c) => Math.max(m, c.input + c.output),
      1,
    );
    document.getElementById("usage-tab-caller").innerHTML =
      `<div class="detail-block"><div class="detail-block-title">Nach Caller</div><div class="caller-list">${u.by_caller
        .map((c) => {
          const t = c.input + c.output;
          const pct = Math.round((t / maxTokens) * 100);
          return `<div class="caller-row"><div class="caller-name">${c.caller}</div><div class="caller-bar-bg"><div class="caller-bar-fg" style="width:${pct}%"></div></div><div class="caller-tokens">${fmtNum(t)}</div></div>`;
        })
        .join("")}</div></div>`;
  } catch {
    document.getElementById("usage-tab-models").innerHTML =
      '<div style="font-size:13px;color:var(--red);padding:1rem;">Fehler.</div>';
  }
  await loadUsageHistory(0);
}

async function loadUsageHistory(page) {
  _usagePage = page;
  const el = document.getElementById("usage-tab-history");
  el.innerHTML = '<div class="empty-state">Lade...</div>';
  try {
    const h = await api(
      "/api/usage/history?page=" + page + "&limit=" + _usageLimit,
    );
    const totalPages = Math.ceil(h.total / _usageLimit);
    el.innerHTML = `<div class="detail-block"><div class="detail-block-title"><span>Calls (${fmtNum(h.total)})</span><span style="font-size:11px;color:var(--text3);">Seite ${page + 1} / ${totalPages}</span></div><div style="display:flex;flex-direction:column;">${h.items.map((item) => `<div style="display:grid;grid-template-columns:1fr auto auto auto auto;align-items:baseline;gap:8px;padding:7px 0;border-bottom:0.5px solid var(--border);"><div style="font-family:var(--mono);font-size:11px;color:var(--text2);overflow:hidden;text-overflow:ellipsis;white-space:nowrap;">${item.caller}</div><div style="font-family:var(--mono);font-size:10px;color:var(--text3);white-space:nowrap;">${item.model || "--"}</div><div style="font-family:var(--mono);font-size:11px;color:var(--accent);">up${fmtNum(item.input_tokens)}</div><div style="font-family:var(--mono);font-size:11px;color:var(--blue);">down${fmtNum(item.output_tokens)}</div><div style="font-family:var(--mono);font-size:10px;color:var(--text3);">${fmt(item.created_at)}</div></div>`).join("")}</div><div style="display:flex;justify-content:space-between;align-items:center;margin-top:12px;"><button class="btn btn-sm" onclick="loadUsageHistory(${page - 1})" ${page === 0 ? 'disabled style="opacity:0.3;"' : ""}>Zurueck</button><button class="btn btn-sm" onclick="loadUsageHistory(${page + 1})" ${page >= totalPages - 1 ? 'disabled style="opacity:0.3;"' : ""}>Weiter</button></div></div>`;
  } catch {
    el.innerHTML =
      '<div style="font-size:13px;color:var(--red);padding:1rem;">Fehler.</div>';
  }
}

async function loadRegistry() {
  const el = document.getElementById("registry-detail");
  el.innerHTML = '<div class="empty-state">Lade...</div>';
  try {
    const data = await api("/api/registry");
    const routingHtml = data.routing
      .map(
        (r) =>
          `<div style="display:grid;grid-template-columns:140px 1fr auto;align-items:center;gap:12px;padding:7px 0;border-bottom:0.5px solid var(--border);"><div style="font-family:var(--mono);font-size:11px;color:var(--accent);">${r.capability}</div><div style="font-family:var(--mono);font-size:11px;color:var(--text2);">${r.model}${r.is_local ? ' <span style="color:var(--amber);">[local]</span>' : ""}</div><div style="font-family:var(--mono);font-size:10px;color:var(--text3);">${r.input_cost_per_mtok === 0 ? "free" : "$" + r.input_cost_per_mtok + "/MTok"}</div></div>`,
      )
      .join("");
    const byProvider = {};
    data.models.forEach((m) => {
      if (!byProvider[m.provider]) byProvider[m.provider] = [];
      byProvider[m.provider].push(m);
    });
    const modelsHtml = Object.entries(byProvider)
      .map(
        ([provider, models]) =>
          `<div class="ns-block"><div class="ns-label">${provider}</div>${models.map((m) => `<div class="data-card" style="opacity:${m.is_available ? 1 : 0.45};"><div class="data-card-header"><div style="display:flex;align-items:center;gap:8px;"><span style="font-size:13px;font-weight:500;">${m.display_name}</span><span class="badge ${m.is_available ? "badge-active" : "badge-inactive"}">${m.is_available ? "verfuegbar" : "nicht verfuegbar"}</span>${m.is_local ? '<span class="badge badge-pipeline">local</span>' : ""}</div><div style="font-family:var(--mono);font-size:10px;color:var(--text3);">${m.input_cost_per_mtok === 0 ? "kostenlos" : "$" + m.input_cost_per_mtok + " / $" + m.output_cost_per_mtok + " MTok"}</div></div><div style="font-family:var(--mono);font-size:10px;color:var(--text3);margin-bottom:6px;">${m.api_model_name} . ctx: ${m.context_window ? fmtNum(m.context_window) : "--"} . out: ${m.max_output_tokens ? fmtNum(m.max_output_tokens) : "--"}</div><div style="display:flex;flex-wrap:wrap;gap:4px;">${m.capabilities.map((c) => `<span class="badge badge-cap">${c}</span>`).join("")}</div>${m.notes ? `<div style="font-size:11px;color:var(--text3);margin-top:6px;">${m.notes}</div>` : ""}</div>`).join("")}</div>`,
      )
      .join("");
    el.innerHTML = `<div class="detail-block"><div class="detail-block-title">Aktives Routing</div>${routingHtml || '<div style="font-size:13px;color:var(--text3);">Keine Daten.</div>'}</div><hr class="divider"><div class="detail-block"><div class="detail-block-title">Alle Modelle (${data.models.length})</div>${modelsHtml}</div>`;
  } catch {
    el.innerHTML =
      '<div style="font-size:13px;color:var(--red);padding:1rem;">Fehler.</div>';
  }
}

async function loadTriggers() {
  const el = document.getElementById("triggers-detail");
  try {
    const triggers = await api("/api/triggers");
    if (!triggers.length) {
      el.innerHTML = '<div class="empty-state">Keine Trigger.</div>';
      return;
    }
    el.innerHTML = triggers
      .map(
        (t) =>
          `<div class="trigger-item"><div class="trigger-header"><span class="trigger-name">${t.target_agent_name}</span><span class="badge ${t.processed_at ? "badge-done" : "badge-pending"}">${t.processed_at ? "verarbeitet" : "ausstehend"}</span></div><div class="trigger-meta">geplant ${fmt(t.scheduled_for)} . erstellt ${fmt(t.created_at)}</div>${Object.keys(t.payload || {}).length ? `<div class="trigger-meta" style="margin-top:2px;">payload: ${JSON.stringify(t.payload)}</div>` : ""}</div>`,
      )
      .join("");
  } catch {
    el.innerHTML =
      '<div style="font-size:13px;color:var(--red);padding:1rem;">Fehler.</div>';
  }
}

async function loadScrapers() {
  try {
    scrapersData = await api("/api/scrapers");
    renderScrapersList();
  } catch {
    document.getElementById("scrapers-list").innerHTML =
      '<div style="font-size:13px;color:var(--red);padding:8px;">Backend nicht erreichbar.</div>';
  }
}
function renderScrapersList() {
  if (!scrapersData.length) {
    document.getElementById("scrapers-list").innerHTML =
      '<div style="font-size:13px;color:var(--text3);padding:8px;">Keine Scraper-Configs.</div>';
    return;
  }
  document.getElementById("scrapers-list").innerHTML = scrapersData
    .map(
      (s) =>
        `<div class="sidebar-item" data-id="${s.id}" onclick="selectScraper(${s.id})"><div class="si-name">${s.platform}<span class="badge ${s.is_active ? "badge-active" : "badge-inactive"}">${s.is_active ? "aktiv" : "inaktiv"}</span><span class="badge badge-pipeline">${s.category}</span></div><div class="si-meta">${s.query} -> ${s.target_agent}</div></div>`,
    )
    .join("");
}

async function selectScraper(id) {
  currentScraperId = id;
  document
    .querySelectorAll("#scrapers-list .sidebar-item")
    .forEach((el) =>
      el.classList.toggle("active", Number(el.dataset.id) === id),
    );
  const s = scrapersData.find((x) => x.id === id);
  const detail = document.getElementById("scrapers-detail");
  detail.innerHTML = '<div class="empty-state">Lade...</div>';
  showDetail(detail, document.getElementById("scrapers-sidebar"), s.query);
  try {
    const listings = await api(
      `/api/listings?category=${encodeURIComponent(s.category)}&platform=${encodeURIComponent(s.platform)}&limit=50`,
    );
    const intervalDisplay =
      s.poll_interval_seconds >= 3600
        ? `${s.poll_interval_seconds / 3600}h`
        : `${s.poll_interval_seconds / 60}min`;
    const filtersDisplay = Object.keys(s.filters || {}).length
      ? Object.entries(s.filters)
          .map(([k, v]) => `${k}: ${v}`)
          .join(", ")
      : "--";
    detail.innerHTML = `
      <div class="action-bar">${s.is_active ? `<button class="btn btn-danger" onclick="stopScraper(${id})">Stoppen</button>` : ""}</div>
      <div class="detail-block"><div class="detail-block-title">Konfiguration</div>
        <div class="detail-mono">Plattform: ${s.platform}</div><div class="detail-mono">Kategorie: ${s.category}</div>
        <div class="detail-mono">Query: ${s.query}</div><div class="detail-mono">Ziel-Agent: ${s.target_agent}</div>
        <div class="detail-mono">Intervall: ${intervalDisplay}</div><div class="detail-mono">Filter: ${filtersDisplay}</div>
        <div class="detail-mono">Letzter Lauf: ${fmt(s.last_scraped_at)}</div></div>
      <hr class="divider">
      <div class="detail-block"><div class="detail-block-title">Listings (${listings.length})</div><div style="display:flex;flex-direction:column;">
        ${listings.length ? listings.map((l) => `<div class="data-card"><div class="data-card-header"><div style="display:flex;align-items:center;gap:8px;flex-wrap:wrap;min-width:0;"><a href="${l.url}" target="_blank" style="font-size:13px;font-weight:500;color:var(--accent);text-decoration:none;word-break:break-word;">${l.title}</a>${l.price !== null ? `<span class="badge badge-active">${l.price} ${l.currency || ""}</span>` : ""}${l.condition ? `<span class="badge badge-pipeline">${l.condition}</span>` : ""}</div><button class="btn btn-sm btn-danger" onclick="deleteListing(${l.id},${id})">Del</button></div><div class="detail-mono" style="margin-top:4px;">${l.location ? l.location + " . " : ""}${l.seller_name || ""}${l.seller_rating ? " (" + l.seller_rating + ")" : " "} . ${fmt(l.first_seen_at)}</div></div>`).join("") : '<div style="font-size:13px;color:var(--text3);padding:8px 0;">Noch keine Listings.</div>'}
      </div></div>`;
  } catch {
    detail.innerHTML =
      '<div style="font-size:13px;color:var(--red);padding:1rem;">Fehler.</div>';
  }
}

async function stopScraper(id) {
  confirmModal("Scraper wirklich stoppen?", async () => {
    try {
      await api("/api/scrapers/" + id, { method: "DELETE" });
      toast("Scraper gestoppt.");
      goBack();
      await loadScrapers();
    } catch {
      toast("Fehler.", true);
    }
  });
}
async function deleteListing(listingId, scraperId) {
  confirmModal("Listing loeschen?", async () => {
    try {
      await api("/api/listings/" + listingId, { method: "DELETE" });
      toast("Geloescht.");
      selectScraper(scraperId);
    } catch {
      toast("Fehler.", true);
    }
  });
}

async function loadAll() {
  STEP_TYPES = await api("/api/step-types").catch(() => []);
  loadAgents();
  loadMemory();
  loadScrapers();
  loadUsage();
  loadRegistry();
  loadTriggers();
}
loadAll();
