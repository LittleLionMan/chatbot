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
let monitorsData = [];
let currentMonitorId = null;
let _stepContextKeys = [];
let _stepRouteKeys = [];
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
  LLM: ["llm_extract", "llm_decide", "llm_summarize", "llm_analyze"],
  Datenzugriff: [
    "web_search",
    "finance",
    "finance_search",
    "http_fetch",
    "xlsx_fetch",
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
  llm_analyze: "var(--accent)",
  llm_analyze: "var(--accent)",
  web_search: "var(--blue)",
  finance: "var(--blue)",
  finance_search: "var(--blue)",
  http_fetch: "var(--blue)",
  xlsx_fetch: "var(--blue)",
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
