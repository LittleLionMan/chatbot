function getBase() {
  return window.__API_URL__ || "";
}

const CAPABILITIES = [
  "fast",
  "balanced",
  "search",
  "reasoning",
  "deep_reasoning",
  "coding",
  "multimodal",
  "long_context",
];

let memMode = "users";
let memData = { users: [], groups: [] };
let agentsData = [];
let currentAgentId = null;
let currentMemSubjectId = null;
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
     <button class="btn btn-danger" onclick="closeModal();_confirmCallback()">Löschen</button>`,
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

document.querySelectorAll(".nav-btn[data-section]").forEach((btn) => {
  btn.addEventListener("click", function () {
    switchSection(this.dataset.section);
  });
});
document.querySelectorAll(".bottom-nav-item").forEach((btn) => {
  btn.addEventListener("click", function () {
    switchSection(this.dataset.section);
  });
});
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
  } catch (e) {
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
    .map(
      (a) => `
    <div class="sidebar-item" data-id="${a.id}" onclick="selectAgent(${a.id})">
      <div class="si-name">${a.name}
        <span class="badge ${a.is_active ? "badge-active" : "badge-inactive"}">${a.is_active ? "aktiv" : "inaktiv"}</span>
        ${a.pipeline && a.pipeline.length ? '<span class="badge badge-pipeline">pipeline</span>' : ""}
      </div>
      <div class="si-meta">${a.type || "—"} · ${a.work_capability || "balanced"} · ${a.schedule}</div>
    </div>
  `,
    )
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
  detail.innerHTML = '<div class="empty-state">Lade…</div>';
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

    const capabilityOptions = CAPABILITIES.map(
      (c) =>
        `<option value="${c}" ${c === a.work_capability ? "selected" : ""}>${c}</option>`,
    ).join("");

    const allSteps = [
      ...(a.pipeline || []).map((s) => ({ ...s, _section: "pipeline" })),
      ...(a.pipeline_after_template || []).map((s) => ({
        ...s,
        _section: "pipeline_after_template",
      })),
    ];
    const hasTemplate =
      a.pipeline_template && typeof a.pipeline_template === "object";
    const totalSteps = allSteps.length + (hasTemplate ? 1 : 0);

    const renderStep = (step, i, section) => `
      <div class="pipeline-step">
        <div class="pipeline-step-header">
          <span class="pipeline-step-id">${step.id}${step.is_router ? " 🔀" : ""}</span>
          <span class="badge badge-cap">${step.capability}</span>
          ${step.only_if_route ? `<span class="badge badge-pipeline">${Array.isArray(step.only_if_route) ? step.only_if_route.join("|") : step.only_if_route}</span>` : ""}
          <button class="btn btn-sm" onclick="editPipelineStep(${id}, ${i}, '${section}')">Bearbeiten</button>
          <button class="btn btn-sm btn-danger" onclick="deletePipelineStep(${id}, ${i}, '${section}')">Löschen</button>
        </div>
        <div class="pipeline-step-prompt">${step.prompt_template.slice(0, 120)}${step.prompt_template.length > 120 ? "…" : ""}</div>
        <div class="pipeline-step-meta">output_key: ${step.output_key}${step.only_if_route ? ` · only_if: ${Array.isArray(step.only_if_route) ? step.only_if_route.join(", ") : step.only_if_route}` : ""}</div>
      </div>
    `;

    const pipelineHtml = `
      <div class="detail-block">
        <div class="detail-block-title">
          Pipeline (${totalSteps} Steps)
          <button class="btn btn-sm btn-accent" onclick="addPipelineStep(${id})">+ Step</button>
        </div>
        ${
          (a.pipeline || []).length
            ? `
          <div class="ns-label" style="margin-bottom:6px;">Feste Steps (pipeline)</div>
          ${(a.pipeline || []).map((s, i) => renderStep(s, i, "pipeline")).join("")}
        `
            : ""
        }
        ${
          hasTemplate
            ? `
          <div class="ns-label" style="margin:10px 0 6px;">Template (${a.pipeline_template.source === "static" ? a.pipeline_template.foreach_items?.length + " Items" : "dynamisch aus " + a.pipeline_template.foreach})</div>
          <div class="pipeline-step" style="border-style:dashed;">
            <div class="pipeline-step-header">
              <span class="pipeline-step-id">∀ ${a.pipeline_template.step?.id || "template"}</span>
              <span class="badge badge-cap">${a.pipeline_template.step?.capability || "?"}</span>
              ${a.pipeline_template.only_if_route ? `<span class="badge badge-pipeline">${a.pipeline_template.only_if_route}</span>` : ""}
              <button class="btn btn-sm" onclick="editPipelineTemplate(${id})">Bearbeiten</button>
            </div>
            <div class="pipeline-step-prompt">${a.pipeline_template.step?.prompt_template?.slice(0, 120) || ""}…</div>
            <div class="pipeline-step-meta">aggregate_key: ${a.pipeline_template.aggregate_key} · batch: ${a.pipeline_template.batch_size || 1}</div>
          </div>
        `
            : ""
        }
        ${
          (a.pipeline_after_template || []).length
            ? `
          <div class="ns-label" style="margin:10px 0 6px;">Nach Template (pipeline_after_template)</div>
          ${(a.pipeline_after_template || []).map((s, i) => renderStep(s, i, "pipeline_after_template")).join("")}
        `
            : ""
        }
        ${totalSteps === 0 ? '<div style="font-size:13px;color:var(--text3);">Keine Pipeline.</div>' : ""}
      </div>
    `;

    detail.innerHTML = `
      <div class="action-bar">
        <button class="btn btn-accent" onclick="triggerAgent(${id})">Jetzt ausführen</button>
        <button class="btn" onclick="editAgent(${id})">Bearbeiten</button>
        ${a.is_active ? `<button class="btn btn-danger" onclick="stopAgent(${id})">Stoppen</button>` : ""}
      </div>
      <div class="detail-block">
        <div class="detail-block-title">Anweisung</div>
        <div class="detail-text">${a.instruction || "—"}</div>
      </div>
      <div class="detail-block">
        <div class="detail-block-title">Konfiguration</div>
        <div class="detail-mono">Zeitplan: ${a.schedule}</div>
        <div class="detail-mono">Letzter Lauf: ${fmt(a.last_run_at)}</div>
        <div class="detail-mono">Nächster Lauf: ${fmt(a.next_run_at)}</div>
        <div class="detail-mono">Typ: ${a.type || "—"}</div>
        <div style="display:flex;align-items:center;gap:8px;margin-top:6px;">
          <span style="font-family:var(--mono);font-size:12px;color:var(--text3);">work_capability:</span>
          <select class="capability-select" id="cap-select-${id}" onchange="saveCapability(${id})">
            ${capabilityOptions}
          </select>
        </div>
        ${a.data_reads && a.data_reads.length ? `<div class="detail-mono" style="margin-top:4px;">data_reads: ${JSON.stringify(a.data_reads)}</div>` : ""}
      </div>
      ${pipelineHtml}
      <hr class="divider">
      <div class="agent-tabs">
        <button class="agent-tab active" data-tab="state">State <span class="agent-tab-count">${state.length}</span></button>
        <button class="agent-tab" data-tab="data">Data <span class="agent-tab-count">${data.length}</span></button>
        <button class="agent-tab" data-tab="memory">Memory <span class="agent-tab-count">${memories.length}</span></button>
      </div>
      <div class="tab-panel active" id="tab-state-${id}">
        <div class="state-grid">
          ${state
            .map(
              (s) => `
            <div class="state-card">
              <div class="state-card-header">
                <div class="state-key">${s.key}</div>
                <div class="state-actions">
                  <button class="btn btn-sm" data-agent="${id}" data-key="${s.key}" data-action="edit-state">Bearbeiten</button>
                  <button class="btn btn-sm btn-danger" data-agent="${id}" data-key="${s.key}" data-action="delete-state">Löschen</button>
                </div>
              </div>
              <div class="state-val">${s.value}</div>
            </div>
          `,
            )
            .join("")}
          ${!state.length ? '<div style="font-size:13px;color:var(--text3);padding:8px 0;">Kein State.</div>' : ""}
        </div>
      </div>
      <div class="tab-panel" id="tab-data-${id}">
        <div style="display:flex;justify-content:flex-end;margin-bottom:10px;">
          <button class="btn btn-sm btn-accent" onclick="addAgentData(${id})">+ Neu</button>
        </div>
        ${
          Object.keys(nsMap).length
            ? Object.entries(nsMap)
                .map(
                  ([ns, entries]) => `
          <div class="ns-block">
            <div class="ns-label">${ns}</div>
            ${entries
              .map(
                (d) => `
              <div class="data-card">
                <div class="data-card-header">
                  <div class="data-key">${d.key} <span style="color:var(--text3);margin-left:6px;">${fmt(d.updated_at)}</span></div>
                  <div class="data-actions">
                    <button class="btn btn-sm" data-agent="${id}" data-idx="${d._idx}" data-action="edit-data">Bearbeiten</button>
                    <button class="btn btn-sm btn-danger" data-agent="${id}" data-idx="${d._idx}" data-action="delete-data">Löschen</button>
                  </div>
                </div>
                <div class="data-val">${d.value}</div>
              </div>
            `,
              )
              .join("")}
          </div>
        `,
                )
                .join("")
            : '<div style="font-size:13px;color:var(--text3);padding:8px 0;">Keine Data-Einträge.</div>'
        }
      </div>
      <div class="tab-panel" id="tab-memory-${id}">
        <div style="display:flex;justify-content:flex-end;margin-bottom:10px;">
          <button class="btn btn-sm btn-accent" onclick="addAgentMemory(${id})">+ Neu</button>
        </div>
        <div class="mem-list">
          ${memories
            .map(
              (m, i) => `
            <div class="mem-row">
              <div class="mem-text">${m.content}</div>
              <div class="mem-date">${fmt(m.created_at)}</div>
              <div class="mem-actions">
                <button class="btn btn-sm" data-agent="${id}" data-idx="${i}" data-action="edit-agent-mem">Edit</button>
                <button class="btn btn-sm btn-danger" data-agent="${id}" data-idx="${i}" data-action="delete-agent-mem">Del</button>
              </div>
            </div>
          `,
            )
            .join("")}
          ${!memories.length ? '<div style="font-size:13px;color:var(--text3);padding:8px 0;">Keine Memories.</div>' : ""}
        </div>
      </div>
    `;

    detail.querySelectorAll(".agent-tab").forEach((tab) => {
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
      });
    });

    detail.removeEventListener("click", handleAgentDetailClick);
    detail.addEventListener("click", handleAgentDetailClick);
  } catch (e) {
    detail.innerHTML =
      '<div style="font-size:13px;color:var(--red);padding:1rem;">Fehler beim Laden.</div>';
  }
}

async function saveCapability(agentId) {
  const val = document.getElementById("cap-select-" + agentId).value;
  try {
    await api("/api/agents/" + agentId, {
      method: "PATCH",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ work_capability: val }),
    });
    const a = agentsData.find((x) => x.id === agentId);
    if (a) a.work_capability = val;
    toast("Capability gespeichert.");
  } catch (e) {
    toast("Fehler.", true);
  }
}

function editPipelineStep(agentId, stepIndex, section) {
  section = section || "pipeline";
  const a = agentsData.find((x) => x.id === agentId);
  const steps =
    section === "pipeline_after_template"
      ? a.pipeline_after_template || []
      : a.pipeline || [];
  const step = steps[stepIndex];
  const capOptions = CAPABILITIES.map(
    (c) =>
      `<option value="${c}" ${c === step.capability ? "selected" : ""}>${c}</option>`,
  ).join("");
  const onlyIfRoute = Array.isArray(step.only_if_route)
    ? step.only_if_route.join(", ")
    : step.only_if_route || "";
  openModal(
    "Pipeline-Step bearbeiten",
    `<div class="modal-field"><div class="modal-label">ID</div><input class="modal-input" id="edit-step-id" value="${step.id}" /></div>
     <div class="modal-field"><div class="modal-label">Capability</div><select class="modal-select" id="edit-step-cap">${capOptions}</select></div>
     <div class="modal-field"><div class="modal-label">Prompt Template</div><textarea class="modal-input" id="edit-step-prompt" style="min-height:200px;">${step.prompt_template}</textarea></div>
     <div class="modal-field"><div class="modal-label">Output Key</div><input class="modal-input" id="edit-step-key" value="${step.output_key}" /></div>
     <div class="modal-field"><div class="modal-label">only_if_route (leer = immer, mehrere mit Komma)</div><input class="modal-input" id="edit-step-route" value="${onlyIfRoute}" placeholder="z.B. normal oder normal, trigger" /></div>
     <div class="modal-field" style="display:flex;align-items:center;gap:8px;">
       <input type="checkbox" id="edit-step-router" ${step.is_router ? "checked" : ""} />
       <label class="modal-label" style="margin:0;">is_router</label>
     </div>`,
    `<button class="btn" onclick="closeModal()">Abbrechen</button>
     <button class="btn btn-accent" onclick="savePipelineStep(${agentId}, ${stepIndex}, '${section}')">Speichern</button>`,
  );
}

async function savePipelineStep(agentId, stepIndex, section) {
  section = section || "pipeline";
  const a = agentsData.find((x) => x.id === agentId);
  const steps =
    section === "pipeline_after_template"
      ? a.pipeline_after_template || []
      : a.pipeline || [];
  const routeRaw = document.getElementById("edit-step-route").value.trim();
  let onlyIfRoute = null;
  if (routeRaw) {
    const parts = routeRaw
      .split(",")
      .map((s) => s.trim())
      .filter(Boolean);
    onlyIfRoute = parts.length === 1 ? parts[0] : parts;
  }
  const updated = {
    id: document.getElementById("edit-step-id").value.trim(),
    capability: document.getElementById("edit-step-cap").value,
    prompt_template: document.getElementById("edit-step-prompt").value,
    output_key: document.getElementById("edit-step-key").value.trim(),
  };
  if (onlyIfRoute !== null) updated.only_if_route = onlyIfRoute;
  if (document.getElementById("edit-step-router").checked)
    updated.is_router = true;

  const newSteps = [...steps];
  newSteps[stepIndex] = updated;
  const patchKey =
    section === "pipeline_after_template"
      ? "pipeline_after_template"
      : "pipeline";

  try {
    await api("/api/agents/" + agentId, {
      method: "PATCH",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ [patchKey]: newSteps }),
    });
    if (section === "pipeline_after_template") {
      a.pipeline_after_template = newSteps;
    } else {
      a.pipeline = newSteps;
    }
    closeModal();
    toast("Step gespeichert.");
    selectAgent(agentId);
  } catch (e) {
    toast("Fehler.", true);
  }
}

async function deletePipelineStep(agentId, stepIndex, section) {
  section = section || "pipeline";
  confirmModal("Pipeline-Step wirklich löschen?", async () => {
    const a = agentsData.find((x) => x.id === agentId);
    const steps =
      section === "pipeline_after_template"
        ? [...(a.pipeline_after_template || [])]
        : [...(a.pipeline || [])];
    steps.splice(stepIndex, 1);
    const patchKey =
      section === "pipeline_after_template"
        ? "pipeline_after_template"
        : "pipeline";
    try {
      await api("/api/agents/" + agentId, {
        method: "PATCH",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ [patchKey]: steps }),
      });
      if (section === "pipeline_after_template") {
        a.pipeline_after_template = steps;
      } else {
        a.pipeline = steps;
      }
      toast("Step gelöscht.");
      selectAgent(agentId);
    } catch (e) {
      toast("Fehler.", true);
    }
  });
}

function editPipelineTemplate(agentId) {
  const a = agentsData.find((x) => x.id === agentId);
  const t = a.pipeline_template || {};
  const step = t.step || {};
  const capOptions = CAPABILITIES.map(
    (c) =>
      `<option value="${c}" ${c === step.capability ? "selected" : ""}>${c}</option>`,
  ).join("");
  const itemsList = Array.isArray(t.foreach_items)
    ? t.foreach_items.join("\n")
    : "";
  openModal(
    "Pipeline Template bearbeiten",
    `<div class="modal-field">
       <div class="modal-label">Source</div>
       <select class="modal-select" id="tmpl-source" onchange="toggleTemplateSource()">
         <option value="static" ${t.source === "static" ? "selected" : ""}>static (feste Liste)</option>
         <option value="state" ${t.source === "state" ? "selected" : ""}>state (aus Agent-State)</option>
         <option value="injected" ${t.source === "injected" ? "selected" : ""}>injected (aus data_reads)</option>
       </select>
     </div>
     <div class="modal-field" id="tmpl-foreach-field" style="${t.source !== "static" ? "" : "display:none"}">
       <div class="modal-label">foreach (State-Key)</div>
       <input class="modal-input" id="tmpl-foreach" value="${t.foreach || ""}" placeholder="z.B. fundamentalanalyse_vorhanden" />
     </div>
     <div class="modal-field" id="tmpl-items-field" style="${t.source === "static" ? "" : "display:none"}">
       <div class="modal-label">foreach_items (ein Item pro Zeile)</div>
       <textarea class="modal-input" id="tmpl-items" style="min-height:100px;">${itemsList}</textarea>
     </div>
     <div class="modal-field"><div class="modal-label">split_by (Trennzeichen)</div><input class="modal-input" id="tmpl-split" value="${t.split_by || ","}" /></div>
     <div class="modal-field"><div class="modal-label">batch_size</div><input class="modal-input" id="tmpl-batch" type="number" value="${t.batch_size || 1}" /></div>
     <div class="modal-field"><div class="modal-label">aggregate_key</div><input class="modal-input" id="tmpl-agg" value="${t.aggregate_key || ""}" /></div>
     <div class="modal-field"><div class="modal-label">only_if_route (leer = immer)</div><input class="modal-input" id="tmpl-route" value="${t.only_if_route || ""}" /></div>
     <hr class="divider">
     <div class="modal-label" style="margin-bottom:8px;">Step Template</div>
     <div class="modal-field"><div class="modal-label">ID ({{item_id}} verfügbar)</div><input class="modal-input" id="tmpl-step-id" value="${step.id || "search_{{item_id}}"}" /></div>
     <div class="modal-field"><div class="modal-label">Capability</div><select class="modal-select" id="tmpl-step-cap">${capOptions}</select></div>
     <div class="modal-field"><div class="modal-label">Prompt Template ({{item}} und {{item_id}} verfügbar)</div><textarea class="modal-input" id="tmpl-step-prompt" style="min-height:140px;">${step.prompt_template || ""}</textarea></div>
     <div class="modal-field"><div class="modal-label">Output Key ({{item_id}} verfügbar)</div><input class="modal-input" id="tmpl-step-key" value="${step.output_key || "result_{{item_id}}"}" /></div>`,
    `<button class="btn" onclick="closeModal()">Abbrechen</button>
     <button class="btn btn-accent" onclick="savePipelineTemplate(${agentId})">Speichern</button>`,
  );
}

function toggleTemplateSource() {
  const src = document.getElementById("tmpl-source").value;
  document.getElementById("tmpl-foreach-field").style.display =
    src !== "static" ? "" : "none";
  document.getElementById("tmpl-items-field").style.display =
    src === "static" ? "" : "none";
}

async function savePipelineTemplate(agentId) {
  const source = document.getElementById("tmpl-source").value;
  const template = {
    source,
    split_by: document.getElementById("tmpl-split").value || ",",
    batch_size: parseInt(document.getElementById("tmpl-batch").value) || 1,
    aggregate_key: document.getElementById("tmpl-agg").value.trim(),
    step: {
      id: document.getElementById("tmpl-step-id").value.trim(),
      capability: document.getElementById("tmpl-step-cap").value,
      prompt_template: document.getElementById("tmpl-step-prompt").value,
      output_key: document.getElementById("tmpl-step-key").value.trim(),
    },
  };
  if (source === "static") {
    const items = document
      .getElementById("tmpl-items")
      .value.split("\n")
      .map((s) => s.trim())
      .filter(Boolean);
    template.foreach_items = items;
  } else {
    template.foreach = document.getElementById("tmpl-foreach").value.trim();
  }
  const routeRaw = document.getElementById("tmpl-route").value.trim();
  if (routeRaw) template.only_if_route = routeRaw;

  try {
    await api("/api/agents/" + agentId, {
      method: "PATCH",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ pipeline_template: template }),
    });
    const a = agentsData.find((x) => x.id === agentId);
    a.pipeline_template = template;
    closeModal();
    toast("Template gespeichert.");
    selectAgent(agentId);
  } catch (e) {
    toast("Fehler.", true);
  }
}

function addPipelineStep(agentId) {
  const a = agentsData.find((x) => x.id === agentId);
  const hasAfterTemplate =
    a.pipeline_after_template && a.pipeline_after_template.length > 0;
  const capOptions = CAPABILITIES.map(
    (c) => `<option value="${c}">${c}</option>`,
  ).join("");
  openModal(
    "Neuen Pipeline-Step hinzufügen",
    `<div class="modal-field">
       <div class="modal-label">Sektion</div>
       <select class="modal-select" id="new-step-section">
         <option value="pipeline">pipeline (feste Steps)</option>
         ${hasAfterTemplate ? '<option value="pipeline_after_template">pipeline_after_template (nach Template)</option>' : ""}
       </select>
     </div>
     <div class="modal-field"><div class="modal-label">ID</div><input class="modal-input" id="new-step-id" placeholder="z.B. search_news" /></div>
     <div class="modal-field"><div class="modal-label">Capability</div><select class="modal-select" id="new-step-cap">${capOptions}</select></div>
     <div class="modal-field"><div class="modal-label">Prompt Template</div><textarea class="modal-input" id="new-step-prompt" style="min-height:160px;" placeholder="Anweisung für diesen Step…"></textarea></div>
     <div class="modal-field"><div class="modal-label">Output Key</div><input class="modal-input" id="new-step-key" placeholder="z.B. search_result" /></div>
     <div class="modal-field"><div class="modal-label">only_if_route (leer = immer)</div><input class="modal-input" id="new-step-route" placeholder="z.B. normal" /></div>
     <div class="modal-field"><div class="modal-label">Position (leer = ans Ende)</div><input class="modal-input" id="new-step-pos" type="number" placeholder="0 = Anfang" /></div>
     <div class="modal-field" style="display:flex;align-items:center;gap:8px;">
       <input type="checkbox" id="new-step-router" />
       <label class="modal-label" style="margin:0;">is_router</label>
     </div>`,
    `<button class="btn" onclick="closeModal()">Abbrechen</button>
     <button class="btn btn-accent" onclick="saveNewPipelineStep(${agentId})">Hinzufügen</button>`,
  );
}

async function saveNewPipelineStep(agentId) {
  const id = document.getElementById("new-step-id").value.trim();
  const key = document.getElementById("new-step-key").value.trim();
  const prompt = document.getElementById("new-step-prompt").value.trim();
  if (!id || !key || !prompt) {
    toast("ID, Output Key und Prompt sind pflicht.", true);
    return;
  }
  const section = document.getElementById("new-step-section").value;
  const routeRaw = document.getElementById("new-step-route").value.trim();
  let onlyIfRoute = null;
  if (routeRaw) {
    const parts = routeRaw
      .split(",")
      .map((s) => s.trim())
      .filter(Boolean);
    onlyIfRoute = parts.length === 1 ? parts[0] : parts;
  }
  const posRaw = document.getElementById("new-step-pos").value.trim();
  const pos = posRaw !== "" ? parseInt(posRaw) : null;
  const step = {
    id,
    capability: document.getElementById("new-step-cap").value,
    prompt_template: prompt,
    output_key: key,
  };
  if (onlyIfRoute !== null) step.only_if_route = onlyIfRoute;
  if (document.getElementById("new-step-router").checked) step.is_router = true;

  const a = agentsData.find((x) => x.id === agentId);
  const currentSteps =
    section === "pipeline_after_template"
      ? [...(a.pipeline_after_template || [])]
      : [...(a.pipeline || [])];

  if (pos === null || pos >= currentSteps.length) {
    currentSteps.push(step);
  } else {
    currentSteps.splice(pos, 0, step);
  }

  try {
    await api("/api/agents/" + agentId, {
      method: "PATCH",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ [section]: currentSteps }),
    });
    if (section === "pipeline_after_template") {
      a.pipeline_after_template = currentSteps;
    } else {
      a.pipeline = currentSteps;
    }
    closeModal();
    toast("Step hinzugefügt.");
    selectAgent(agentId);
  } catch (e) {
    toast("Fehler.", true);
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
      `<div class="modal-field"><div class="modal-label">Key</div><input class="modal-input" value="${key}" disabled /></div>
       <div class="modal-field"><div class="modal-label">Value</div><textarea class="modal-input" id="edit-state-val" style="min-height:140px;">${currentVal}</textarea></div>`,
      `<button class="btn" onclick="closeModal()">Abbrechen</button>
       <button class="btn btn-accent" onclick="saveStateEntry(${agentId}, '${key}')">Speichern</button>`,
    );
  } else if (action === "delete-state") {
    const key = btn.dataset.key;
    confirmModal('State-Eintrag "' + key + '" wirklich löschen?', () => {
      api("/api/agents/" + agentId + "/state/" + encodeURIComponent(key), {
        method: "DELETE",
      })
        .then(() => {
          toast("Gelöscht.");
          selectAgent(agentId);
        })
        .catch(() => toast("Fehler.", true));
    });
  } else if (action === "edit-data") {
    const idx = Number(btn.dataset.idx);
    const d = _agentDataCache[agentId][idx];
    openModal(
      "Data bearbeiten",
      `<div class="modal-field"><div class="modal-label">Namespace</div><input class="modal-input" value="${d.namespace}" disabled /></div>
       <div class="modal-field"><div class="modal-label">Key</div><input class="modal-input" value="${d.key}" disabled /></div>
       <div class="modal-field"><div class="modal-label">Value</div><textarea class="modal-input" id="edit-data-val" style="min-height:200px;">${d.value}</textarea></div>`,
      `<button class="btn" onclick="closeModal()">Abbrechen</button>
       <button class="btn btn-accent" onclick="saveAgentData(${agentId}, ${idx})">Speichern</button>`,
    );
  } else if (action === "delete-data") {
    const idx = Number(btn.dataset.idx);
    const d = _agentDataCache[agentId][idx];
    confirmModal('Eintrag "' + d.namespace + "/" + d.key + '" löschen?', () => {
      api(
        "/api/agents/" +
          agentId +
          "/data/" +
          encodeURIComponent(d.namespace) +
          "/" +
          encodeURIComponent(d.key),
        { method: "DELETE" },
      )
        .then(() => {
          toast("Gelöscht.");
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
      `<button class="btn" onclick="closeModal()">Abbrechen</button>
       <button class="btn btn-accent" onclick="saveAgentMemory(${agentId}, ${idx})">Speichern</button>`,
    );
  } else if (action === "delete-agent-mem") {
    const idx = Number(btn.dataset.idx);
    const mem = _agentMemCache[agentId][idx];
    confirmModal("Memory löschen?", () => {
      api("/api/agents/" + agentId + "/memories", {
        method: "DELETE",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ content: mem.content, subject_type: "agent" }),
      })
        .then(() => {
          toast("Gelöscht.");
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
  } catch (e) {
    toast("Fehler.", true);
  }
}

async function saveAgentData(agentId, idx) {
  const d = _agentDataCache[agentId][idx];
  const value = document.getElementById("edit-data-val").value;
  try {
    await api(
      "/api/agents/" +
        agentId +
        "/data/" +
        encodeURIComponent(d.namespace) +
        "/" +
        encodeURIComponent(d.key),
      {
        method: "PATCH",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ value }),
      },
    );
    closeModal();
    toast("Gespeichert.");
    selectAgent(agentId);
  } catch (e) {
    toast("Fehler.", true);
  }
}

function addAgentData(agentId) {
  const existing = _agentDataCache[agentId] || [];
  const namespaces = [...new Set(existing.map((d) => d.namespace))];
  openModal(
    "Data-Eintrag hinzufügen",
    `<div class="modal-field">
       <div class="modal-label">Namespace</div>
       <input class="modal-input" id="new-data-ns" list="ns-suggestions" placeholder="z.B. analyses" />
       <datalist id="ns-suggestions">${namespaces.map((ns) => `<option value="${ns}">`).join("")}</datalist>
     </div>
     <div class="modal-field"><div class="modal-label">Key</div><input class="modal-input" id="new-data-key" placeholder="z.B. EOSE" /></div>
     <div class="modal-field"><div class="modal-label">Value</div><textarea class="modal-input" id="new-data-val" style="min-height:120px;" placeholder="Inhalt…"></textarea></div>`,
    `<button class="btn" onclick="closeModal()">Abbrechen</button>
     <button class="btn btn-accent" onclick="saveNewAgentData(${agentId})">Speichern</button>`,
  );
}

async function saveNewAgentData(agentId) {
  const namespace = document.getElementById("new-data-ns").value.trim();
  const key = document.getElementById("new-data-key").value.trim();
  const value = document.getElementById("new-data-val").value.trim();
  if (!namespace || !key || !value) {
    toast("Alle Felder sind pflicht.", true);
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
  } catch (e) {
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
  } catch (e) {
    toast("Fehler.", true);
  }
}

function addAgentMemory(agentId) {
  openModal(
    "Memory hinzufügen",
    `<div class="modal-field"><div class="modal-label">Inhalt</div><textarea class="modal-input" id="new-mem-content" placeholder="Neue Beobachtung…"></textarea></div>`,
    `<button class="btn" onclick="closeModal()">Abbrechen</button>
     <button class="btn btn-accent" onclick="saveNewAgentMemory(${agentId})">Speichern</button>`,
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
  } catch (e) {
    toast("Fehler.", true);
  }
}

async function triggerAgent(id) {
  try {
    await api("/api/agents/" + id + "/trigger", { method: "POST" });
    toast("Trigger gesetzt — läuft beim nächsten Scheduler-Durchlauf.");
  } catch (e) {
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
    } catch (e) {
      toast("Fehler.", true);
    }
  });
}

function editAgent(id) {
  const a = agentsData.find((x) => x.id === id);
  openModal(
    "Agent bearbeiten",
    `<div class="modal-field"><div class="modal-label">Name</div><input class="modal-input" id="edit-name" value="${a.name}" /></div>
     <div class="modal-field"><div class="modal-label">Zeitplan (Cron)</div><input class="modal-input" id="edit-schedule" value="${a.schedule}" /></div>
     <div class="modal-field"><div class="modal-label">Anweisung</div><textarea class="modal-input" id="edit-instruction" style="min-height:140px;">${a.instruction}</textarea></div>`,
    `<button class="btn" onclick="closeModal()">Abbrechen</button>
     <button class="btn btn-accent" onclick="saveAgent(${id})">Speichern</button>`,
  );
}

async function saveAgent(id) {
  const name = document.getElementById("edit-name").value.trim();
  const schedule = document.getElementById("edit-schedule").value.trim();
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
  } catch (e) {
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
  } catch (e) {
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
    '<div class="empty-state">← Eintrag auswählen</div>';
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
      '<div style="font-size:13px;color:var(--text3);padding:8px;">Keine Einträge.</div>';
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
      return `<div class="sidebar-item" data-id="${i.id}" onclick="selectMemItem(${i.id})">
      <div class="si-name">${name}</div>
      <div class="si-meta">${sub || ""}</div>
    </div>`;
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
  detail.innerHTML = '<div class="empty-state">Lade…</div>';
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

    detail.innerHTML = `
      <div class="action-bar">
        <button class="btn btn-accent" onclick="addMemory(${id})">+ Neue Memory</button>
      </div>
      ${availableTypes
        .map(
          (type) => `
        <div class="detail-block">
          <div class="detail-block-title"><span>${type} (${byType[type].length})</span></div>
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
              </div>
            `,
              )
              .join("")}
            ${!byType[type].length ? '<div style="font-size:13px;color:var(--text3);padding:6px 0;">Keine Einträge.</div>' : ""}
          </div>
        </div>
      `,
        )
        .join("")}
    `;
    detail.removeEventListener("click", handleMemDetailClick);
    detail.addEventListener("click", handleMemDetailClick);
  } catch (e) {
    detail.innerHTML =
      '<div style="font-size:13px;color:var(--red);padding:1rem;">Fehler.</div>';
  }
}

function handleMemDetailClick(e) {
  const btn = e.target.closest("[data-action]");
  if (!btn) return;
  const action = btn.dataset.action;
  const subjectId = Number(btn.dataset.subject);
  const idx = Number(btn.dataset.idx);
  const mem = _memCache[subjectId][idx];
  if (!mem) return;
  const endpoint =
    memMode === "users"
      ? "/api/users/" + subjectId + "/memories"
      : "/api/groups/" + subjectId + "/memories";
  if (action === "edit-mem") {
    openModal(
      "Memory bearbeiten",
      `<div class="modal-field"><div class="modal-label">Typ</div><input class="modal-input" value="${mem.type}" disabled /></div>
       <div class="modal-field"><div class="modal-label">Inhalt</div><textarea class="modal-input" id="edit-mem-content" style="min-height:100px;">${mem.content}</textarea></div>`,
      `<button class="btn" onclick="closeModal()">Abbrechen</button>
       <button class="btn btn-accent" onclick="saveMemory(${subjectId}, ${idx})">Speichern</button>`,
    );
  } else if (action === "delete-mem") {
    confirmModal("Memory löschen?", () => {
      api(endpoint, {
        method: "DELETE",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ content: mem.content, subject_type: mem.type }),
      })
        .then(() => {
          toast("Gelöscht.");
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
  } catch (e) {
    toast("Fehler.", true);
  }
}

function addMemory(subjectId) {
  const types =
    memMode === "users"
      ? ["user", "reflection"]
      : ["group", "bot", "reflection"];
  openModal(
    "Memory hinzufügen",
    `<div class="modal-field">
       <div class="modal-label">Typ</div>
       <select class="modal-select" id="new-mem-type">${types.map((t) => `<option value="${t}">${t}</option>`).join("")}</select>
     </div>
     <div class="modal-field"><div class="modal-label">Inhalt</div><textarea class="modal-input" id="new-mem-content" placeholder="Inhalt…"></textarea></div>`,
    `<button class="btn" onclick="closeModal()">Abbrechen</button>
     <button class="btn btn-accent" onclick="saveNewMemory(${subjectId})">Speichern</button>`,
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
  } catch (e) {
    toast("Fehler.", true);
  }
}

document
  .getElementById("usage-tabs")
  .querySelectorAll(".agent-tab")
  .forEach((tab) => {
    tab.addEventListener("click", function () {
      document
        .querySelectorAll("#usage-tabs .agent-tab")
        .forEach((t) => t.classList.remove("active"));
      document.getElementById("usage-tab-models").classList.remove("active");
      document.getElementById("usage-tab-caller").classList.remove("active");
      document.getElementById("usage-tab-history").classList.remove("active");
      this.classList.add("active");
      document
        .getElementById("usage-tab-" + this.dataset.tab)
        .classList.add("active");
    });
  });

async function loadUsage() {
  try {
    const u = await api("/api/usage");

    const totalCost = u.by_model.reduce((s, m) => s + m.estimated_cost_usd, 0);

    document.getElementById("usage-tab-models").innerHTML = `
      <div class="metrics-grid">
        <div class="metric-card"><div class="metric-label">input tokens</div><div class="metric-value">${fmtNum(u.total_input)}</div></div>
        <div class="metric-card"><div class="metric-label">output tokens</div><div class="metric-value">${fmtNum(u.total_output)}</div></div>
        <div class="metric-card"><div class="metric-label">geschätzte kosten</div><div class="metric-value">${fmtCost(totalCost)}</div></div>
      </div>
      <div class="detail-block">
        <div class="detail-block-title">Nach Modell</div>
        <div style="display:flex;flex-direction:column;">
          ${u.by_model
            .map(
              (m) => `
            <div style="display:grid;grid-template-columns:1fr auto auto auto;align-items:baseline;gap:12px;padding:8px 0;border-bottom:0.5px solid var(--border);">
              <div style="font-family:var(--mono);font-size:11px;color:var(--text2);overflow:hidden;text-overflow:ellipsis;white-space:nowrap;">${m.model}</div>
              <div style="font-family:var(--mono);font-size:11px;color:var(--text3);white-space:nowrap;">↑${fmtNum(m.input)} ↓${fmtNum(m.output)}</div>
              <div style="font-family:var(--mono);font-size:11px;color:var(--text3);white-space:nowrap;">${m.calls} calls</div>
              <div style="font-family:var(--mono);font-size:11px;color:var(--accent);white-space:nowrap;">${fmtCost(m.estimated_cost_usd)}</div>
            </div>
          `,
            )
            .join("")}
          ${!u.by_model.length ? '<div style="font-size:13px;color:var(--text3);padding:8px 0;">Noch keine Model-Daten — wird nach dem nächsten Bot-Restart befüllt.</div>' : ""}
        </div>
      </div>
    `;

    const maxTokens = u.by_caller.reduce(
      (m, c) => Math.max(m, c.input + c.output),
      1,
    );
    document.getElementById("usage-tab-caller").innerHTML = `
      <div class="detail-block">
        <div class="detail-block-title">Nach Caller</div>
        <div class="caller-list">
          ${u.by_caller
            .map((c) => {
              const t = c.input + c.output;
              const pct = Math.round((t / maxTokens) * 100);
              return `<div class="caller-row">
              <div class="caller-name">${c.caller}</div>
              <div class="caller-bar-bg"><div class="caller-bar-fg" style="width:${pct}%"></div></div>
              <div class="caller-tokens">${fmtNum(t)}</div>
            </div>`;
            })
            .join("")}
        </div>
      </div>
    `;
  } catch (e) {
    document.getElementById("usage-tab-models").innerHTML =
      '<div style="font-size:13px;color:var(--red);padding:1rem;">Fehler.</div>';
  }
  await loadUsageHistory(0);
}

async function loadUsageHistory(page) {
  _usagePage = page;
  const el = document.getElementById("usage-tab-history");
  el.innerHTML = '<div class="empty-state">Lade…</div>';
  try {
    const h = await api(
      "/api/usage/history?page=" + page + "&limit=" + _usageLimit,
    );
    const totalPages = Math.ceil(h.total / _usageLimit);
    el.innerHTML = `
      <div class="detail-block">
        <div class="detail-block-title">
          <span>Calls (${fmtNum(h.total)})</span>
          <span style="font-size:11px;color:var(--text3);">Seite ${page + 1} / ${totalPages}</span>
        </div>
        <div style="display:flex;flex-direction:column;">
          ${h.items
            .map(
              (item) => `
            <div style="display:grid;grid-template-columns:1fr auto auto auto auto;align-items:baseline;gap:8px;padding:7px 0;border-bottom:0.5px solid var(--border);">
              <div style="font-family:var(--mono);font-size:11px;color:var(--text2);overflow:hidden;text-overflow:ellipsis;white-space:nowrap;">${item.caller}</div>
              <div style="font-family:var(--mono);font-size:10px;color:var(--text3);white-space:nowrap;">${item.model || "—"}</div>
              <div style="font-family:var(--mono);font-size:11px;color:var(--accent);">↑${fmtNum(item.input_tokens)}</div>
              <div style="font-family:var(--mono);font-size:11px;color:var(--blue);">↓${fmtNum(item.output_tokens)}</div>
              <div style="font-family:var(--mono);font-size:10px;color:var(--text3);">${fmt(item.created_at)}</div>
            </div>
          `,
            )
            .join("")}
        </div>
        <div style="display:flex;justify-content:space-between;align-items:center;margin-top:12px;">
          <button class="btn btn-sm" onclick="loadUsageHistory(${page - 1})" ${page === 0 ? 'disabled style="opacity:0.3;"' : ""}>← Zurück</button>
          <button class="btn btn-sm" onclick="loadUsageHistory(${page + 1})" ${page >= totalPages - 1 ? 'disabled style="opacity:0.3;"' : ""}>Weiter →</button>
        </div>
      </div>
    `;
  } catch (e) {
    el.innerHTML =
      '<div style="font-size:13px;color:var(--red);padding:1rem;">Fehler.</div>';
  }
}

async function loadRegistry() {
  const el = document.getElementById("registry-detail");
  el.innerHTML = '<div class="empty-state">Lade…</div>';
  try {
    const data = await api("/api/registry");

    const routingHtml = data.routing
      .map(
        (r) => `
      <div style="display:grid;grid-template-columns:140px 1fr auto;align-items:center;gap:12px;padding:7px 0;border-bottom:0.5px solid var(--border);">
        <div style="font-family:var(--mono);font-size:11px;color:var(--accent);">${r.capability}</div>
        <div style="font-family:var(--mono);font-size:11px;color:var(--text2);">${r.model} ${r.is_local ? '<span style="color:var(--amber);">[local]</span>' : ""}</div>
        <div style="font-family:var(--mono);font-size:10px;color:var(--text3);">${r.input_cost_per_mtok === 0 ? "free" : "$" + r.input_cost_per_mtok + "/MTok"}</div>
      </div>
    `,
      )
      .join("");

    const byProvider = {};
    data.models.forEach((m) => {
      if (!byProvider[m.provider]) byProvider[m.provider] = [];
      byProvider[m.provider].push(m);
    });

    const modelsHtml = Object.entries(byProvider)
      .map(
        ([provider, models]) => `
      <div class="ns-block">
        <div class="ns-label">${provider}</div>
        ${models
          .map(
            (m) => `
          <div class="data-card" style="opacity:${m.is_available ? 1 : 0.45};">
            <div class="data-card-header">
              <div style="display:flex;align-items:center;gap:8px;">
                <span style="font-size:13px;font-weight:500;">${m.display_name}</span>
                <span class="badge ${m.is_available ? "badge-active" : "badge-inactive"}">${m.is_available ? "verfügbar" : "nicht verfügbar"}</span>
                ${m.is_local ? '<span class="badge badge-pipeline">local</span>' : ""}
              </div>
              <div style="font-family:var(--mono);font-size:10px;color:var(--text3);">
                ${m.input_cost_per_mtok === 0 ? "kostenlos" : "$" + m.input_cost_per_mtok + " / $" + m.output_cost_per_mtok + " MTok"}
              </div>
            </div>
            <div style="font-family:var(--mono);font-size:10px;color:var(--text3);margin-bottom:6px;">${m.api_model_name} · ctx: ${m.context_window ? fmtNum(m.context_window) : "—"} · out: ${m.max_output_tokens ? fmtNum(m.max_output_tokens) : "—"}</div>
            <div style="display:flex;flex-wrap:wrap;gap:4px;">
              ${m.capabilities.map((c) => `<span class="badge badge-cap">${c}</span>`).join("")}
            </div>
            ${m.notes ? `<div style="font-size:11px;color:var(--text3);margin-top:6px;">${m.notes}</div>` : ""}
          </div>
        `,
          )
          .join("")}
      </div>
    `,
      )
      .join("");

    el.innerHTML = `
      <div class="detail-block">
        <div class="detail-block-title">Aktives Routing</div>
        ${routingHtml || '<div style="font-size:13px;color:var(--text3);">Keine Routing-Daten — Bot noch nicht gestartet.</div>'}
      </div>
      <hr class="divider">
      <div class="detail-block">
        <div class="detail-block-title">Alle Modelle (${data.models.length})</div>
        ${modelsHtml}
      </div>
    `;
  } catch (e) {
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
        (t) => `
      <div class="trigger-item">
        <div class="trigger-header">
          <span class="trigger-name">${t.target_agent_name}</span>
          <span class="badge ${t.processed_at ? "badge-done" : "badge-pending"}">${t.processed_at ? "verarbeitet" : "ausstehend"}</span>
        </div>
        <div class="trigger-meta">geplant ${fmt(t.scheduled_for)} · erstellt ${fmt(t.created_at)}</div>
        ${Object.keys(t.payload || {}).length ? `<div class="trigger-meta" style="margin-top:2px;">payload: ${JSON.stringify(t.payload)}</div>` : ""}
      </div>
    `,
      )
      .join("");
  } catch (e) {
    el.innerHTML =
      '<div style="font-size:13px;color:var(--red);padding:1rem;">Fehler.</div>';
  }
}

function loadAll() {
  loadAgents();
  loadMemory();
  loadUsage();
  loadRegistry();
  loadTriggers();
}

loadAll();
