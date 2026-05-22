const RATING_LABELS = {
  perfekt: "Perfekt",
  sehr_gut: "Sehr gut",
  gut: "Gut",
  ausreichend: "Ausreichend",
  ungenuegend: "Ungenuegend",
};

const RATING_COLORS = {
  perfekt: "var(--accent)",
  sehr_gut: "var(--accent)",
  gut: "var(--amber)",
  ausreichend: "var(--text3)",
  ungenuegend: "var(--red)",
};

function ratingBadge(rating) {
  if (!rating)
    return '<span style="font-size:11px;color:var(--text3);">nicht bewertet</span>';
  const color = RATING_COLORS[rating] || "var(--text3)";
  const label = RATING_LABELS[rating] || rating;
  return `<span class="badge" style="background:transparent;border:1px solid ${color};color:${color};">${label}</span>`;
}

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
      const ratingColor = a.current_rating
        ? RATING_COLORS[a.current_rating]
        : null;
      const ratingDot = ratingColor
        ? `<span style="width:6px;height:6px;border-radius:50%;background:${ratingColor};display:inline-block;flex-shrink:0;"></span>`
        : "";
      return `<div class="sidebar-item" data-id="${a.id}" onclick="selectAgent(${a.id})">
        <div class="si-name">${a.name}<span class="badge ${a.is_active ? "badge-active" : "badge-inactive"}">${a.is_active ? "aktiv" : "inaktiv"}</span>${totalSteps > 0 ? `<span class="badge badge-pipeline">${totalSteps} steps</span>` : ""}${ratingDot}</div>
        <div class="si-meta">${a.type || "--"} . ${a.schedule || "trigger-only"}</div>
      </div>`;
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
    const [state, data, memories, notifications] = await Promise.all([
      api("/api/agents/" + id + "/state"),
      api("/api/agents/" + id + "/data"),
      api("/api/agents/" + id + "/memories"),
      api("/api/agents/" + id + "/notifications").catch(() => []),
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
        <button class="btn" onclick="openRatingModal(${id})">Bewerten</button>
        ${a.is_active ? `<button class="btn btn-danger" onclick="stopAgent(${id})">Stoppen</button>` : ""}
      </div>
      <div class="detail-block"><div class="detail-block-title">Anweisung</div><div class="detail-text">${a.instruction || "--"}</div></div>
      <div class="detail-block"><div class="detail-block-title">Konfiguration</div>
        <div class="detail-mono">Zeitplan: ${a.schedule || "nur auf Trigger"}</div>
        <div class="detail-mono">Letzter Lauf: ${fmt(a.last_run_at)}</div>
        <div class="detail-mono">Naechster Lauf: ${a.next_run_at ? fmt(a.next_run_at) : "--"}</div>
        <div class="detail-mono">Typ: ${a.type || "--"}</div>
      </div>
      <div class="detail-block">
        <div class="detail-block-title">Bewertung</div>
        <div style="display:flex;align-items:center;gap:10px;margin-bottom:6px;">
          ${ratingBadge(a.current_rating)}
          ${a.last_rated_at ? `<span style="font-size:11px;color:var(--text3);">${fmt(a.last_rated_at)}</span>` : ""}
        </div>
        ${a.current_rating_note ? `<div class="detail-text" style="font-size:12px;color:var(--text2);margin-bottom:6px;">${a.current_rating_note}</div>` : ""}
        <button class="btn btn-sm" onclick="loadRatingHistory(${id})">Verlauf anzeigen</button>
        <div id="rating-history-${id}"></div>
      </div>
      <div class="detail-block"><div class="detail-block-title">Pipeline (${allSteps.length} Steps)<button class="btn btn-sm btn-accent" onclick="addStep(${id})">+ Step</button></div>
        ${allSteps.length ? allSteps.map((s, i) => renderStep(s, i, id)).join("") : '<div style="font-size:13px;color:var(--text3);">Keine Pipeline.</div>'}</div>
      <hr class="divider">
      <div class="agent-tabs">
        <button class="agent-tab active" data-tab="state">State <span class="agent-tab-count">${state.length}</span></button>
        <button class="agent-tab" data-tab="data">Data <span class="agent-tab-count">${data.length}</span></button>
        <button class="agent-tab" data-tab="notifications">Notifications <span class="agent-tab-count">${notifications.length}</span></button>
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
      <div class="tab-panel" id="tab-notifications-${id}">
        <div class="mem-list">
          ${
            notifications.length
              ? notifications
                  .map(
                    (n) => `
            <div class="mem-row">
              <div class="mem-text">
                <span class="badge badge-pipeline" style="margin-right:6px;">${n.notification_type}</span>
                <span style="font-size:12px;">${n.payload_summary?.summary || "—"}</span>
              </div>
              <div class="mem-date">${fmt(n.created_at)}</div>
            </div>`,
                  )
                  .join("")
              : '<div style="font-size:13px;color:var(--text3);padding:8px 0;">Noch keine Notifications.</div>'
          }
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
  } catch (e) {
    console.error("selectAgent error:", e);
    detail.innerHTML =
      '<div style="font-size:13px;color:var(--red);padding:1rem;">Fehler beim Laden.</div>';
  }
}

async function loadRatingHistory(agentId) {
  const el = document.getElementById(`rating-history-${agentId}`);
  if (!el) return;
  try {
    const history = await api(`/api/agents/${agentId}/ratings`);
    if (!history.length) {
      el.innerHTML =
        '<div style="font-size:12px;color:var(--text3);margin-top:6px;">Kein Verlauf.</div>';
      return;
    }
    el.innerHTML = `<div class="mem-list" style="margin-top:8px;">${history
      .map(
        (h) => `<div class="mem-row">
          ${ratingBadge(h.rating)}
          <div class="mem-text" style="font-size:12px;">${h.note || "—"}</div>
          <div class="mem-date">${fmt(h.rated_at)}</div>
        </div>`,
      )
      .join("")}</div>`;
  } catch {
    el.innerHTML =
      '<div style="font-size:12px;color:var(--red);margin-top:6px;">Fehler.</div>';
  }
}

function openRatingModal(agentId) {
  const a = agentsData.find((x) => x.id === agentId);
  const currentRating = a?.current_rating || "";
  const currentNote = a?.current_rating_note || "";

  const options = [
    { value: "perfekt", label: "Perfekt — macht genau was es soll" },
    {
      value: "sehr_gut",
      label: "Sehr gut — funktioniert fast immer, minimale LLM-Varianz",
    },
    { value: "gut", label: "Gut — funktioniert, kann aber noch Bugs haben" },
    {
      value: "ausreichend",
      label:
        "Ausreichend — laeuft, produziert ueberwiegend unbefriedigende Ergebnisse",
    },
    { value: "ungenuegend", label: "Ungenuegend — laeuft nicht" },
  ];

  const optionsHtml = options
    .map(
      (
        o,
      ) => `<label style="display:flex;align-items:flex-start;gap:10px;padding:8px;border-radius:6px;cursor:pointer;border:0.5px solid ${currentRating === o.value ? "var(--accent)" : "transparent"};margin-bottom:4px;" class="rating-option" data-value="${o.value}">
        <input type="radio" name="rating" value="${o.value}" ${currentRating === o.value ? "checked" : ""} style="margin-top:2px;flex-shrink:0;" />
        <span style="font-size:13px;color:${RATING_COLORS[o.value]};">${o.label}</span>
      </label>`,
    )
    .join("");

  openModal(
    "Agent bewerten",
    `<div class="modal-field">
       <div class="modal-label">Qualitaet</div>
       <div id="rating-options">${optionsHtml}</div>
     </div>
     <div class="modal-field">
       <div class="modal-label">Notiz (optional) — was genau ist problematisch oder besonders gut?</div>
       <textarea class="modal-input" id="rating-note" style="min-height:80px;" placeholder="z.B. llm_decide variiert zu stark bei mehrdeutigen Titeln...">${currentNote}</textarea>
     </div>`,
    `<button class="btn" onclick="closeModal()">Abbrechen</button>
     <button class="btn btn-accent" onclick="saveRating(${agentId})">Speichern</button>`,
  );

  document.querySelectorAll(".rating-option").forEach((label) => {
    label.addEventListener("click", () => {
      document.querySelectorAll(".rating-option").forEach((l) => {
        l.style.borderColor = "transparent";
      });
      label.style.borderColor = "var(--accent)";
    });
  });
}

async function saveRating(agentId) {
  const selected = document.querySelector('input[name="rating"]:checked');
  if (!selected) {
    toast("Bitte eine Bewertung auswaehlen.", true);
    return;
  }
  const rating = selected.value;
  const note = document.getElementById("rating-note").value.trim() || null;
  try {
    await api(`/api/agents/${agentId}/rating`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ rating, note }),
    });
    closeModal();
    toast("Bewertung gespeichert.");
    const a = agentsData.find((x) => x.id === agentId);
    if (a) {
      a.current_rating = rating;
      a.current_rating_note = note;
      a.last_rated_at = new Date().toISOString();
    }
    renderAgentsList();
    selectAgent(agentId);
  } catch {
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
