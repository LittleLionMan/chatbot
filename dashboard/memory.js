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
  document
    .getElementById("mem-system-btn")
    .classList.toggle("active", mode === "system");
  document.getElementById("memory-detail").classList.remove("visible");
  document.getElementById("memory-sidebar").classList.remove("hidden");
  document.getElementById("topbar-back").classList.remove("visible");
  document.getElementById("topbar-dot").style.display = "";
  document.getElementById("topbar-name").textContent = "Bob Dashboard";

  if (mode === "system") {
    document.getElementById("memory-list").innerHTML = "";
    document.getElementById("memory-search").style.display = "none";
    renderSystemMemory();
  } else {
    document.getElementById("memory-search").style.display = "";
    document.getElementById("memory-detail").innerHTML =
      '<div class="empty-state">Eintrag auswaehlen</div>';
    renderMemoryList();
  }
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

async function renderSystemMemory() {
  const detail = document.getElementById("memory-detail");
  detail.innerHTML = '<div class="empty-state">Lade...</div>';

  if (window.innerWidth <= 768) {
    document.getElementById("memory-sidebar").classList.add("hidden");
    detail.classList.add("visible");
  }

  try {
    const [patternsRes, agentsRes] = await Promise.all([
      api("/api/skills/patterns"),
      api("/api/agents"),
    ]);

    const patterns = patternsRes.patterns;
    const isDirty = patternsRes.is_dirty;
    const updatedAt = patternsRes.updated_at;

    const statusColor = isDirty ? "var(--amber)" : "var(--accent)";
    const statusLabel = isDirty ? "Neuberechnung ausstehend" : "Aktuell";

    let patternsHtml = "";
    if (!patterns) {
      patternsHtml =
        '<div style="font-size:13px;color:var(--text3);padding:8px 0;">Noch keine Patterns extrahiert. Agenten bewerten oder Extraktion manuell starten.</div>';
    } else {
      const stepPatterns = patterns.step_patterns || [];
      const mistakes = patterns.common_mistakes || [];
      const antiPatterns = patterns.anti_patterns || [];
      const orderingRules = patterns.step_ordering_rules || [];
      const contextReqs = patterns.context_requirements || {};

      if (stepPatterns.length) {
        patternsHtml += `<div class="detail-block">
          <div class="detail-block-title">Bewaehrte Step-Muster (${stepPatterns.length})</div>`;
        stepPatterns.forEach((p) => {
          const conf =
            typeof p.confidence === "number"
              ? ` · Konfidenz: ${(p.confidence * 100).toFixed(0)}%`
              : "";
          const freqColor =
            p.frequency === "always"
              ? "var(--accent)"
              : p.frequency === "often"
                ? "var(--amber)"
                : "var(--text3)";
          patternsHtml += `<div class="data-card" style="margin-bottom:8px;">
            <div style="display:flex;align-items:center;gap:8px;margin-bottom:4px;flex-wrap:wrap;">
              <span class="badge" style="border:1px solid ${freqColor};color:${freqColor};background:transparent;">${p.frequency || "?"}</span>
              <span style="font-family:var(--mono);font-size:10px;color:var(--text3);">${conf}</span>
            </div>
            <div style="font-size:12px;color:var(--text2);margin-bottom:4px;"><strong style="color:var(--text);">Wenn:</strong> ${p.trigger || ""}</div>
            <div style="font-family:var(--mono);font-size:11px;color:var(--accent);margin-bottom:4px;">${p.pattern || ""}</div>
            <div style="font-size:12px;color:var(--text3);">${p.rationale || ""}</div>
          </div>`;
        });
        patternsHtml += `</div>`;
      }

      if (mistakes.length) {
        patternsHtml += `<div class="detail-block">
          <div class="detail-block-title">Haeufige LLM-Fehler (${mistakes.length})</div>`;
        mistakes.forEach((m) => {
          patternsHtml += `<div class="mem-row" style="flex-direction:column;align-items:flex-start;gap:4px;padding:8px 0;">
            <div style="font-size:12px;color:var(--red);">✗ ${m.mistake || ""}</div>
            <div style="font-size:12px;color:var(--accent);">✓ ${m.correction || ""}</div>
            ${m.source ? `<div style="font-size:10px;color:var(--text3);font-family:var(--mono);">${m.source}</div>` : ""}
          </div>`;
        });
        patternsHtml += `</div>`;
      }

      if (antiPatterns.length) {
        patternsHtml += `<div class="detail-block">
          <div class="detail-block-title">Anti-Patterns (${antiPatterns.length})</div>`;
        antiPatterns.forEach((a) => {
          patternsHtml += `<div class="mem-row" style="flex-direction:column;align-items:flex-start;gap:2px;padding:8px 0;">
            <div style="font-size:12px;color:var(--red);font-family:var(--mono);">${a.pattern || ""}</div>
            <div style="font-size:12px;color:var(--text3);">${a.reason || ""}</div>
          </div>`;
        });
        patternsHtml += `</div>`;
      }

      if (orderingRules.length) {
        patternsHtml += `<div class="detail-block">
          <div class="detail-block-title">Reihenfolge-Regeln (${orderingRules.length})</div>
          <div class="mem-list">`;
        orderingRules.forEach((r) => {
          patternsHtml += `<div class="mem-row"><div class="mem-text" style="font-size:12px;">${r}</div></div>`;
        });
        patternsHtml += `</div></div>`;
      }

      if (Object.keys(contextReqs).length) {
        patternsHtml += `<div class="detail-block">
          <div class="detail-block-title">Step-Voraussetzungen</div>
          <div class="mem-list">`;
        Object.entries(contextReqs).forEach(([step, preds]) => {
          patternsHtml += `<div class="mem-row">
            <div class="mem-text" style="font-size:12px;font-family:var(--mono);">
              <span style="color:var(--accent);">${step}</span>
              <span style="color:var(--text3);"> braucht davor: </span>
              <span style="color:var(--text2);">${Array.isArray(preds) ? preds.join(", ") : preds}</span>
            </div>
          </div>`;
        });
        patternsHtml += `</div></div>`;
      }
    }

    const ratedAgents = agentsRes.filter((a) => a.current_rating);
    const unratedAgents = agentsRes.filter(
      (a) => !a.current_rating && a.is_active,
    );

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

    let ratingsHtml = "";
    if (ratedAgents.length) {
      ratedAgents.forEach((a) => {
        const color = RATING_COLORS[a.current_rating] || "var(--text3)";
        const label = RATING_LABELS[a.current_rating] || a.current_rating;
        ratingsHtml += `<div class="mem-row">
          <div class="mem-text">
            <span style="font-size:13px;font-weight:500;">${a.name}</span>
            <span class="badge" style="border:1px solid ${color};color:${color};background:transparent;margin-left:6px;">${label}</span>
            ${a.current_rating_note ? `<div style="font-size:11px;color:var(--text3);margin-top:2px;">${a.current_rating_note}</div>` : ""}
          </div>
          <div class="mem-date">${a.last_rated_at ? fmt(a.last_rated_at) : ""}</div>
        </div>`;
      });
    }
    if (unratedAgents.length) {
      ratingsHtml += `<div style="font-size:11px;color:var(--text3);padding:8px 0;font-family:var(--mono);">Nicht bewertet: ${unratedAgents.map((a) => a.name).join(", ")}</div>`;
    }
    if (!ratingsHtml) {
      ratingsHtml =
        '<div style="font-size:13px;color:var(--text3);padding:8px 0;">Noch keine Bewertungen.</div>';
    }

    detail.innerHTML = `
      <div class="action-bar">
        <button class="btn btn-accent" onclick="triggerSkillExtraction()">Patterns neu berechnen</button>
      </div>
      <div class="detail-block">
        <div class="detail-block-title">Status</div>
        <div class="detail-mono">
          <span style="color:${statusColor};">● ${statusLabel}</span>
        </div>
        ${updatedAt ? `<div class="detail-mono">Letzte Extraktion: ${fmt(updatedAt)}</div>` : ""}
      </div>
      <hr class="divider">
      <div class="detail-block">
        <div class="detail-block-title">Agent-Bewertungen</div>
        <div class="mem-list">${ratingsHtml}</div>
      </div>
      <hr class="divider">
      ${patternsHtml || '<div class="empty-state">Keine Patterns.</div>'}`;
  } catch (e) {
    console.error("renderSystemMemory error:", e);
    detail.innerHTML =
      '<div style="font-size:13px;color:var(--red);padding:1rem;">Fehler beim Laden.</div>';
  }
}

async function triggerSkillExtraction() {
  try {
    await api("/api/skills/extract", { method: "POST" });
    toast("Extraktion beim naechsten Scheduler-Tick gestartet.");
    setTimeout(() => renderSystemMemory(), 3000);
  } catch {
    toast("Fehler.", true);
  }
}
