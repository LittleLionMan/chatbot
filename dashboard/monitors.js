async function loadMonitors() {
  try {
    monitorsData = await api("/api/monitors");
    renderMonitorsList();
  } catch {
    document.getElementById("monitors-list").innerHTML =
      '<div style="font-size:13px;color:var(--red);padding:8px;">Backend nicht erreichbar.</div>';
  }
}

function renderMonitorsList() {
  if (!monitorsData.length) {
    document.getElementById("monitors-list").innerHTML =
      '<div style="font-size:13px;color:var(--text3);padding:8px;">Keine Monitor-Configs.</div>';
    return;
  }
  document.getElementById("monitors-list").innerHTML = monitorsData
    .map((m) => {
      const sourceLabel =
        m.source === "static" ? "static" : `agent: ${m.source_agent}`;
      return `<div class="sidebar-item" data-id="${m.id}" onclick="selectMonitor(${m.id})">
        <div class="si-name">${m.name}<span class="badge ${m.is_active ? "badge-active" : "badge-inactive"}">${m.is_active ? "aktiv" : "inaktiv"}</span><span class="badge badge-pipeline">${m.monitor_type}</span></div>
        <div class="si-meta">${sourceLabel} → ${m.target_agent}</div>
      </div>`;
    })
    .join("");
}

async function selectMonitor(id) {
  currentMonitorId = id;
  document
    .querySelectorAll("#monitors-list .sidebar-item")
    .forEach((el) =>
      el.classList.toggle("active", Number(el.dataset.id) === id),
    );
  const m = monitorsData.find((x) => x.id === id);
  const detail = document.getElementById("monitors-detail");
  detail.innerHTML = '<div class="empty-state">Lade...</div>';
  showDetail(detail, document.getElementById("monitors-sidebar"), m.name);
  try {
    const seen = await api(`/api/monitors/${id}/seen?limit=20`);
    const intervalDisplay =
      m.poll_interval_seconds >= 3600
        ? `${m.poll_interval_seconds / 3600}h`
        : `${m.poll_interval_seconds / 60}min`;
    const feedsDisplay = (m.feed_templates || []).join("\n") || "--";
    const keywordsDisplay = (m.keywords || []).join(", ") || "--";
    detail.innerHTML = `
      <div class="action-bar">
        <button class="btn" onclick="editMonitor(${id})">Bearbeiten</button>
        <button class="btn" onclick="clearMonitorSeen(${id})">Seen leeren</button>
        ${m.is_active ? `<button class="btn btn-danger" onclick="stopMonitor(${id})">Stoppen</button>` : ""}
      </div>
      <div class="detail-block"><div class="detail-block-title">Konfiguration</div>
        <div class="detail-mono">Typ: ${m.monitor_type}</div>
        <div class="detail-mono">Source: ${m.source}</div>
        ${
          m.source === "agent"
            ? `
          <div class="detail-mono">Source-Agent: ${m.source_agent}</div>
          <div class="detail-mono">State-Key: ${m.source_state_key}</div>
          <div class="detail-mono">Format: ${m.source_format}</div>
        `
            : ""
        }
        <div class="detail-mono">Ziel-Agent: ${m.target_agent}</div>
        <div class="detail-mono">Intervall: ${intervalDisplay}</div>
        <div class="detail-mono">Keywords: ${keywordsDisplay}</div>
      </div>
      <div class="detail-block"><div class="detail-block-title">Feed-Templates / URLs</div>
        <pre class="detail-mono" style="white-space:pre-wrap;word-break:break-all;">${feedsDisplay}</pre>
      </div>
      <hr class="divider">
      <div class="detail-block">
        <div class="detail-block-title">Zuletzt gesehen (${seen.length})</div>
        <div class="mem-list">
          ${
            seen.length
              ? seen
                  .map(
                    (s) => `
            <div class="mem-row">
              <div class="mem-text" style="font-family:var(--mono);font-size:11px;">${s.fingerprint}</div>
              <div class="mem-date">${fmt(s.seen_at)}</div>
            </div>`,
                  )
                  .join("")
              : '<div style="font-size:13px;color:var(--text3);padding:8px 0;">Noch keine gesehenen Eintraege.</div>'
          }
        </div>
      </div>`;
  } catch {
    detail.innerHTML =
      '<div style="font-size:13px;color:var(--red);padding:1rem;">Fehler beim Laden.</div>';
  }
}

function editMonitor(id) {
  const m = monitorsData.find((x) => x.id === id);
  const feedsStr = (m.feed_templates || []).join("\n");
  const keywordsStr = (m.keywords || []).join(", ");
  openModal(
    "Monitor bearbeiten",
    `<div class="modal-field"><div class="modal-label">Name</div><input class="modal-input" id="mon-name" value="${m.name}" /></div>
     <div class="modal-field"><div class="modal-label">Ziel-Agent</div><input class="modal-input" id="mon-target" value="${m.target_agent}" /></div>
     <div class="modal-field"><div class="modal-label">Feed-Templates / URLs (eine pro Zeile)</div><textarea class="modal-input" id="mon-feeds" style="min-height:100px;">${feedsStr}</textarea></div>
     <div class="modal-field"><div class="modal-label">Keywords (kommasepariert)</div><input class="modal-input" id="mon-keywords" value="${keywordsStr}" /></div>
     <div class="modal-field"><div class="modal-label">Poll-Intervall (Sekunden)</div><input class="modal-input" id="mon-poll" type="number" value="${m.poll_interval_seconds}" /></div>
     ${
       m.source === "agent"
         ? `
       <div class="modal-field"><div class="modal-label">Source-Agent</div><input class="modal-input" id="mon-source-agent" value="${m.source_agent}" /></div>
       <div class="modal-field"><div class="modal-label">Source State-Key</div><input class="modal-input" id="mon-source-key" value="${m.source_state_key}" /></div>
       <div class="modal-field"><div class="modal-label">Source Format</div><input class="modal-input" id="mon-source-format" value="${m.source_format}" /></div>
     `
         : ""
     }`,
    `<button class="btn" onclick="closeModal()">Abbrechen</button>
     <button class="btn btn-accent" onclick="saveMonitor(${id})">Speichern</button>`,
  );
}

async function saveMonitor(id) {
  const m = monitorsData.find((x) => x.id === id);
  const name = document.getElementById("mon-name").value.trim();
  const target_agent = document.getElementById("mon-target").value.trim();
  const feedsRaw = document.getElementById("mon-feeds").value.trim();
  const feed_templates = feedsRaw
    ? feedsRaw
        .split("\n")
        .map((f) => f.trim())
        .filter(Boolean)
    : [];
  const keywordsRaw = document.getElementById("mon-keywords").value.trim();
  const keywords = keywordsRaw
    ? keywordsRaw
        .split(",")
        .map((k) => k.trim())
        .filter(Boolean)
    : [];
  const poll_interval_seconds =
    parseInt(document.getElementById("mon-poll").value) ||
    m.poll_interval_seconds;
  const body = {
    name,
    target_agent,
    feed_templates,
    keywords,
    poll_interval_seconds,
  };
  if (m.source === "agent") {
    body.source_agent =
      document.getElementById("mon-source-agent")?.value.trim() ||
      m.source_agent;
    body.source_state_key =
      document.getElementById("mon-source-key")?.value.trim() ||
      m.source_state_key;
    body.source_format =
      document.getElementById("mon-source-format")?.value.trim() ||
      m.source_format;
  }
  try {
    await api(`/api/monitors/${id}`, {
      method: "PATCH",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    });
    closeModal();
    toast("Gespeichert.");
    await loadMonitors();
    selectMonitor(id);
  } catch {
    toast("Fehler.", true);
  }
}

async function stopMonitor(id) {
  confirmModal("Monitor wirklich stoppen?", async () => {
    try {
      await api(`/api/monitors/${id}`, { method: "DELETE" });
      toast("Monitor gestoppt.");
      goBack();
      await loadMonitors();
    } catch {
      toast("Fehler.", true);
    }
  });
}

async function clearMonitorSeen(id) {
  confirmModal(
    "Alle gesehenen Eintraege loeschen? Der Monitor wird danach alle Artikel neu triggern.",
    async () => {
      try {
        await api(`/api/monitors/${id}/seen`, { method: "DELETE" });
        toast("Seen geleert.");
        selectMonitor(id);
      } catch {
        toast("Fehler.", true);
      }
    },
  );
}
