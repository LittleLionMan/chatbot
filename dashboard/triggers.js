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
