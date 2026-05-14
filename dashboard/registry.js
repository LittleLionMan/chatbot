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
