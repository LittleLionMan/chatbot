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
