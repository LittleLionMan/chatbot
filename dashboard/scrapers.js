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
