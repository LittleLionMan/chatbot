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

async function loadAll() {
  STEP_TYPES = await api("/api/step-types").catch(() => []);
  loadAgents();
  loadMemory();
  loadScrapers();
  loadMonitors();
  loadUsage();
  loadRegistry();
  loadTriggers();
}
loadAll();
