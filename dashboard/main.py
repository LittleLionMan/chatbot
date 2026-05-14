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
