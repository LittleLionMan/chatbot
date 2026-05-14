function renderStep(step, idx, agentId) {
  const summary = stepSummary(step);
  return `
    <div class="pipeline-step">
      <div class="pipeline-step-header">
        <div class="pipeline-step-info">
          <span class="pipeline-step-id">${step.id}</span>
          ${stepTypeBadge(step.type || "?")}
          ${step.is_output ? '<span class="badge badge-active">output</span>' : ""}
        </div>
        <div class="pipeline-step-actions">
          <button class="btn btn-sm" onclick="editStep(${agentId},${idx})">Bearbeiten</button>
          <button class="btn btn-sm btn-danger" onclick="deleteStep(${agentId},${idx})">Loeschen</button>
        </div>
      </div>
      ${summary ? `<div class="pipeline-step-prompt">${summary}</div>` : ""}
    </div>`;
}

function editStep(agentId, idx) {
  const a = agentsData.find((x) => x.id === agentId);
  const steps = a.steps || [];
  const step = steps[idx];
  _stepContextKeys = steps
    .slice(0, idx)
    .filter((s) => s.output_key)
    .map((s) => s.output_key);
  _stepRouteKeys = steps
    .slice(0, idx)
    .filter((s) => s.type === "router_match" || s.type === "router_llm")
    .flatMap((s) => {
      const routes = (s.rules || []).map((r) => r.then).filter(Boolean);
      if (s.default) routes.push(s.default);
      return routes;
    });
  const posField = `<div class="modal-field"><div class="modal-label">Position</div><input class="modal-input" id="sf-position" type="number" value="${idx}" /></div>`;
  openModal(
    "Step bearbeiten",
    posField + buildStepFormBody(step),
    `<button class="btn" onclick="closeModal()">Abbrechen</button><button class="btn btn-accent" onclick="saveStep(${agentId},${idx})">Speichern</button>`,
  );
}

async function saveStep(agentId, idx) {
  const a = agentsData.find((x) => x.id === agentId);
  const steps = [...(a.steps || [])];
  const updatedStep = _readStepFromForm();
  const posRaw = _val("sf-position");
  const newPos =
    posRaw !== "" && posRaw !== null && posRaw !== undefined
      ? parseInt(posRaw, 10)
      : idx;
  steps.splice(idx, 1);
  if (isNaN(newPos) || newPos >= steps.length) steps.push(updatedStep);
  else steps.splice(Math.max(0, newPos), 0, updatedStep);
  try {
    await api("/api/agents/" + agentId, {
      method: "PATCH",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ steps }),
    });
    a.steps = steps;
    closeModal();
    toast("Gespeichert.");
    selectAgent(agentId);
  } catch {
    toast("Fehler.", true);
  }
}

function deleteStep(agentId, idx) {
  confirmModal("Step wirklich loeschen?", async () => {
    const a = agentsData.find((x) => x.id === agentId);
    const steps = [...(a.steps || [])];
    steps.splice(idx, 1);
    try {
      await api("/api/agents/" + agentId, {
        method: "PATCH",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ steps }),
      });
      a.steps = steps;
      toast("Geloescht.");
      selectAgent(agentId);
    } catch {
      toast("Fehler.", true);
    }
  });
}

function addStep(agentId) {
  const a = agentsData.find((x) => x.id === agentId);
  const steps = a.steps || [];
  _stepContextKeys = steps.filter((s) => s.output_key).map((s) => s.output_key);
  _stepRouteKeys = steps
    .filter((s) => s.type === "router_match" || s.type === "router_llm")
    .flatMap((s) => {
      const routes = (s.rules || []).map((r) => r.then).filter(Boolean);
      if (s.default) routes.push(s.default);
      return routes;
    });
  const posField = `<div class="modal-field"><div class="modal-label">Position (leer = ans Ende)</div><input class="modal-input" id="sf-position" type="number" placeholder="0 = Anfang" /></div>`;
  openModal(
    "Neuen Step hinzufuegen",
    posField + buildStepFormBody({ id: "", type: "llm_extract" }),
    `<button class="btn" onclick="closeModal()">Abbrechen</button><button class="btn btn-accent" onclick="saveNewStep(${agentId})">Hinzufuegen</button>`,
  );
}

async function saveNewStep(agentId) {
  const step = _readStepFromForm();
  if (!step.id) {
    toast("ID ist Pflichtfeld.", true);
    return;
  }
  const posRaw = _val("sf-position");
  const pos =
    posRaw !== "" && posRaw !== null && posRaw !== undefined
      ? parseInt(posRaw, 10)
      : null;
  const a = agentsData.find((x) => x.id === agentId);
  const steps = [...(a.steps || [])];
  if (pos === null || isNaN(pos) || pos >= steps.length) steps.push(step);
  else steps.splice(pos, 0, step);
  try {
    await api("/api/agents/" + agentId, {
      method: "PATCH",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ steps }),
    });
    a.steps = steps;
    closeModal();
    toast("Hinzugefuegt.");
    selectAgent(agentId);
  } catch {
    toast("Fehler.", true);
  }
}
