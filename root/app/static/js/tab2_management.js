/*
 * Tab 2: live management table (status/enabled come from the shared
 * status socket) plus the add-camera form and per-row enable/disable/
 * restart/remove actions, all hitting the /api/cameras endpoints.
 */
(function () {
  const tableBody = document.getElementById("camera-table-body");
  const form = document.getElementById("add-camera-form");
  const errorBox = document.getElementById("add-camera-error");

  function statusBadge(status) {
    return `<span class="status-badge status-${status}"><span class="status-dot"></span>${status}</span>`;
  }

  function renderRow(summary) {
    const cam = summary.camera;
    return `
      <tr data-camera-id="${cam.camera_id}">
        <td>
          <div>${cam.name}</div>
          <div class="mono text-muted" style="font-size:12px;">${cam.camera_id}</div>
        </td>
        <td class="mono" style="font-size:12px;">${cam.source}</td>
        <td>${statusBadge(cam.status)}</td>
        <td>${cam.enabled ? "Yes" : "No"}</td>
        <td>
          <div class="btn-row">
            <button data-action="${cam.enabled ? "disable" : "enable"}">${cam.enabled ? "Disable" : "Enable"}</button>
            <button data-action="restart">Restart</button>
            <button data-action="remove" class="danger">Remove</button>
          </div>
        </td>
      </tr>
    `;
  }

  function render(summaries) {
    if (!summaries.length) {
      tableBody.innerHTML = `<tr><td colspan="5" class="text-muted">No cameras configured yet.</td></tr>`;
      return;
    }
    tableBody.innerHTML = summaries.map(renderRow).join("");
  }

  window.cameraStatusSocket.subscribe(render);

  tableBody.addEventListener("click", async (event) => {
    const button = event.target.closest("button[data-action]");
    if (!button) return;
    const row = button.closest("tr");
    const cameraId = row.dataset.cameraId;
    const action = button.dataset.action;

    if (action === "remove" && !confirm(`Remove camera "${cameraId}"? This deletes its recorded counts too.`)) {
      return;
    }

    button.disabled = true;
    try {
      const method = action === "remove" ? "DELETE" : "POST";
      const url = action === "remove" ? `/api/cameras/${cameraId}` : `/api/cameras/${cameraId}/${action}`;
      const res = await fetch(url, { method });
      if (!res.ok) {
        const detail = await res.text();
        alert(`Action failed: ${detail}`);
      }
    } catch (err) {
      alert(`Action failed: ${err}`);
    } finally {
      button.disabled = false;
    }
  });

  form.addEventListener("submit", async (event) => {
    event.preventDefault();
    errorBox.textContent = "";

    const data = new FormData(form);
    const classesRaw = (data.get("classes") || "").trim();
    const classes = classesRaw ? classesRaw.split(",").map((c) => c.trim()).filter(Boolean) : null;

    const x1 = data.get("x1"), y1 = data.get("y1"), x2 = data.get("x2"), y2 = data.get("y2");
    const hasLine = [x1, y1, x2, y2].every((v) => v !== "");
    const count_line = hasLine
      ? { x1: Number(x1), y1: Number(y1), x2: Number(x2), y2: Number(y2) }
      : null;

    const payload = {
      camera_id: data.get("camera_id"),
      name: data.get("name"),
      source: data.get("source"),
      enabled: data.get("enabled") === "on",
      classes,
      count_line,
      model_path: data.get("model_path") || null,
      device: data.get("device"),
      target_w: Number(data.get("target_w")),
      target_h: Number(data.get("target_h")),
      conf_threshold: Number(data.get("conf_threshold")),
      iou_threshold: Number(data.get("iou_threshold")),
      lost_track_buffer: 30,
    };

    try {
      const res = await fetch("/api/cameras", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
      });
      if (!res.ok) {
        const detail = await res.json().catch(() => ({}));
        errorBox.textContent = detail.detail || `Failed to add camera (${res.status})`;
        return;
      }
      form.reset();
    } catch (err) {
      errorBox.textContent = String(err);
    }
  });
})();