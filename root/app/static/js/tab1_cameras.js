/*
 * Tab 1: renders one card per camera with its status badge and cumulative
 * classwise IN/OUT counts. All data comes from the shared status socket —
 * no separate fetch needed, since it pushes an initial payload right after
 * connecting.
 */
(function () {
  const container = document.getElementById("camera-cards");

  function statusBadge(status) {
    return `<span class="status-badge status-${status}"><span class="status-dot"></span>${status}</span>`;
  }

  function formatHeartbeat(iso) {
    if (!iso) return "—";
    return new Date(iso).toLocaleTimeString();
  }

  function renderCounts(counts) {
    if (!counts.length) {
      return `<div class="text-muted" style="font-size:13px;margin-top:10px;">No crossings recorded yet</div>`;
    }
    const items = counts
      .map(
        (c) =>
          `<li><span>${c.class_name}</span><span><span class="count-in">IN ${c.in_count}</span>` +
          `&nbsp;&nbsp;<span class="count-out">OUT ${c.out_count}</span></span></li>`
      )
      .join("");
    return `<ul class="count-list">${items}</ul>`;
  }

  function renderCard(summary) {
    const cam = summary.camera;
    return `
      <div class="card">
        <div class="card-title-row">
          <div>
            <div class="camera-name">${cam.name}</div>
            <div class="camera-id mono">${cam.camera_id}</div>
          </div>
          ${statusBadge(cam.status)}
        </div>
        <div class="text-muted" style="font-size:12px;">
          Last heartbeat: ${formatHeartbeat(cam.last_heartbeat)}
        </div>
        ${renderCounts(summary.counts)}
      </div>
    `;
  }

  function render(summaries) {
    if (!summaries.length) {
      container.innerHTML = `<div class="empty-state">No cameras configured yet. Add one from the Manage tab.</div>`;
      return;
    }
    container.innerHTML = summaries.map(renderCard).join("");
  }

  window.cameraStatusSocket.subscribe(render);
})();