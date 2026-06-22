/*
 * Tab 3: one tile per camera, each with an <img> pointed at its MJPEG
 * endpoint. Tiles are created once per camera_id and never recreated on
 * subsequent status updates — only the badge/offline overlay inside an
 * existing tile is touched, so the live stream connection in <img> is
 * never reset by a routine status push.
 */
(function () {
  const grid = document.getElementById("video-grid");
  const tiles = new Map(); // camera_id -> tile element

  function statusBadge(status) {
    return `<span class="status-badge status-${status}"><span class="status-dot"></span>${status}</span>`;
  }

  function createTile(cameraId) {
    const tile = document.createElement("div");
    tile.className = "video-tile";
    tile.innerHTML = `
      <img src="/api/streams/${cameraId}/mjpeg" alt="${cameraId} live feed" />
      <div class="tile-status"></div>
    `;
    return tile;
  }

  function updateTile(tile, summary) {
    const cam = summary.camera;
    tile.querySelector(".tile-status").innerHTML = statusBadge(cam.status);

    const existingOffline = tile.querySelector(".tile-offline");
    if (cam.status !== "running") {
      if (!existingOffline) {
        const overlay = document.createElement("div");
        overlay.className = "tile-offline";
        overlay.textContent = `${cam.name} — ${cam.status}`;
        tile.appendChild(overlay);
      } else {
        existingOffline.textContent = `${cam.name} — ${cam.status}`;
      }
    } else if (existingOffline) {
      existingOffline.remove();
    }
  }

  function render(summaries) {
    if (!summaries.length) {
      grid.innerHTML = `<div class="empty-state">No cameras configured yet. Add one from the Manage tab.</div>`;
      tiles.clear();
      return;
    }

    if (grid.querySelector(".empty-state")) {
      grid.innerHTML = "";
    }

    const seen = new Set();
    for (const summary of summaries) {
      const cameraId = summary.camera.camera_id;
      seen.add(cameraId);
      let tile = tiles.get(cameraId);
      if (!tile) {
        tile = createTile(cameraId);
        tiles.set(cameraId, tile);
        grid.appendChild(tile);
      }
      updateTile(tile, summary);
    }

    // Remove tiles for cameras that no longer exist (e.g. removed in Tab 2).
    for (const [cameraId, tile] of tiles) {
      if (!seen.has(cameraId)) {
        tile.remove();
        tiles.delete(cameraId);
      }
    }
  }

  window.cameraStatusSocket.subscribe(render);
})();