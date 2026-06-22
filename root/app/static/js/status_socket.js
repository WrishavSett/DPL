/*
 * Shared WebSocket connection to /ws/status, loaded once in base.html so
 * every tab gets live camera data without opening its own connection.
 * Other tab scripts call window.cameraStatusSocket.subscribe(fn) to
 * receive each update; fn receives the array of camera summaries in the
 * same shape as GET /api/cameras returns.
 */
(function () {
  const listeners = new Set();
  let socket = null;
  let reconnectDelay = 1000;
  const MAX_RECONNECT_DELAY = 15000;

  function wsUrl() {
    const protocol = location.protocol === "https:" ? "wss:" : "ws:";
    return `${protocol}//${location.host}/ws/status`;
  }

  function updateFleetSummary(summaries) {
    const el = document.getElementById("fleet-summary");
    if (!el) return;
    const total = summaries.length;
    const online = summaries.filter((s) => s.camera.status === "running").length;
    el.textContent = `${online} / ${total} online`;
  }

  function connect() {
    socket = new WebSocket(wsUrl());

    socket.onopen = () => {
      reconnectDelay = 1000;
    };

    socket.onmessage = (event) => {
      let summaries;
      try {
        summaries = JSON.parse(event.data);
      } catch (err) {
        console.error("Bad status payload", err);
        return;
      }
      updateFleetSummary(summaries);
      listeners.forEach((fn) => fn(summaries));
    };

    socket.onclose = () => {
      setTimeout(connect, reconnectDelay);
      reconnectDelay = Math.min(reconnectDelay * 2, MAX_RECONNECT_DELAY);
    };

    socket.onerror = () => {
      socket.close();
    };
  }

  connect();

  window.cameraStatusSocket = {
    /** fn(summaries) is called on every status push. Returns an unsubscribe function. */
    subscribe(fn) {
      listeners.add(fn);
      return () => listeners.delete(fn);
    },
  };
})();