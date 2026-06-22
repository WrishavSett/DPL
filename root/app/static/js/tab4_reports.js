/*
 * Tab 4: report list is fetched on load (reports change hourly, not worth
 * a live socket for this). The "Generate for current hour" button is
 * mainly for testing/demo without waiting for the cron trigger.
 */
(function () {
  const tableBody = document.getElementById("reports-table-body");
  const generateBtn = document.getElementById("generate-now");

  function formatDate(iso) {
    return new Date(iso).toLocaleString();
  }

  function renderRow(report) {
    return `
      <tr>
        <td class="mono">${report.filename}</td>
        <td class="mono">${formatDate(report.period_start)}</td>
        <td class="mono">${formatDate(report.period_end)}</td>
        <td class="mono">${formatDate(report.generated_at)}</td>
        <td><a class="btn" href="/api/reports/${report.filename}">Download</a></td>
      </tr>
    `;
  }

  async function loadReports() {
    tableBody.innerHTML = `<tr><td colspan="5" class="text-muted">Loading…</td></tr>`;
    try {
      const res = await fetch("/api/reports");
      const reports = await res.json();
      if (!reports.length) {
        tableBody.innerHTML = `<tr><td colspan="5" class="text-muted">No reports generated yet.</td></tr>`;
        return;
      }
      tableBody.innerHTML = reports.map(renderRow).join("");
    } catch (err) {
      tableBody.innerHTML = `<tr><td colspan="5" class="text-muted">Failed to load reports: ${err}</td></tr>`;
    }
  }

  generateBtn.addEventListener("click", async () => {
    generateBtn.disabled = true;
    try {
      const res = await fetch("/api/reports/generate", { method: "POST" });
      if (!res.ok) {
        const detail = await res.text();
        alert(`Failed to generate report: ${detail}`);
      }
      await loadReports();
    } finally {
      generateBtn.disabled = false;
    }
  });

  loadReports();
})();