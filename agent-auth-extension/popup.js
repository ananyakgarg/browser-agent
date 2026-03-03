// Load status on popup open
chrome.runtime.sendMessage({ action: "status" }, (response) => {
  if (!response) return;

  document.getElementById("cookieCount").textContent = response.totalCookies;
  document.getElementById("domainCount").textContent =
    Object.keys(response.domains).length;

  const sorted = Object.entries(response.domains)
    .sort((a, b) => b[1] - a[1])
    .slice(0, 20);

  const list = document.getElementById("domainList");
  list.innerHTML = sorted
    .map(
      ([domain, count]) =>
        `<div class="domain-row">
      <span class="name">${domain}</span>
      <span class="count">${count}</span>
    </div>`
    )
    .join("");
});

// Load auto-export toggle state
chrome.runtime.sendMessage({ action: "getAutoExport" }, (response) => {
  if (response) {
    document.getElementById("autoExportToggle").checked = response.enabled;
  }
});

document.getElementById("autoExportToggle").addEventListener("change", (e) => {
  chrome.runtime.sendMessage({
    action: "setAutoExport",
    enabled: e.target.checked,
  });
});

function getDomainFilter() {
  const val = document.getElementById("domainFilter").value.trim();
  if (!val) return null;
  return val
    .split(",")
    .map((d) => d.trim())
    .filter(Boolean);
}

function showResult(msg, isError = false) {
  const el = document.getElementById("result");
  el.textContent = msg;
  el.className = `result ${isError ? "error" : "success"}`;
  setTimeout(() => {
    el.className = "result";
  }, 4000);
}

// Export to file
document.getElementById("exportBtn").addEventListener("click", () => {
  const btn = document.getElementById("exportBtn");
  btn.disabled = true;
  btn.textContent = "Exporting...";

  chrome.runtime.sendMessage(
    { action: "exportToFile", domains: getDomainFilter() },
    (response) => {
      btn.disabled = false;
      btn.textContent = "Export to File";

      if (response && response.success) {
        showResult(
          `Exported ${response.cookieCount} cookies, ${response.originCount} origins`
        );
      } else {
        showResult(`${response?.error || "Export failed"}`, true);
      }
    }
  );
});

// Copy to clipboard
document.getElementById("copyBtn").addEventListener("click", () => {
  const btn = document.getElementById("copyBtn");
  btn.disabled = true;

  chrome.runtime.sendMessage(
    { action: "export", domains: getDomainFilter() },
    (response) => {
      btn.disabled = false;

      if (response && response.success) {
        const json = JSON.stringify(response.state, null, 2);
        navigator.clipboard
          .writeText(json)
          .then(() => {
            showResult(`Copied ${response.cookieCount} cookies to clipboard`);
          })
          .catch(() => {
            showResult("Clipboard write failed", true);
          });
      } else {
        showResult(`${response?.error || "Export failed"}`, true);
      }
    }
  );
});
