// ============================================================
// background.js — Agent Auth Bridge
// ============================================================

/**
 * Playwright storage state format:
 * {
 *   cookies: [{ name, value, domain, path, httpOnly, secure, sameSite, expires }],
 *   origins: [{ origin, localStorage: [{ name, value }] }]
 * }
 *
 * Chrome's sameSite uses "unspecified"/"lax"/"strict"/"no_restriction",
 * Playwright uses "Lax"/"Strict"/"None".
 */

const SAME_SITE_MAP = {
  no_restriction: "None",
  lax: "Lax",
  strict: "Strict",
  unspecified: "None",
};

// Auto-export interval in minutes
const AUTO_EXPORT_INTERVAL_MIN = 15;
const AUTO_EXPORT_ALARM = "auto-export-state";
// Default export filename — goes to ~/Downloads but user should move to ~/.agent-auth/
const EXPORT_FILENAME = "agent-auth-state.json";

async function getAllCookies() {
  const chromeCookies = await chrome.cookies.getAll({});
  return chromeCookies.map((c) => ({
    name: c.name,
    value: c.value,
    domain: c.domain,
    path: c.path,
    httpOnly: c.httpOnly,
    secure: c.secure,
    sameSite: SAME_SITE_MAP[c.sameSite] || "None",
    expires: c.expirationDate ? Math.floor(c.expirationDate) : -1,
  }));
}

/**
 * Get localStorage from all open tabs.
 * Injects a content script into each tab to read window.localStorage.
 * Only works for tabs with http(s) URLs.
 */
async function getAllLocalStorage() {
  const origins = {};

  try {
    const tabs = await chrome.tabs.query({});

    for (const tab of tabs) {
      if (!tab.url || !tab.url.startsWith("http")) continue;

      try {
        const origin = new URL(tab.url).origin;
        if (origins[origin]) continue;

        const results = await chrome.scripting.executeScript({
          target: { tabId: tab.id },
          func: () => {
            const items = [];
            for (let i = 0; i < localStorage.length; i++) {
              const key = localStorage.key(i);
              items.push({ name: key, value: localStorage.getItem(key) });
            }
            return items;
          },
        });

        if (results && results[0] && results[0].result) {
          origins[origin] = results[0].result;
        }
      } catch (e) {
        console.debug(
          `Could not read localStorage for tab ${tab.id}: ${e.message}`
        );
      }
    }
  } catch (e) {
    console.error("Error reading tabs:", e);
  }

  return Object.entries(origins).map(([origin, items]) => ({
    origin,
    localStorage: items,
  }));
}

/**
 * Filter cookies/localStorage to specific domains if requested.
 */
function filterByDomains(storageState, domains) {
  if (!domains || domains.length === 0) return storageState;

  const domainSet = new Set(domains.map((d) => d.toLowerCase()));

  const matchesDomain = (cookieDomain) => {
    const clean = cookieDomain.startsWith(".")
      ? cookieDomain.slice(1)
      : cookieDomain;
    return (
      domainSet.has(clean) ||
      [...domainSet].some(
        (d) => clean.endsWith(`.${d}`) || d.endsWith(`.${clean}`) || d === clean
      )
    );
  };

  return {
    cookies: storageState.cookies.filter((c) => matchesDomain(c.domain)),
    origins: storageState.origins.filter((o) => {
      try {
        const host = new URL(o.origin).hostname;
        return matchesDomain(host);
      } catch {
        return false;
      }
    }),
  };
}

async function buildStorageState(domains = null) {
  const cookies = await getAllCookies();
  const origins = await getAllLocalStorage();
  let state = { cookies, origins };
  if (domains && domains.length > 0) {
    state = filterByDomains(state, domains);
  }
  return state;
}

/**
 * Export storage state as a downloaded JSON file.
 */
async function exportToFile(domains = null) {
  const state = await buildStorageState(domains);
  const blob = new Blob([JSON.stringify(state, null, 2)], {
    type: "application/json",
  });
  const url = URL.createObjectURL(blob);

  await chrome.downloads.download({
    url: url,
    filename: EXPORT_FILENAME,
    saveAs: false,
    conflictAction: "overwrite",
  });

  return {
    success: true,
    cookieCount: state.cookies.length,
    originCount: state.origins.length,
  };
}

// ============================================================
// Auto-export via chrome.alarms
// ============================================================

async function setupAutoExport() {
  const settings = await chrome.storage.local.get(["autoExport"]);
  if (settings.autoExport === false) return;

  // Default: auto-export enabled
  chrome.alarms.create(AUTO_EXPORT_ALARM, {
    periodInMinutes: AUTO_EXPORT_INTERVAL_MIN,
  });
  console.log(
    `Auto-export alarm set: every ${AUTO_EXPORT_INTERVAL_MIN} minutes`
  );
}

chrome.runtime.onInstalled.addListener(() => {
  setupAutoExport();
});

chrome.runtime.onStartup.addListener(() => {
  setupAutoExport();
});

chrome.alarms.onAlarm.addListener(async (alarm) => {
  if (alarm.name === AUTO_EXPORT_ALARM) {
    try {
      const result = await exportToFile();
      console.log(
        `Auto-exported: ${result.cookieCount} cookies, ${result.originCount} origins`
      );
    } catch (e) {
      console.error("Auto-export failed:", e);
    }
  }
});

// ============================================================
// Message handler — popup communicates via messages
// ============================================================

chrome.runtime.onMessage.addListener((request, sender, sendResponse) => {
  if (request.action === "export") {
    buildStorageState(request.domains)
      .then((state) => {
        sendResponse({
          success: true,
          state: state,
          cookieCount: state.cookies.length,
          originCount: state.origins.length,
        });
      })
      .catch((err) => {
        sendResponse({ success: false, error: err.message });
      });
    return true;
  }

  if (request.action === "exportToFile") {
    exportToFile(request.domains)
      .then((result) => sendResponse(result))
      .catch((err) => sendResponse({ success: false, error: err.message }));
    return true;
  }

  if (request.action === "status") {
    getAllCookies().then((cookies) => {
      const domainCounts = {};
      cookies.forEach((c) => {
        const domain = c.domain.startsWith(".") ? c.domain.slice(1) : c.domain;
        domainCounts[domain] = (domainCounts[domain] || 0) + 1;
      });
      sendResponse({
        totalCookies: cookies.length,
        domains: domainCounts,
      });
    });
    return true;
  }

  if (request.action === "setAutoExport") {
    chrome.storage.local.set({ autoExport: request.enabled });
    if (request.enabled) {
      setupAutoExport();
      sendResponse({ success: true, enabled: true });
    } else {
      chrome.alarms.clear(AUTO_EXPORT_ALARM);
      sendResponse({ success: true, enabled: false });
    }
    return true;
  }

  if (request.action === "getAutoExport") {
    chrome.storage.local.get(["autoExport"]).then((settings) => {
      sendResponse({ enabled: settings.autoExport !== false });
    });
    return true;
  }
});
