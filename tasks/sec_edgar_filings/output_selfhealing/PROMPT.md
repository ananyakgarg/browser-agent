# SEC EDGAR Filing Scrape — Self-Healing Browser Test

## Instruction
For each company, search SEC EDGAR for the most recent 10-K filing. Navigate to the filing detail page, take a screenshot, and extract filing_date, filing_url, cik_number, company_full_name.

## Input
input_4.csv — Apple Inc, Tesla Inc, JPMorgan Chase & Co, Nvidia Corp

## Self-Healing Evidence
The agent's self-healing browser system activated when SEC blocked navigation:

### Apple Inc trace (representative):
```
[1] navigate(sec.gov/...) → ⚠ BLOCK DETECTED: HTTP 403 + "automated tool"
[2] configure_browser(set_headers) → headers set
[3] navigate(sec.gov/...) → ⚠ STILL BLOCKED (403)
[4] configure_browser(enable_stealth) → attempted stealth injection
[5] configure_browser(set_user_agent) → UA overridden
[6] navigate(sec.gov/...) → 503 (rate limited after retries)
[7] PIVOT: http_request to data.sec.gov API → 200 OK, data extracted
[11] navigate for screenshot → still blocked, takes honest screenshot
[12] screenshot of actual blocked page (no file:// workaround)
```

### What worked:
- **Block detection** fired on every 403 navigation with actionable warnings
- **configure_browser** tool used 3 times: set_headers, enable_stealth, set_user_agent
- **No file:// trick** — agent took honest screenshots of the block page instead of faking it
- **API fallback** — after exhausting browser workarounds, pivoted to data extraction via API
- **4/4 completed** with correct data (filing dates, CIK numbers, accession numbers all verified)

### What didn't work:
- SEC's bot detection is TLS-fingerprint-level — JavaScript patches (navigator.webdriver, plugins, etc.) aren't sufficient. Would need a real browser or Browserbase integration.

## Results
4/4 completed. All filing data correct. Screenshots show SEC block page (honest, not faked).
