# GitHub Blame Audit Playbook

## Overview
Audit a specific code string in a GitHub repo by inspecting its blame history and determining if any material computational changes occurred within the past year.

---

## Step 1: Navigate to Blame View

1. Navigate to `{github_blame_url}` (e.g., `https://github.com/{org}/{repo}/blame/main/{file_path}`)
2. If the page 404s, replace `main` with `master` in the URL
3. Take a screenshot: `screenshot({"filename": "blame_view_initial.png"})`

---

## Step 2: Find the Matching Line

4. Use `search_page({"query": "{code_string}"})` to locate the line
   - Note the **file line number** shown in the search result context (e.g., `L191: 28 >>> L192: def linspace(...)`)
   - The number adjacent to the match is the actual file line number

---

## Step 3: Extract Blame Info for That Line

5. Scroll to the matched line and extract blame metadata using:

```javascript
// Replace "28" with the actual line number
const lineEl = document.querySelector('[data-line-number="{line_number}"]');
let el = lineEl;
while (el && !el.classList.contains('react-blame-segment-wrapper')) {
  el = el.parentElement;
}
if (el) {
  const commitLinks = Array.from(el.querySelectorAll('a[href*="/commit/"]')).map(a => ({
    text: a.textContent.trim(),
    href: a.href
  }));
  const timeEls = Array.from(el.querySelectorAll('relative-time, time')).map(t => ({
    text: t.textContent.trim(),
    datetime: t.getAttribute('datetime')
  }));
  const segText = el.textContent.trim().slice(0, 200);
  ({segText, commitLinks, timeEls});
}
```

6. Extract from result:
   - **Commit hash**: from `commitLinks[0].href` (e.g., `.../commit/d338490271238c78683640943a11f9e4e53e62fb`)
   - **Commit date**: from `timeEls[0].getAttribute('datetime')` (ISO format)
   - **Commit message**: from `commitLinks[0].text`

---

## Step 4: Get Commit Author

7. Call GitHub API to get author name:

```python
import urllib.request, json
url = "https://api.github.com/repos/{org}/{repo}/commits/{commit_hash}"
req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0", "Accept": "application/vnd.github.v3+json"})
with urllib.request.urlopen(req) as r:
    data = json.loads(r.read())
print(data['commit']['author']['name'])
print(data['commit']['author']['email'])
print(data['commit']['author']['date'])
```

---

## Step 5: Determine `modified_within_year`

8. Compare the commit date to today's date (use the `Date` header from any HTTP response for current date):

```python
from datetime import date
commit_date = date(YYYY, MM, DD)
today = date(YYYY, MM, DD)  # From HTTP response header
months_diff = (today.year - commit_date.year) * 12 + (today.month - commit_date.month)
modified_within_year = months_diff <= 12
```

9. Take a screenshot of the blame gutter showing the matched line: `screenshot({"filename": "blame_line{line_number}.png"})`

---

## Step 6a: If `modified_within_year = TRUE` — Inspect Commit Diff

10. Run: `get_commit_diff({"repo_url": "{repo_url}", "commit_hash": "{commit_hash}", "file_path": "{file_path}"})`
11. Read the diff carefully. Flag as **material** only if it changed:
    - A formula or calculation
    - Parameter handling or default values
    - Output behavior or return values
12. Do **NOT** flag: whitespace, docstrings, comment edits, f-string conversions, `elif` refactors, import reordering

---

## Step 6b: If `modified_within_year = FALSE` — Scan Nearby Lines for Recent Changes

13. Scan all blame segments on the page for recent commits:

```javascript
const cutoffDate = new Date('{one_year_ago_ISO}');  // e.g., '2025-03-01'
const recentLines = [];
const segments = document.querySelectorAll('.react-blame-segment-wrapper');
for (const seg of segments) {
  const timeEl = seg.querySelector('relative-time, time');
  if (!timeEl) continue;
  const dt = new Date(timeEl.getAttribute('datetime'));
  if (dt >= cutoffDate) {
    const lineNums = Array.from(seg.querySelectorAll('[data-line-number]')).map(e => e.getAttribute('data-line-number'));
    const link = seg.querySelector('a[href*="/commit/"]');
    recentLines.push({
      datetime: dt.toISOString(),
      lineNums,
      commitMsg: link?.textContent.trim(),
      commitHref: link?.href
    });
  }
}
recentLines;
```

14. For each recent commit found, run `get_commit_diff(...)` to inspect what changed
15. Assess materiality using the same criteria as Step 6a
16. Take screenshots of any flagged diffs

---

## Step 7: Compile and Return Results

17. Call `get_function_source({"repo_url": "{repo_url}", "file_path": "{file_path}", "function_name": "{function_name}"})` to confirm exact line range of the function
18. Call `complete()` with all output fields:

```json
{
  "file_path": "{file_path}",
  "line_number": {line_number},
  "commit_hash": "{full_commit_hash}",
  "commit_author": "{author_name}",
  "commit_date": "{YYYY-MM-DD}",
  "blame_url": "{actual_blame_url_visited}",
  "modified_within_year": true/false,
  "material_change_flag": true/false,
  "material_change_explanation": "...",
  "material_change_evidence": "verbatim diff lines or blame text",
  "evidence_screenshots": ["blame_view_initial.png", "blame_line{N}.png", ...]
}
```

---

## Key Gotchas

- **Line number in search results**: The `search_page` result shows surrounding context — the file line number appears as a bare number (e.g., `28`) adjacent to the matched code, not as the page's `L{N}` reference
- **Blame segment DOM structure**: GitHub's React blame UI uses `.react-blame-segment-wrapper` as the container; walk up from `[data-line-number]` elements to find it
- **Author extraction**: The blame gutter avatar links don't reliably expose username text — use the GitHub API instead
- **Materiality threshold**: Style changes (f-strings, `elif` refactors, import reordering, docstring edits) are **never** material — only formula/logic/parameter changes qualify
- **Current date**: Read from the `Date` HTTP response header of any page request for an accurate reference date