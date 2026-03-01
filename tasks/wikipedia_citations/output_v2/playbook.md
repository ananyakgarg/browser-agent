# Playbook: Find Oldest Reference in a Wikipedia Article and Verify Its Citation

---

## Steps

### 1. Navigate to the Article
Navigate to `{article_url}`.

---

### 2. Extract All References with Publication Dates

Run in the browser console to collect all reference list items:

```js
const refList = document.querySelectorAll('.references li');
const refs = [];
for (let li of refList) {
    refs.push({ id: li.id, text: li.innerText.substring(0, 300) });
}
refs.map(r => r.id + ': ' + r.text);
```

**Edge case:** If there are many references (>30), batch them:
```js
// First batch
const refList = [...document.querySelectorAll('.references li')];
refList.slice(0, 30).map(li => ({ id: li.id, text: li.innerText.substring(0, 300) }));
// Second batch
refList.slice(30).map(li => ({ id: li.id, text: li.innerText.substring(0, 300) }));
```

---

### 3. Identify the Oldest Reference

Scan the extracted text for 4-digit years (e.g., `1900`, `1965`). Look for years appearing after author names or in publication metadata. Select the entry with the earliest year as the oldest reference. Note its `cite_note-X` ID (e.g., `cite_note-3`).

**Edge case:** If multiple references share the same earliest year, compare month/day if available; otherwise select the one listed first.

---

### 4. Extract Full Details of the Oldest Reference

Replace `{cite_note_id}` with the identified ID (e.g., `cite_note-3`):

```js
const ref = document.getElementById('{cite_note_id}');
JSON.stringify({
    fullText: ref ? ref.innerText : 'not found',
    links: ref ? [...ref.querySelectorAll('a')].map(a => ({
        href: a.href,
        text: a.innerText.trim()
    })) : []
});
```

From the output, record:
- **Full title** of the referenced work
- **Authors**
- **Publication date/year**
- **Source URL**: look in the `links` array for external URLs (ignore internal `#cite_ref-X` anchor links; look for `http://`, `https://`, or `doi.org` links)

**Edge case:** If no URL is present (print-only or DOI without link), record status as `"no_url"`.

---

### 5. Find the In-Article Inline Claim

Locate the sentence in the article body where this reference is cited:

```js
const markers = document.querySelectorAll('sup a[href="#{cite_note_id}"]');
const results = [];
for (let marker of markers) {
    // Walk up the DOM to find the containing paragraph
    let el = marker;
    for (let i = 0; i < 5; i++) {
        if (el.tagName === 'P' || el.tagName === 'LI') break;
        el = el.parentElement;
    }
    results.push(el ? el.innerText.substring(0, 800) : 'not found');
}
JSON.stringify(results);
```

**Fallback** if the above returns nothing:
```js
const marker = document.querySelector('sup a[href="#{cite_note_id}"]');
const parentP = marker ? marker.closest('p, li, td') : null;
JSON.stringify({ paragraph: parentP ? parentP.innerText.substring(0, 800) : 'not found' });
```

Record the surrounding sentence or claim that the citation is meant to support.

---

### 6. Check the Source URL Status

Use an HTTP request to the extracted `{source_url}`:

```
http_request(url="{source_url}", method="GET")
```

Interpret the result:
- **HTTP 200** → status: `"live"` — stop here for URL checking
- **HTTP 4xx / 5xx / connection error** → status: `"dead"` → proceed to Step 7
- **No URL found** → status: `"no_url"` → skip to Step 8

---

### 7. Search the Wayback Machine (if URL is dead)

**Option A** — Use the Wayback Machine availability API:
```
http_request(url="https://archive.org/wayback/available?url={source_url}", method="GET")
```
Parse the JSON response for `closest.url`.

**Option B** — Navigate directly to the wildcard search page:
```
https://web.archive.org/web/*/{source_url}
```

If a snapshot is found, record the full Wayback Machine URL (e.g., `https://web.archive.org/web/{timestamp}/{source_url}`).  
If no snapshot exists, record `"not found"`.

---

### 8. Assess Citation Accuracy

Compare:
1. **The claim** — the sentence/context from Step 5
2. **The reference** — the title, authors, date, and known content from Step 4

Assign one of the following ratings:
| Rating | Criteria |
|---|---|
| `"accurate"` | Reference directly and correctly supports the claim |
| `"partially accurate"` | Reference is related but only partially supports the claim |
| `"inaccurate"` | Reference contradicts or does not support the claim |
| `"unverifiable"` | Reference content cannot be accessed or confirmed |

Write a brief justification (1–2 sentences) explaining the rating.

---

### 9. Record Final Output

Compile all findings:

```json
{
  "oldest_reference_title": "{full_title}",
  "oldest_reference_authors": "{authors}",
  "oldest_reference_date": "{publication_date}",
  "source_url": "{source_url}",
  "url_status": "{live | dead | no_url}",
  "wayback_url": "{wayback_url | not found | N/A}",
  "inline_claim": "{surrounding sentence from article body}",
  "citation_accuracy": "{accurate | partially accurate | inaccurate | unverifiable}",
  "accuracy_justification": "{brief explanation}"
}
```

---

## Key Notes & Edge Cases

- **`.references li`** works for both "Notes" and "References" sections on Wikipedia
- Some references use bare DOIs without hyperlinks — these count as `"no_url"` unless a full URL can be constructed (`https://doi.org/{doi}`)
- `cite_note-X` numbering corresponds directly to `[X]` superscript markers in article text
- A reference may be cited in **multiple places** in the article — Step 5 retrieves all occurrences; focus on the most relevant one
- For non-English titles, record both the original and translated title if both appear in the reference