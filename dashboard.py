"""Live + post-mortem web dashboard for browser automation runs.

Live mode:  Started automatically via --dashboard flag.
Post-mortem: python dashboard.py output/task_name

Stack: FastAPI + WebSocket + vanilla JS (inline HTML, no build step).
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import sys
import threading
import time
import webbrowser
from pathlib import Path
from typing import Any

from observability import WorkerEvent

logger = logging.getLogger(__name__)

DASHBOARD_PORT = 8484

# ---------------------------------------------------------------------------
# Inline HTML/JS dashboard (single-page app)
# ---------------------------------------------------------------------------

DASHBOARD_HTML = """\
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Browser Agent Dashboard</title>
<style>
  * { margin: 0; padding: 0; box-sizing: border-box; }
  body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
         background: #0d1117; color: #c9d1d9; }
  .header { background: #161b22; padding: 16px 24px; border-bottom: 1px solid #30363d;
            display: flex; align-items: center; gap: 20px; }
  .header h1 { font-size: 18px; color: #58a6ff; }
  .header .stats { display: flex; gap: 16px; font-size: 14px; }
  .progress-bar { flex: 1; height: 8px; background: #21262d; border-radius: 4px;
                  overflow: hidden; min-width: 200px; }
  .progress-fill { height: 100%; border-radius: 4px; transition: width 0.3s; }
  .progress-fill.success { background: #3fb950; }
  .progress-fill.fail { background: #f85149; }
  .badge { padding: 2px 8px; border-radius: 12px; font-size: 12px; font-weight: 600; }
  .badge.completed { background: #238636; color: #fff; }
  .badge.failed { background: #da3633; color: #fff; }
  .badge.in_progress { background: #1f6feb; color: #fff; }
  .badge.pending { background: #30363d; color: #8b949e; }

  .controls { padding: 12px 24px; background: #161b22; border-bottom: 1px solid #30363d;
              display: flex; gap: 12px; align-items: center; }
  .controls button { padding: 6px 12px; border-radius: 6px; border: 1px solid #30363d;
                    background: #21262d; color: #c9d1d9; cursor: pointer; font-size: 13px; }
  .controls button:hover { background: #30363d; }
  .controls button.active { background: #1f6feb; border-color: #1f6feb; color: #fff; }

  .main { display: flex; height: calc(100vh - 100px); }
  .sample-list { width: 400px; overflow-y: auto; border-right: 1px solid #30363d; }
  .sample-row { padding: 10px 16px; border-bottom: 1px solid #21262d; cursor: pointer;
                display: flex; align-items: center; gap: 10px; }
  .sample-row:hover { background: #161b22; }
  .sample-row.selected { background: #1c2128; border-left: 3px solid #58a6ff; }
  .sample-row .name { flex: 1; font-size: 14px; white-space: nowrap;
                      overflow: hidden; text-overflow: ellipsis; }
  .sample-row .iter { font-size: 12px; color: #8b949e; }
  .sample-row .error-msg { font-size: 12px; color: #f85149; margin-top: 2px;
                           white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }

  .detail { flex: 1; overflow-y: auto; padding: 20px; }
  .detail h2 { color: #58a6ff; margin-bottom: 16px; font-size: 16px; }
  .detail .empty { color: #8b949e; font-style: italic; padding: 40px; text-align: center; }

  .trace-step { margin-bottom: 12px; border: 1px solid #30363d; border-radius: 8px;
                overflow: hidden; }
  .trace-step.failed { border-color: #f85149; }
  .trace-step.warning { border-color: #d29922; }
  .step-header { padding: 8px 12px; background: #161b22; display: flex;
                 align-items: center; gap: 8px; font-size: 13px; cursor: pointer; }
  .step-header .iter-badge { background: #30363d; padding: 2px 6px;
                             border-radius: 4px; font-size: 11px; }
  .step-header .tool-name { color: #d2a8ff; font-weight: 600; }
  .step-header .time { margin-left: auto; color: #8b949e; font-size: 11px; }
  .step-body { padding: 12px; background: #0d1117; font-size: 13px;
               display: none; }
  .step-body.open { display: block; }
  .step-body pre { background: #161b22; padding: 10px; border-radius: 6px;
                   overflow-x: auto; margin: 6px 0; font-size: 12px;
                   white-space: pre-wrap; word-break: break-word; }
  .step-body .label { color: #8b949e; font-size: 11px; text-transform: uppercase;
                      margin-top: 8px; margin-bottom: 4px; }
  .step-body .reasoning { color: #c9d1d9; white-space: pre-wrap; margin-bottom: 8px; }
  .step-body .warning-text { color: #d29922; font-weight: 600; margin: 4px 0; }

  .screenshot { max-width: 100%; border-radius: 6px; margin: 8px 0;
                border: 1px solid #30363d; }

  .diagnosis-panel { margin-top: 20px; padding: 16px; background: #161b22;
                     border-radius: 8px; border: 1px solid #30363d; }
  .diagnosis-panel h3 { color: #f0883e; margin-bottom: 8px; font-size: 14px; }
  .diagnosis-panel .content { font-size: 13px; white-space: pre-wrap; line-height: 1.6; }

  .connection-status { font-size: 12px; padding: 2px 8px; border-radius: 4px; }
  .connection-status.connected { color: #3fb950; }
  .connection-status.disconnected { color: #f85149; }
</style>
</head>
<body>
<div class="header">
  <h1 id="taskName">Browser Agent Dashboard</h1>
  <div class="stats">
    <span id="elapsed">0:00</span>
    <span id="connStatus" class="connection-status disconnected">disconnected</span>
  </div>
  <div class="progress-bar">
    <div id="progressFill" class="progress-fill success" style="width:0%"></div>
  </div>
  <div class="stats">
    <span id="progressText">0/0</span>
  </div>
</div>

<div class="controls">
  <button class="active" onclick="filterStatus('all')">All</button>
  <button onclick="filterStatus('failed')">Failed</button>
  <button onclick="filterStatus('in_progress')">In Progress</button>
  <button onclick="filterStatus('completed')">Completed</button>
  <button onclick="filterStatus('pending')">Pending</button>
</div>

<div class="main">
  <div class="sample-list" id="sampleList"></div>
  <div class="detail" id="detail">
    <div class="empty">Select a sample to view its trace</div>
  </div>
</div>

<script>
const MODE = '{{MODE}}';  // 'live' or 'postmortem'
let samples = {};  // sample_id -> {status, error, iterations, events[]}
let selectedSample = null;
let currentFilter = 'all';
let startTime = Date.now();
let ws = null;

// Initialize
if (MODE === 'live') {
  connectWebSocket();
} else {
  loadPostMortemData();
}

function connectWebSocket() {
  const proto = location.protocol === 'https:' ? 'wss:' : 'ws:';
  ws = new WebSocket(`${proto}//${location.host}/ws`);

  ws.onopen = () => {
    document.getElementById('connStatus').textContent = 'live';
    document.getElementById('connStatus').className = 'connection-status connected';
  };

  ws.onclose = () => {
    document.getElementById('connStatus').textContent = 'disconnected';
    document.getElementById('connStatus').className = 'connection-status disconnected';
    // Reconnect after 2s
    setTimeout(connectWebSocket, 2000);
  };

  ws.onmessage = (e) => {
    const event = JSON.parse(e.data);
    handleEvent(event);
  };
}

function handleEvent(event) {
  const sid = event.sample_id;
  if (!samples[sid]) {
    samples[sid] = { status: 'pending', error: null, iterations: 0, events: [] };
  }

  samples[sid].events.push(event);

  if (event.event_type === 'status_change') {
    samples[sid].status = event.status || samples[sid].status;
    if (event.error) samples[sid].error = event.error;
  }
  if (event.event_type === 'tool_start') {
    samples[sid].status = 'in_progress';
    if (event.iteration) samples[sid].iterations = event.iteration;
  }
  if (event.event_type === 'tool_end') {
    if (event.iteration) samples[sid].iterations = event.iteration;
  }

  renderSampleList();
  if (selectedSample === sid) renderDetail(sid);
  updateProgress();
}

async function loadPostMortemData() {
  document.getElementById('connStatus').textContent = 'post-mortem';
  document.getElementById('connStatus').className = 'connection-status connected';

  try {
    const resp = await fetch('/api/samples');
    const data = await resp.json();

    document.getElementById('taskName').textContent = data.task_name || 'Post-mortem';

    for (const s of data.samples) {
      samples[s.sample_id] = {
        status: s.status,
        error: s.error,
        iterations: s.iterations || 0,
        events: [],
        trace: s.trace || [],
        metadata: s.metadata || null,
      };
    }

    renderSampleList();
    updateProgress();
  } catch (e) {
    document.getElementById('detail').innerHTML =
      '<div class="empty">Failed to load data: ' + e.message + '</div>';
  }
}

function filterStatus(status) {
  currentFilter = status;
  document.querySelectorAll('.controls button').forEach(b => b.classList.remove('active'));
  event.target.classList.add('active');
  renderSampleList();
}

function renderSampleList() {
  const list = document.getElementById('sampleList');
  const entries = Object.entries(samples);

  // Sort: failed first, then in_progress, then pending, then completed
  const order = { failed: 0, in_progress: 1, pending: 2, completed: 3 };
  entries.sort((a, b) => (order[a[1].status] ?? 4) - (order[b[1].status] ?? 4));

  let html = '';
  for (const [sid, s] of entries) {
    if (currentFilter !== 'all' && s.status !== currentFilter) continue;

    const selected = sid === selectedSample ? ' selected' : '';
    html += `<div class="sample-row${selected}" onclick="selectSample('${sid.replace(/'/g, "\\'")}')">
      <span class="badge ${s.status}">${s.status}</span>
      <span class="name">${sid}</span>
      <span class="iter">${s.iterations ? '#' + s.iterations : ''}</span>
    </div>`;
    if (s.error) {
      html = html.slice(0, -6) +
        `<div class="error-msg" title="${s.error}">${s.error}</div></div>`;
    }
  }

  list.innerHTML = html || '<div class="empty">No samples</div>';
}

function selectSample(sid) {
  selectedSample = sid;
  renderSampleList();
  renderDetail(sid);
}

async function renderDetail(sid) {
  const detail = document.getElementById('detail');
  const s = samples[sid];
  if (!s) { detail.innerHTML = '<div class="empty">Sample not found</div>'; return; }

  let html = `<h2>${sid} <span class="badge ${s.status}">${s.status}</span></h2>`;

  // Load trace for post-mortem mode
  let traceSteps = s.trace || [];
  if (MODE === 'postmortem' && traceSteps.length === 0) {
    try {
      const resp = await fetch('/api/trace/' + encodeURIComponent(sid));
      const data = await resp.json();
      traceSteps = data.trace || [];
      s.trace = traceSteps;
    } catch (e) { /* ignore */ }
  }

  // Build trace from events for live mode
  if (MODE === 'live' && s.events.length > 0) {
    const eventsByIter = {};
    for (const ev of s.events) {
      const iter = ev.iteration || 0;
      if (!eventsByIter[iter]) eventsByIter[iter] = [];
      eventsByIter[iter].push(ev);
    }
    traceSteps = Object.entries(eventsByIter).map(([iter, evts]) => ({
      iteration: parseInt(iter),
      tools: evts.filter(e => e.event_type === 'tool_end').map(e => ({
        tool: e.tool_name,
        input: e.tool_input,
        result: e.tool_result,
      })),
    }));
  }

  // Render trace steps
  if (traceSteps.length > 0) {
    for (const step of traceSteps) {
      const isWarning = step.loop_nudge || step.budget_warning;
      const cls = isWarning ? ' warning' : '';

      html += `<div class="trace-step${cls}">`;
      html += `<div class="step-header" onclick="this.nextElementSibling.classList.toggle('open')">`;
      html += `<span class="iter-badge">#${step.iteration || '?'}</span>`;

      const tools = step.tools || [];
      if (tools.length > 0) {
        const names = tools.map(t => t.tool).join(', ');
        html += `<span class="tool-name">${names}</span>`;
      } else {
        html += `<span class="tool-name">${step.action || 'thinking'}</span>`;
      }

      if (step.timestamp) {
        html += `<span class="time">${step.timestamp}s</span>`;
      }
      html += `</div>`;

      html += `<div class="step-body">`;
      if (step.reasoning) {
        html += `<div class="label">Reasoning</div>`;
        html += `<div class="reasoning">${escapeHtml(step.reasoning.substring(0, 500))}</div>`;
      }
      for (const t of tools) {
        html += `<div class="label">${escapeHtml(t.tool)}</div>`;
        html += `<pre>${escapeHtml(typeof t.input === 'string' ? t.input : JSON.stringify(t.input, null, 2))}</pre>`;
        if (t.result) {
          html += `<div class="label">Result</div>`;
          html += `<pre>${escapeHtml(String(t.result).substring(0, 1000))}</pre>`;
        }
      }
      if (step.loop_nudge) {
        html += `<div class="warning-text">LOOP DETECTED — agent was stuck</div>`;
      }
      if (step.budget_warning) {
        html += `<div class="warning-text">BUDGET WARNING — running low on iterations</div>`;
      }
      html += `</div></div>`;
    }
  } else {
    html += '<div class="empty">No trace data available</div>';
  }

  // Screenshots
  if (MODE === 'postmortem') {
    try {
      const resp = await fetch('/api/screenshots/' + encodeURIComponent(sid));
      const data = await resp.json();
      if (data.screenshots && data.screenshots.length > 0) {
        html += '<h2 style="margin-top:20px">Screenshots</h2>';
        for (const sc of data.screenshots) {
          html += `<img class="screenshot" src="/screenshots/${encodeURIComponent(sid)}/${sc}" alt="${sc}">`;
        }
      }
    } catch (e) { /* ignore */ }
  }

  // Failure diagnosis
  if (s.status === 'failed' && MODE === 'postmortem') {
    try {
      const resp = await fetch('/api/diagnosis/' + encodeURIComponent(sid));
      const data = await resp.json();
      if (data.diagnosis) {
        html += `<div class="diagnosis-panel">
          <h3>Failure Diagnosis</h3>
          <div class="content">${escapeHtml(data.diagnosis)}</div>
        </div>`;
      }
    } catch (e) { /* ignore */ }
  }

  detail.innerHTML = html;
}

function updateProgress() {
  const total = Object.keys(samples).length;
  const completed = Object.values(samples).filter(s => s.status === 'completed').length;
  const failed = Object.values(samples).filter(s => s.status === 'failed').length;

  document.getElementById('progressText').textContent = `${completed + failed}/${total}`;

  if (total > 0) {
    const pct = ((completed + failed) / total * 100).toFixed(1);
    document.getElementById('progressFill').style.width = pct + '%';
    if (failed > 0 && completed === 0) {
      document.getElementById('progressFill').className = 'progress-fill fail';
    } else {
      document.getElementById('progressFill').className = 'progress-fill success';
    }
  }

  // Update elapsed
  const elapsed = Math.floor((Date.now() - startTime) / 1000);
  const mins = Math.floor(elapsed / 60);
  const secs = elapsed % 60;
  document.getElementById('elapsed').textContent = `${mins}:${secs.toString().padStart(2, '0')}`;
}

setInterval(updateProgress, 1000);

function escapeHtml(text) {
  const div = document.createElement('div');
  div.textContent = text;
  return div.innerHTML;
}
</script>
</body>
</html>
"""


# ---------------------------------------------------------------------------
# Dashboard server (used in live mode from orchestrator)
# ---------------------------------------------------------------------------

class DashboardServer:
    """Manages the FastAPI dashboard server lifecycle for live mode.

    Runs uvicorn in a dedicated thread with its own event loop so that
    WebSocket connections stay alive even when the main event loop is blocked
    by synchronous API calls (e.g. anthropic.Anthropic).
    """

    def __init__(self, task_name: str, output_dir: Path, total_samples: int):
        self.task_name = task_name
        self.output_dir = output_dir
        self.total_samples = total_samples
        self._server = None
        self._thread: threading.Thread | None = None
        self._loop: asyncio.AbstractEventLoop | None = None  # server thread's loop
        self._clients: list[Any] = []  # WebSocket connections
        self._lock = threading.Lock()  # protects _clients

    def broadcast_event(self, event: WorkerEvent) -> None:
        """Broadcast a WorkerEvent to all connected WebSocket clients.

        Thread-safe: schedules sends on the server thread's event loop.
        """
        if not self._loop or not self._clients:
            return
        msg = event.to_json()
        with self._lock:
            for ws in list(self._clients):
                try:
                    self._loop.call_soon_threadsafe(
                        asyncio.ensure_future, ws.send_text(msg)
                    )
                except Exception:
                    pass

    async def start(self, port: int = DASHBOARD_PORT):
        """Start the dashboard server in a background thread."""
        started = threading.Event()
        server_self = self

        def _run_server():
            import uvicorn as _uvicorn
            from starlette.applications import Starlette
            from starlette.responses import HTMLResponse
            from starlette.routing import Route, WebSocketRoute
            from starlette.websockets import WebSocket as _WS

            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            server_self._loop = loop

            async def index(request):
                html = DASHBOARD_HTML.replace("'{{MODE}}'", "'live'")
                return HTMLResponse(html)

            async def websocket_endpoint(websocket: _WS):
                await websocket.accept()
                with server_self._lock:
                    server_self._clients.append(websocket)
                try:
                    while True:
                        await websocket.receive_text()
                except Exception:
                    pass
                finally:
                    with server_self._lock:
                        if websocket in server_self._clients:
                            server_self._clients.remove(websocket)

            app = Starlette(routes=[
                Route("/", index),
                WebSocketRoute("/ws", websocket_endpoint),
            ])

            config = _uvicorn.Config(app, host="0.0.0.0", port=port, log_level="warning", ws="wsproto")
            server_self._server = _uvicorn.Server(config)

            loop.run_until_complete(server_self._server.serve())

        self._thread = threading.Thread(target=_run_server, daemon=True)
        self._thread.start()

        # Wait for server to be ready
        for _ in range(40):
            time.sleep(0.25)
            if self._server and getattr(self._server, 'started', False):
                break

        logger.info(f"Dashboard started at http://localhost:{port}")

        try:
            webbrowser.open(f"http://localhost:{port}")
        except Exception:
            pass

    async def stop(self):
        """Stop the dashboard server."""
        if self._server:
            self._server.should_exit = True
        if self._thread:
            self._thread.join(timeout=5)


# ---------------------------------------------------------------------------
# Post-mortem server (standalone CLI)
# ---------------------------------------------------------------------------

def create_postmortem_app(output_dir: Path):
    """Create a FastAPI app for browsing completed run results."""
    from fastapi import FastAPI
    from fastapi.responses import HTMLResponse, JSONResponse, FileResponse

    app = FastAPI()

    # Load progress
    progress_path = output_dir / "progress.json"
    progress_data = {}
    if progress_path.exists():
        progress_data = json.loads(progress_path.read_text())

    # Load failure report
    failure_report_path = output_dir / "failure_report.md"
    failure_report = ""
    if failure_report_path.exists():
        failure_report = failure_report_path.read_text()

    # Parse per-sample diagnoses from failure report (simple split by ## Sample:)
    diagnosis_map: dict[str, str] = {}
    if failure_report:
        sections = failure_report.split("## Sample: ")
        for section in sections[1:]:
            lines = section.split("\n", 1)
            if lines:
                sid = lines[0].strip()
                content = lines[1] if len(lines) > 1 else ""
                diagnosis_map[sid] = content.strip()

    task_name = output_dir.name

    @app.get("/")
    async def index():
        html = DASHBOARD_HTML.replace("'{{MODE}}'", "'postmortem'")
        return HTMLResponse(html)

    @app.get("/api/samples")
    async def get_samples():
        samples = []
        samples_dir = output_dir / "samples"
        if samples_dir.exists():
            for sample_dir in sorted(samples_dir.iterdir()):
                if not sample_dir.is_dir():
                    continue
                sid = sample_dir.name
                entry = progress_data.get(sid, {})

                # Count trace iterations
                trace_path = sample_dir / "trace.json"
                iterations = 0
                if trace_path.exists():
                    try:
                        trace = json.loads(trace_path.read_text())
                        iterations = len(trace)
                    except (json.JSONDecodeError, OSError):
                        pass

                samples.append({
                    "sample_id": sid,
                    "status": entry.get("status", "unknown"),
                    "error": entry.get("error"),
                    "iterations": iterations,
                })

        return JSONResponse({
            "task_name": task_name,
            "samples": samples,
        })

    @app.get("/api/trace/{sample_id}")
    async def get_trace(sample_id: str):
        safe_id = sample_id.replace("/", "_").replace("\\", "_")
        trace_path = output_dir / "samples" / safe_id / "trace.json"
        if not trace_path.exists():
            return JSONResponse({"trace": []})
        try:
            trace = json.loads(trace_path.read_text())
            return JSONResponse({"trace": trace})
        except (json.JSONDecodeError, OSError):
            return JSONResponse({"trace": []})

    @app.get("/api/screenshots/{sample_id}")
    async def get_screenshots(sample_id: str):
        safe_id = sample_id.replace("/", "_").replace("\\", "_")
        sample_dir = output_dir / "samples" / safe_id
        screenshots = []
        if sample_dir.exists():
            screenshots = [p.name for p in sorted(sample_dir.glob("*.png"))]
        return JSONResponse({"screenshots": screenshots})

    @app.get("/screenshots/{sample_id}/{filename}")
    async def serve_screenshot(sample_id: str, filename: str):
        safe_id = sample_id.replace("/", "_").replace("\\", "_")
        filepath = output_dir / "samples" / safe_id / filename
        if filepath.exists() and filepath.suffix == ".png":
            return FileResponse(filepath, media_type="image/png")
        return JSONResponse({"error": "not found"}, status_code=404)

    @app.get("/api/diagnosis/{sample_id}")
    async def get_diagnosis(sample_id: str):
        diag = diagnosis_map.get(sample_id, "")
        return JSONResponse({"diagnosis": diag})

    return app


def main():
    parser = argparse.ArgumentParser(
        description="Post-mortem dashboard for browser automation runs"
    )
    parser.add_argument(
        "output_dir",
        help="Path to task output directory (contains progress.json and samples/)",
    )
    parser.add_argument(
        "--port", "-p",
        type=int,
        default=DASHBOARD_PORT,
        help=f"Port to serve on (default: {DASHBOARD_PORT})",
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    if not output_dir.exists():
        print(f"Directory not found: {output_dir}", file=sys.stderr)
        sys.exit(1)
    if not (output_dir / "progress.json").exists() and not (output_dir / "samples").exists():
        print(f"Not a valid task output dir: {output_dir}", file=sys.stderr)
        sys.exit(1)

    logging.basicConfig(level=logging.INFO)

    import uvicorn
    app = create_postmortem_app(output_dir)

    print(f"Dashboard: http://localhost:{args.port}")
    try:
        webbrowser.open(f"http://localhost:{args.port}")
    except Exception:
        pass

    uvicorn.run(app, host="0.0.0.0", port=args.port, log_level="info")


if __name__ == "__main__":
    main()
