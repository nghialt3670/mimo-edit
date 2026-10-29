"""Serve a local page for interactive single-sample mixed-prompt inference."""

from __future__ import annotations

import argparse
import base64
import io
import json
import math
import re
import threading
import time
import traceback
from datetime import datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

from PIL import Image

PROMPT_TYPES = ("point", "box", "scribble")
PLACEHOLDER_RE = re.compile(r"<((?:point|box|scribble)\d*)>")
TOKEN_RE = re.compile(r"(?:point|box|scribble)\d*")


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", type=Path, help="Initial source image; otherwise upload one in the page.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--output-root", type=Path, default=Path(".out/ui"))
    parser.add_argument("--max-steps", type=int, default=16)
    args = parser.parse_args(argv)
    if args.max_steps < 1:
        parser.error("--max-steps must be at least 1.")
    return args


def _png_base64(image: Image.Image) -> str:
    buffer = io.BytesIO()
    image.save(buffer, "PNG")
    return base64.b64encode(buffer.getvalue()).decode("ascii")


def _coordinate(value: Any, limit: int) -> int:
    number = float(value)
    if not math.isfinite(number):
        raise ValueError("Coordinates must be finite numbers.")
    return min(max(round(number), 0), limit - 1)


def _pair(value: Any) -> tuple[Any, Any]:
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        raise ValueError("Each point needs an x and a y coordinate.")
    return value[0], value[1]


def _prompt_value(prompt_type: str, value: Any, size: tuple[int, int]) -> Any:
    width, height = size
    if prompt_type == "point":
        x, y = _pair(value)
        return [_coordinate(x, width), _coordinate(y, height)]
    if prompt_type == "box":
        if not isinstance(value, (list, tuple)) or len(value) != 4:
            raise ValueError("A box needs four coordinates.")
        left, right = sorted((_coordinate(value[0], width), _coordinate(value[2], width)))
        top, bottom = sorted((_coordinate(value[1], height), _coordinate(value[3], height)))
        if left == right or top == bottom:
            raise ValueError("A box has no area.")
        return [left, top, right, bottom]
    if not isinstance(value, list) or not value:
        raise ValueError("A scribble needs at least one point.")
    return [
        [_coordinate(x, width), _coordinate(y, height)]
        for x, y in (_pair(point) for point in value)
    ]


def validate_request(payload: dict[str, Any], size: tuple[int, int]) -> tuple[str, list[dict[str, Any]]]:
    instruction = str(payload.get("instruction") or "").strip()
    if not instruction:
        raise ValueError("Enter an instruction.")
    prompts: list[dict[str, Any]] = []
    for item in payload.get("prompts") or []:
        if not isinstance(item, dict) or item.get("type") not in PROMPT_TYPES:
            raise ValueError(f"Prompt type must be one of {', '.join(PROMPT_TYPES)}.")
        token = str(item.get("token") or "")
        if not TOKEN_RE.fullmatch(token) or not token.startswith(item["type"]):
            raise ValueError(f"Invalid placeholder name '{token}'.")
        if any(prompt["token"] == token for prompt in prompts):
            raise ValueError(f"Duplicate placeholder <{token}>.")
        prompts.append(
            {
                "type": item["type"],
                "token": token,
                "value": _prompt_value(item["type"], item.get("value"), size),
            }
        )
    tokens = {prompt["token"] for prompt in prompts}
    for match in PLACEHOLDER_RE.finditer(instruction):
        if match.group(1) not in tokens:
            raise ValueError(f"<{match.group(1)}> has no matching prompt.")
    return instruction, prompts


MEMORY_SAMPLE_SECONDS = 2.0


def _memory_snapshot() -> str:
    import resource

    peak_mb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1024 * 1024)  # bytes on macOS
    parts = [f"peak_rss={peak_mb:.0f}MB"]
    try:
        import torch

        if torch.backends.mps.is_available():
            parts.append(f"mps_alloc={torch.mps.current_allocated_memory() / 1e6:.0f}MB")
            parts.append(f"mps_driver={torch.mps.driver_allocated_memory() / 1e6:.0f}MB")
    except Exception as exc:
        parts.append(f"mps_unavailable={type(exc).__name__}")
    return " ".join(parts)


def _start_memory_monitor(label: str) -> threading.Event:
    stop = threading.Event()

    def sample() -> None:
        while not stop.wait(MEMORY_SAMPLE_SECONDS):
            print(f"[memory {label}] {_memory_snapshot()}", flush=True)

    threading.Thread(target=sample, daemon=True).start()
    return stop


class Runtime:
    """Loads the shared model and tools once; each run gets a fresh Agent."""

    def __init__(self, output_root: Path, max_steps: int) -> None:
        from evaluation.common import AGENT_EXECUTION_TIMEOUT_SECONDS
        from evaluation.notebook import create_agent_model
        from framework.tools import image_editing_tools
        from framework.utils.project import load_project_environment

        load_project_environment()
        self.output_root = output_root
        self.max_steps = max_steps
        self.timeout = AGENT_EXECUTION_TIMEOUT_SECONDS
        print("Loading agent model and inference tools...")
        self.model = create_agent_model()
        self.tools = image_editing_tools()
        print("Agent ready.")

    def run(
        self,
        image: Image.Image,
        instruction: str,
        prompts: list[dict[str, Any]],
    ) -> dict[str, Any]:
        from evaluation.common import (
            annotate_sample,
            build_agent_context,
            build_agent_task,
        )
        from framework.core import Agent, local_executor

        references = []
        for prompt in prompts:
            variable = prompt["token"]
            references.append(
                {
                    "token": None,
                    "display_token": f"<{variable}>",
                    "span": None,
                    "spatial_destination": False,
                    "selected_prompt": {
                        "type": prompt["type"],
                        "value": prompt["value"],
                        "variable": variable,
                    },
                }
            )
        sample = {
            "sample_id": "ui",
            "instruction": instruction,
            "source_image": image,
            "references": references,
        }
        task, objects = build_agent_task(sample)
        agent_context = build_agent_context(sample, "mixed")

        run_dir = self.output_root / datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        run_dir.mkdir(parents=True, exist_ok=True)
        annotated = annotate_sample(sample)
        image.save(run_dir / "source.png")
        annotated.save(run_dir / "annotated.png")

        agent = None
        output = None
        message = ""
        error = None
        failure_trace = None
        started = time.perf_counter()
        print(f"[memory {run_dir.name} start] {_memory_snapshot()}", flush=True)
        monitor = _start_memory_monitor(run_dir.name)
        try:
            agent = Agent(
                model=self.model,
                tools=self.tools,
                max_steps=self.max_steps,
                instructions=agent_context,
                executor=local_executor(timeout_seconds=self.timeout),
                verbosity_level=0,
            )
            result = agent.send(task, objects=objects)
            message = result.message.text
            images = [obj for obj in result.message.objects if isinstance(obj, Image.Image)]
            if images:
                output = images[0].convert("RGB")
            else:
                error = "The agent finished without returning an image."
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"
            failure_trace = traceback.format_exc()
        monitor.set()
        print(f"[memory {run_dir.name} end] {_memory_snapshot()}", flush=True)
        elapsed = round(time.perf_counter() - started, 2)

        trajectory = agent.trajectory_text if agent is not None else ""
        (run_dir / "trajectory.txt").write_text(trajectory, encoding="utf-8")
        if output is not None:
            output.save(run_dir / "output.png")
        summary = {
            "instruction": instruction,
            "prompts": prompts,
            "message": message,
            "error": error,
            "traceback": failure_trace,
            "elapsed_seconds": elapsed,
        }
        (run_dir / "run.json").write_text(
            json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        return {
            "ok": output is not None,
            "error": error,
            "message": message,
            "elapsed_seconds": elapsed,
            "run_dir": str(run_dir),
            "annotated_png": _png_base64(annotated),
            "output_png": _png_base64(output) if output is not None else None,
            "trajectory": trajectory,
        }


class App:
    def __init__(self, runtime: Runtime, image: Image.Image | None) -> None:
        self.runtime = runtime
        self.image = image
        self.lock = threading.Lock()


class Handler(BaseHTTPRequestHandler):
    server: Any

    def do_GET(self) -> None:
        app = self.server.app
        if self.path == "/":
            self._send(200, "text/html; charset=utf-8", HTML_PAGE.encode("utf-8"))
        elif self.path.split("?")[0] == "/image" and app.image is not None:
            buffer = io.BytesIO()
            app.image.save(buffer, "PNG")
            self._send(200, "image/png", buffer.getvalue())
        else:
            self._send(404, "text/plain; charset=utf-8", b"Not found")

    def do_POST(self) -> None:
        length = int(self.headers.get("Content-Length") or 0)
        body = self.rfile.read(length)
        if self.path == "/upload":
            self._upload(body)
        elif self.path == "/run":
            self._run(body)
        else:
            self._send(404, "text/plain; charset=utf-8", b"Not found")

    def _upload(self, body: bytes) -> None:
        try:
            image = Image.open(io.BytesIO(body)).convert("RGB")
        except Exception as exc:
            self._json(400, {"ok": False, "error": f"Not an image: {exc}"})
            return
        with self.server.app.lock:
            self.server.app.image = image
        self._json(200, {"ok": True, "width": image.width, "height": image.height})

    def _run(self, body: bytes) -> None:
        app = self.server.app
        if app.image is None:
            self._json(400, {"ok": False, "error": "Upload an image first."})
            return
        try:
            payload = json.loads(body or b"{}")
            instruction, prompts = validate_request(payload, app.image.size)
        except (ValueError, TypeError, json.JSONDecodeError) as exc:
            self._json(400, {"ok": False, "error": str(exc)})
            return
        with app.lock:
            result = app.runtime.run(app.image, instruction, prompts)
        self._json(200 if result["ok"] else 500, result)

    def _json(self, status: int, data: dict[str, Any]) -> None:
        self._send(status, "application/json; charset=utf-8", json.dumps(data).encode("utf-8"))

    def _send(self, status: int, content_type: str, data: bytes) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)


def _make_server(host: str, port: int, app: App) -> ThreadingHTTPServer:
    for candidate in range(port, port + 20):
        try:
            server = ThreadingHTTPServer((host, candidate), Handler)
        except OSError:
            continue
        server.app = app
        return server
    raise OSError(f"No free port found from {port} to {port + 19}.")


def main(argv: list[str] | None = None) -> None:
    args = _parse_args(argv)
    runtime = Runtime(args.output_root.resolve(), args.max_steps)
    image = None
    if args.image is not None:
        image = Image.open(args.image).convert("RGB")
    server = _make_server(args.host, args.port, App(runtime, image))
    host, port = server.server_address[:2]
    print(f"Inference UI: http://{host}:{port}  (Ctrl+C to stop)")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


HTML_PAGE = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>MixedPrompt inference UI</title>
<style>
body { font-family: system-ui, sans-serif; margin: 16px; color: #222; }
.layout { display: flex; gap: 16px; flex-wrap: wrap; align-items: flex-start; }
.left { flex: 1 1 520px; min-width: 320px; }
.right { flex: 1 1 420px; min-width: 320px; }
canvas { max-width: 100%; border: 1px solid #ccc; cursor: crosshair; background: #eee; }
button { padding: 6px 12px; margin: 2px; }
button.active { background: #333; color: #fff; }
textarea { width: 100%; height: 72px; font-size: 15px; box-sizing: border-box; }
#prompts li { margin: 4px 0; }
.err { color: #b00020; white-space: pre-wrap; }
.warn { color: #8a6d00; min-height: 1em; }
.muted { color: #666; font-size: 13px; }
pre { background: #f6f6f6; padding: 8px; max-height: 360px; overflow: auto; white-space: pre-wrap; font-size: 12px; }
.row { display: flex; gap: 8px; flex-wrap: wrap; }
.row figure { margin: 0; flex: 1 1 260px; }
.row img { max-width: 100%; border: 1px solid #ccc; }
</style>
</head>
<body>
<h2>MixedPrompt single-sample inference</h2>
<div class="layout">
  <div class="left">
    <div>
      <input type="file" id="file" accept="image/*">
      <button data-tool="point" class="active">Point</button>
      <button data-tool="box">Box</button>
      <button data-tool="scribble">Scribble</button>
    </div>
    <canvas id="canvas"></canvas>
    <p class="muted">Click to place a point, drag for a box, or drag to draw a scribble.</p>
  </div>
  <div class="right">
    <label for="instruction">Instruction (click a prompt button to insert its placeholder)</label>
    <textarea id="instruction" placeholder="e.g. Remove <point_0>"></textarea>
    <ul id="prompts"></ul>
    <div id="checks" class="warn"></div>
    <button id="run">Run</button>
    <button id="reset">Reset</button>
    <span id="status" class="muted"></span>
    <div id="error" class="err"></div>
    <div id="result"></div>
  </div>
</div>
<script>
const canvas = document.getElementById("canvas");
const ctx = canvas.getContext("2d");
const ta = document.getElementById("instruction");
const list = document.getElementById("prompts");
const checks = document.getElementById("checks");
const errorBox = document.getElementById("error");
const resultBox = document.getElementById("result");
const statusBox = document.getElementById("status");
const runBtn = document.getElementById("run");
const img = new Image();
const COLORS = { point: "#ffdc00", box: "#ff4040", scribble: "#00dcff" };
let imageLoaded = false;
let prompts = [];
let tool = "point";
let draft = null;
let timer = null;

canvas.style.touchAction = "none";

img.onload = () => {
  canvas.width = img.naturalWidth;
  canvas.height = img.naturalHeight;
  canvas.style.width = Math.min(img.naturalWidth, 900) + "px";
  imageLoaded = true;
  statusBox.textContent = "";
  redraw();
};
img.onerror = () => {
  imageLoaded = false;
  statusBox.textContent = "Upload an image to start.";
  redraw();
};

function loadImage() {
  img.src = "/image?t=" + Date.now();
}

function toImage(e) {
  const r = canvas.getBoundingClientRect();
  const x = (e.clientX - r.left) * canvas.width / r.width;
  const y = (e.clientY - r.top) * canvas.height / r.height;
  return { x: Math.min(Math.max(x, 0), canvas.width - 1), y: Math.min(Math.max(y, 0), canvas.height - 1) };
}

canvas.addEventListener("pointerdown", (e) => {
  if (!imageLoaded) return;
  const p = toImage(e);
  if (tool === "point") {
    addPrompt("point", [p.x, p.y]);
    return;
  }
  canvas.setPointerCapture(e.pointerId);
  draft = { type: tool, start: p, end: p, points: [[p.x, p.y]] };
  redraw();
});

canvas.addEventListener("pointermove", (e) => {
  if (!draft) return;
  const p = toImage(e);
  if (draft.type === "box") {
    draft.end = p;
  } else {
    const last = draft.points[draft.points.length - 1];
    if (Math.hypot(p.x - last[0], p.y - last[1]) >= 3) draft.points.push([p.x, p.y]);
  }
  redraw();
});

canvas.addEventListener("pointerup", (e) => {
  if (!draft) return;
  const p = toImage(e);
  if (draft.type === "box") {
    const s = draft.start;
    if (Math.abs(p.x - s.x) >= 3 && Math.abs(p.y - s.y) >= 3) addPrompt("box", [s.x, s.y, p.x, p.y]);
  } else {
    draft.points.push([p.x, p.y]);
    if (draft.points.length >= 2) addPrompt("scribble", draft.points);
  }
  draft = null;
  redraw();
});

function draftAsPrompt() {
  if (draft.type === "box") {
    const e = draft.end || draft.start;
    return { type: "box", value: [draft.start.x, draft.start.y, e.x, e.y] };
  }
  return { type: "scribble", value: draft.points };
}

function redraw() {
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  if (imageLoaded) ctx.drawImage(img, 0, 0);
  const lw = Math.max(2, canvas.width / 256);
  prompts.forEach((p) => drawPrompt(p, lw, false));
  if (draft) drawPrompt(draftAsPrompt(), lw, true);
}

function labelAt(text, x, y, color) {
  if (!text) return;
  ctx.font = Math.max(12, canvas.width / 60) + "px sans-serif";
  ctx.fillStyle = color;
  ctx.fillText(text, x, y);
}

function drawPrompt(p, lw, dashed) {
  ctx.save();
  const color = COLORS[p.type];
  const label = p.token ? `<${p.token}>` : "";
  ctx.strokeStyle = color;
  ctx.fillStyle = color;
  ctx.lineWidth = lw;
  if (dashed) ctx.setLineDash([6, 4]);
  if (p.type === "point") {
    const [x, y] = p.value;
    ctx.beginPath();
    ctx.arc(x, y, lw * 2, 0, Math.PI * 2);
    ctx.fill();
    ctx.strokeStyle = "#000";
    ctx.stroke();
    labelAt(label, x + lw * 3, y - lw * 3, color);
  } else if (p.type === "box") {
    const [x1, y1, x2, y2] = p.value;
    const left = Math.min(x1, x2), top = Math.min(y1, y2);
    ctx.strokeRect(left, top, Math.abs(x2 - x1), Math.abs(y2 - y1));
    labelAt(label, left, top - lw * 2, color);
  } else {
    const pts = p.value;
    ctx.beginPath();
    pts.forEach(([x, y], k) => (k ? ctx.lineTo(x, y) : ctx.moveTo(x, y)));
    ctx.lineWidth = lw * 2;
    ctx.stroke();
    labelAt(label, pts[0][0], pts[0][1] - lw * 3, color);
  }
  ctx.restore();
}

function summary(p) {
  const r = (v) => Math.round(v);
  if (p.type === "point") return `(${r(p.value[0])}, ${r(p.value[1])})`;
  if (p.type === "box") return `[${p.value.map(r).join(", ")}]`;
  return `${p.value.length} points`;
}

function refresh() {
  renderPrompts();
  redraw();
}

function renderPrompts() {
  list.innerHTML = "";
  prompts.forEach((p, i) => {
    const token = `<${p.token}>`;
    const li = document.createElement("li");
    li.style.borderLeft = `6px solid ${COLORS[p.type]}`;
    li.style.paddingLeft = "6px";
    const insert = document.createElement("button");
    insert.textContent = token;
    insert.onclick = () => insertToken(token);
    const desc = document.createElement("span");
    desc.className = "muted";
    desc.textContent = " " + summary(p);
    const del = document.createElement("button");
    del.textContent = "x";
    del.onclick = () => removePrompt(i);
    li.append(insert, desc, del);
    list.append(li);
  });
  checkPlaceholders();
}

const TOKEN_RE = /<((?:point|box|scribble)\d*)>/g;

function nextToken(type) {
  const taken = new Set(prompts.map((p) => p.token));
  for (const m of ta.value.matchAll(TOKEN_RE)) taken.add(m[1]);
  let token = type, n = 2;
  while (taken.has(token)) token = type + n++;
  return token;
}

function addPrompt(type, value) {
  const token = nextToken(type);
  const text = ta.value;
  const sep = text === "" || /\s$/.test(text) ? "" : " ";
  ta.value = text + sep + `<${token}>`;
  prompts.push({ type, token, value });
  refresh();
}

function removePrompt(i) {
  const { token } = prompts[i];
  prompts.splice(i, 1);
  ta.value = ta.value.replace(new RegExp(`\\s*<${token}>`, "g"), "").trim();
  refresh();
}

function insertToken(token) {
  const s = ta.selectionStart, e = ta.selectionEnd;
  ta.value = ta.value.slice(0, s) + token + ta.value.slice(e);
  ta.focus();
  ta.selectionStart = ta.selectionEnd = s + token.length;
  checkPlaceholders();
}

function checkPlaceholders() {
  const names = new Set(prompts.map((p) => p.token));
  const used = new Set([...ta.value.matchAll(TOKEN_RE)].map((m) => m[1]));
  const messages = [];
  used.forEach((n) => {
    if (!names.has(n)) messages.push(`Unknown placeholder <${n}>.`);
  });
  names.forEach((n) => {
    if (!used.has(n)) messages.push(`<${n}> is not used in the instruction.`);
  });
  checks.textContent = messages.join(" ");
}

function addImage(row, caption, b64) {
  const fig = document.createElement("figure");
  const cap = document.createElement("figcaption");
  cap.className = "muted";
  cap.textContent = caption;
  const im = document.createElement("img");
  im.src = "data:image/png;base64," + b64;
  fig.append(cap, im);
  row.append(fig);
}

function showResult(d) {
  if (d.error) errorBox.textContent = d.error;
  const row = document.createElement("div");
  row.className = "row";
  if (d.annotated_png) addImage(row, "Annotated source", d.annotated_png);
  if (d.output_png) addImage(row, "Edited output", d.output_png);
  resultBox.append(row);
  if (d.message) {
    const p = document.createElement("p");
    p.textContent = "Agent message: " + d.message;
    resultBox.append(p);
  }
  if (d.elapsed_seconds !== undefined) {
    const p = document.createElement("p");
    p.className = "muted";
    p.textContent = `Elapsed: ${d.elapsed_seconds} s. Saved to: ${d.run_dir || "(not saved)"}`;
    resultBox.append(p);
  }
  if (d.trajectory) {
    const det = document.createElement("details");
    const sum = document.createElement("summary");
    sum.textContent = "Trajectory";
    const pre = document.createElement("pre");
    pre.textContent = d.trajectory;
    det.append(sum, pre);
    resultBox.append(det);
  }
}

async function run() {
  errorBox.textContent = "";
  resultBox.innerHTML = "";
  const instruction = ta.value.trim();
  if (!instruction) {
    errorBox.textContent = "Enter an instruction.";
    return;
  }
  runBtn.disabled = true;
  const started = Date.now();
  timer = setInterval(() => {
    statusBox.textContent = `Running ${Math.round((Date.now() - started) / 1000)} s...`;
  }, 500);
  let data;
  try {
    const r = await fetch("/run", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ instruction, prompts }),
    });
    data = await r.json();
  } catch (err) {
    data = { ok: false, error: String(err) };
  }
  clearInterval(timer);
  runBtn.disabled = false;
  statusBox.textContent = "";
  showResult(data);
}

function reset() {
  prompts = [];
  draft = null;
  ta.value = "";
  errorBox.textContent = "";
  resultBox.innerHTML = "";
  refresh();
}

document.querySelectorAll("[data-tool]").forEach((b) => {
  b.onclick = () => {
    tool = b.dataset.tool;
    draft = null;
    document.querySelectorAll("[data-tool]").forEach((x) => x.classList.toggle("active", x === b));
    redraw();
  };
});

document.getElementById("file").onchange = async (e) => {
  const f = e.target.files[0];
  if (!f) return;
  const r = await fetch("/upload", {
    method: "POST",
    headers: { "Content-Type": f.type || "application/octet-stream" },
    body: f,
  });
  const d = await r.json();
  if (!d.ok) {
    errorBox.textContent = d.error;
    return;
  }
  reset();
  loadImage();
};

ta.addEventListener("input", checkPlaceholders);
runBtn.onclick = run;
document.getElementById("reset").onclick = reset;
renderPrompts();
loadImage();
</script>
</body>
</html>
"""


if __name__ == "__main__":
    main()
