"""Local demo: watch source files, rerun Python, and serve the latest result.

Each run uses a new process so edits to imported Python modules take effect.
The browser polls status; no Node, build step, notebook, or CDN is needed.
"""

import argparse
import hashlib
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import mimetypes
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import threading
import time
from urllib.parse import unquote, urlsplit
import webbrowser


def file_revision(paths: list[Path]) -> str:
    digest = hashlib.sha256()
    for path in sorted(set(paths)):
        try:
            digest.update(str(path).encode())
            digest.update(path.read_bytes())
        except FileNotFoundError:
            continue # an editor may replace a file between enumeration and read
    return digest.hexdigest()[:16]


class ExperimentRunner:
    def __init__(self, root: Path, example: Path, config: Path, duration: float | None = None, pattern: str | None = None, mode: str = "custom"):
        self.root, self.example, self.config = root.resolve(), example.resolve(), config.resolve()
        self.duration = duration
        self.pattern = pattern
        self.mode = mode
        self.generation = 0
        self.condition = threading.Condition()
        self.stopped = threading.Event()
        self.revision = ""
        self.result_revision = ""
        self.result: bytes | None = None
        self.error: str | None = None
        self.state = "running"
        self.pending: tuple[str, int] | None = None
        self.process: subprocess.Popen | None = None
        self.threads: list[threading.Thread] = []

    def source_files(self) -> list[Path]:
        files = [self.example, self.config, self.root / "pyproject.toml"]
        for folder, extension in (("src", "*.py"), ("examples", "*.py"), ("config", "*.toml")):
            files.extend((self.root / folder).rglob(extension))
        return files

    def request_run(self, revision: str | None = None) -> None:
        with self.condition:
            self.revision = revision or file_revision(self.source_files())
            self.generation += 1
            self.pending = (self.revision, self.generation)
            self.state, self.error = "running", None
            self.condition.notify()

    def start(self) -> None:
        self.request_run()
        for target in (self._worker, self._watch):
            thread = threading.Thread(target=target, daemon=True)
            self.threads.append(thread)
            thread.start()

    def select_experiment(self, mode: str, pattern: str | None) -> None:
        from .presets import PATTERNS
        if self.mode == "custom":
            raise ValueError("This server was started with a custom example or config; restart to change it")
        if mode not in ("lab", "generic") or (pattern is not None and (mode != "generic" or pattern not in PATTERNS)):
            raise ValueError("Choose lab or generic mode and a known generic pattern")
        with self.condition:
            self.mode, self.pattern = mode, pattern
            self.example = self.root / f"examples/{mode}_demo.py"
            self.config = self.root / f"config/{mode}.toml"
            self.request_run()

    def close(self) -> None:
        self.stopped.set()
        with self.condition:
            if self.process is not None and self.process.poll() is None:
                self.process.terminate()
            self.condition.notify_all()
        for thread in self.threads:
            thread.join(timeout=3)

    def _watch(self) -> None:
        candidate, changed_at = self.revision, time.monotonic()
        while not self.stopped.wait(0.25):
            revision = file_revision(self.source_files())
            if revision != candidate:
                candidate, changed_at = revision, time.monotonic()
            # Debounce bursts of writes from an editor.
            if candidate != self.revision and time.monotonic() - changed_at >= 0.35:
                self.request_run(candidate)

    def _worker(self) -> None:
        while not self.stopped.is_set():
            with self.condition:
                self.condition.wait_for(lambda: self.pending is not None or self.stopped.is_set())
                if self.stopped.is_set():
                    return
                (revision, generation), self.pending = self.pending, None
                example, config, pattern = self.example, self.config, self.pattern
            try:
                with tempfile.TemporaryDirectory(prefix="lab-sim-") as scratch:
                    output = Path(scratch) / "result.json"
                    command = [sys.executable, "-B", str(example), "--config", str(config), "--output", str(output)]
                    if self.duration is not None:
                        command += ["--duration", str(self.duration)]
                    if pattern is not None:
                        command += ["--pattern", pattern]
                    env = os.environ.copy()
                    env["PYTHONPATH"] = str(self.root / "src") + os.pathsep + env.get("PYTHONPATH", "")
                    # A fresh cache location avoids stale same-second .pyc reads.
                    env["PYTHONPYCACHEPREFIX"] = str(Path(scratch) / "pycache")
                    with self.condition:
                        if self.stopped.is_set():
                            return
                        self.process = subprocess.Popen(command, cwd=self.root, env=env, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
                    try:
                        stdout, stderr = self.process.communicate(timeout=120)
                    except subprocess.TimeoutExpired:
                        self.process.kill()
                        self.process.communicate()
                        raise ValueError("Experiment exceeded 120 seconds; shorten the duration or simplify the policy")
                    if self.process.returncode:
                        raise ValueError((stderr or stdout or "Experiment failed")[-8000:])
                    result = json.loads(output.read_text())
                    # Catch invalid custom output before handing it to the browser.
                    if result.get("schema_version") != 1 or not result.get("history", {}).get("time"):
                        raise ValueError("Example must write a simulation result; use example_main()")
                    payload = json.dumps(result, allow_nan=False, separators=(",", ":")).encode()
                with self.condition:
                    if revision == self.revision and generation == self.generation and self.pending is None:
                        self.result, self.result_revision = payload, f"{revision}:{generation}"
                        self.state, self.error = "ready", None
            except Exception as exc:
                with self.condition:
                    if revision == self.revision and generation == self.generation and self.pending is None:
                        self.state, self.error = "error", str(exc)
            finally:
                with self.condition:
                    self.process = None

    def status(self) -> dict:
        with self.condition:
            return {"state": self.state, "revision": self.revision, "result_revision": self.result_revision, "error": self.error, "mode": self.mode, "pattern": self.pattern, "example": str(self.example.relative_to(self.root)) if self.example.is_relative_to(self.root) else self.example.name, "web_revision": file_revision(list((self.root / "web").rglob("*.*")))}


def make_handler(root: Path, runner: ExperimentRunner):
    web_root = (root / "web").resolve()

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, format, *args):
            pass

        def respond(self, payload: bytes, content_type: str, status=HTTPStatus.OK) -> None:
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(payload)))
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Content-Type-Options", "nosniff")
            self.end_headers()
            self.wfile.write(payload)

        def do_GET(self) -> None:
            path = unquote(urlsplit(self.path).path)
            if path == "/api/status":
                self.respond(json.dumps(runner.status()).encode(), "application/json")
            elif path == "/api/result":
                with runner.condition:
                    payload = runner.result
                if payload is None:
                    self.respond(b'{"error":"Simulation is still running"}', "application/json", HTTPStatus.SERVICE_UNAVAILABLE)
                else:
                    self.respond(payload, "application/json")
            else:
                asset = (web_root / ("index.html" if path == "/" else path.lstrip("/"))).resolve()
                if not asset.is_relative_to(web_root) or not asset.is_file():
                    self.respond(b"Not found", "text/plain", HTTPStatus.NOT_FOUND)
                    return
                self.respond(asset.read_bytes(), mimetypes.guess_type(asset.name)[0] or "application/octet-stream")

        def do_POST(self) -> None:
            path = urlsplit(self.path).path
            if path == "/api/experiment":
                try:
                    length = int(self.headers.get("Content-Length", "0"))
                    if length <= 0 or length > 1024:
                        raise ValueError("Expected a small JSON experiment selection")
                    selection = json.loads(self.rfile.read(length))
                    runner.select_experiment(selection["mode"], selection.get("pattern"))
                except (ValueError, KeyError, TypeError) as exc:
                    self.respond(json.dumps({"error": str(exc)}).encode(), "application/json", HTTPStatus.BAD_REQUEST)
                    return
                self.respond(b'{"state":"running"}', "application/json", HTTPStatus.ACCEPTED)
                return
            if path != "/api/rerun":
                self.respond(b"Not found", "text/plain", HTTPStatus.NOT_FOUND)
                return
            runner.request_run()
            self.respond(b'{"state":"running"}', "application/json", HTTPStatus.ACCEPTED)

    return Handler


def main() -> None:
    parser = argparse.ArgumentParser(description="Live local preview of a Python lab experiment.")
    parser.add_argument("--root", type=Path, default=Path.cwd(), help="Starter checkout (default: current directory)")
    parser.add_argument("--mode", choices=["lab", "generic"], default="lab")
    parser.add_argument("--example", type=Path, help="Custom experiment script")
    parser.add_argument("--config", type=Path, help="Override the mode's TOML config")
    from .presets import PATTERNS
    parser.add_argument("--pattern", choices=list(PATTERNS), help="A fancy Lissajous preset for generic mode")
    parser.add_argument("--duration", type=float)
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--open", action="store_true", help="Open the demo in your browser")
    args = parser.parse_args()
    root = args.root.resolve()
    if args.pattern and (args.mode != "generic" or args.example):
        parser.error("--pattern requires --mode generic and the built-in generic example")
    example = root / (args.example or Path(f"examples/{args.mode}_demo.py"))
    config = root / (args.config or Path(f"config/{args.mode}.toml"))
    if not example.is_file() or not config.is_file() or not (root / "web/index.html").is_file():
        parser.error("Run from the starter checkout or supply --root; example, config and web/index.html must exist")
    runner = ExperimentRunner(root, example, config, args.duration, args.pattern, "custom" if args.example or args.config else args.mode)
    server = ThreadingHTTPServer(("127.0.0.1", args.port), make_handler(root, runner))
    runner.start()
    url = f"http://127.0.0.1:{server.server_port}"
    print(f"Lab demo: {url}\nWatching src/, examples/, config/, and web/. Press Ctrl+C to stop.", flush=True)
    if args.open:
        webbrowser.open(url)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        runner.close()
        server.server_close()


if __name__ == "__main__":
    main()
