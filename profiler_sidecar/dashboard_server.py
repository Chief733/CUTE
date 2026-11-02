#!/usr/bin/env python3
"""Serve or export a dependency-free dashboard for one sidecar run."""

import argparse
import json
import os
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from dashboard_data import build_dashboard_data

HERE = Path(__file__).resolve().parent


def export_html(run, output):
    data = json.dumps(build_dashboard_data(run), ensure_ascii=False, separators=(",", ":"))
    html = (HERE / "dashboard.html").read_text()
    marker = "window.__ARGUS_SNAPSHOT__ = null;"
    if marker not in html:
        raise RuntimeError("dashboard snapshot marker is missing")
    html = html.replace(marker, "window.__ARGUS_SNAPSHOT__ = " + data + ";", 1)
    output = Path(output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(output.name + f".tmp.{os.getpid()}")
    temporary.write_text(html)
    os.replace(temporary, output)
    return output


def handler_for(run):
    run = Path(run).resolve()
    page = (HERE / "dashboard.html").read_bytes()

    class Handler(BaseHTTPRequestHandler):
        def _send(self, code, body, content_type):
            self.send_response(code)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            if self.path in ("/", "/dashboard.html"):
                self._send(200, page, "text/html; charset=utf-8")
                return
            if self.path == "/api/data":
                try:
                    body = json.dumps(build_dashboard_data(run), ensure_ascii=False,
                                      separators=(",", ":")).encode()
                    self._send(200, body, "application/json; charset=utf-8")
                except Exception as error:
                    body = json.dumps({"error": repr(error)}, ensure_ascii=False).encode()
                    self._send(500, body, "application/json; charset=utf-8")
                return
            if self.path == "/api/health":
                self._send(200, b'{"ok":true}', "application/json")
                return
            self._send(404, b"not found\n", "text/plain; charset=utf-8")

        def log_message(self, pattern, *args):
            print("dashboard:", pattern % args)

    return Handler


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    parser.add_argument("--bind", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--export-html", type=Path)
    args = parser.parse_args()
    if not args.run.is_dir():
        parser.error("run directory does not exist: " + str(args.run))
    if args.export_html:
        output = export_html(args.run, args.export_html)
        print(json.dumps({"output": str(output), "run": str(args.run.resolve())}, ensure_ascii=False))
        return 0
    server = ThreadingHTTPServer((args.bind, args.port), handler_for(args.run))
    print(f"ARGUS dashboard: http://{args.bind}:{args.port}/")
    print(f"Run: {args.run.resolve()}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
