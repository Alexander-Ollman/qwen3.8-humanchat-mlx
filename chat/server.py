#!/usr/bin/env python3
"""Serve the chat page and forward chat requests to llama-server.

The page and the API share one origin, so the browser needs no CORS access.
The server binds to loopback only. Standard library only.
"""
import argparse
import json
import urllib.error
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

PAGE = Path(__file__).with_name("index.html")


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.0"
    upstream = ""

    def log_message(self, fmt, *args):
        print(f"{self.address_string()} {fmt % args}", flush=True)

    def reply(self, code, ctype, body):
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def reply_error(self, code, message):
        body = json.dumps({"error": {"message": message}}).encode()
        self.reply(code, "application/json", body)

    def do_GET(self):
        if self.path == "/":
            self.reply(200, "text/html; charset=utf-8", PAGE.read_bytes())
        elif self.path == "/health":
            self.check_health()
        else:
            self.reply_error(404, "not found")

    def check_health(self):
        try:
            with urllib.request.urlopen(self.upstream + "/api/version", timeout=3) as resp:
                self.reply(resp.status, "application/json", resp.read())
        except (urllib.error.URLError, OSError) as exc:
            self.reply_error(502, f"model server is not reachable: {exc}")

    def do_POST(self):
        if self.path != "/v1/chat/completions":
            self.reply_error(404, "not found")
            return
        length = int(self.headers.get("Content-Length", "0"))
        req = urllib.request.Request(
            self.upstream + self.path,
            data=self.rfile.read(length),
            headers={"Content-Type": "application/json"},
        )
        try:
            resp = urllib.request.urlopen(req, timeout=1800)
        except urllib.error.HTTPError as exc:
            self.reply(exc.code, "application/json", exc.read())
            return
        except (urllib.error.URLError, OSError) as exc:
            self.reply_error(502, f"model server is not reachable: {exc}")
            return
        with resp:
            self.send_response(200)
            self.send_header("Content-Type", resp.headers.get("Content-Type", "text/event-stream"))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            try:
                while chunk := resp.read1(65536):
                    self.wfile.write(chunk)
                    self.wfile.flush()
            except (BrokenPipeError, ConnectionResetError):
                # The browser stopped the request. Closing `resp` stops generation.
                pass


def main():
    ap = argparse.ArgumentParser(description="Serve the local chat page.")
    ap.add_argument("--port", type=int, default=8091)
    ap.add_argument("--upstream", default="http://127.0.0.1:11434")
    a = ap.parse_args()
    Handler.upstream = a.upstream.rstrip("/")
    print(f"chat page at http://127.0.0.1:{a.port}, model at {Handler.upstream}", flush=True)
    ThreadingHTTPServer(("127.0.0.1", a.port), Handler).serve_forever()


if __name__ == "__main__":
    main()
