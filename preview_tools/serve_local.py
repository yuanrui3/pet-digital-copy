"""Local static server for the pet page preview (no caching). POST /rec/NAME writes the body to ROOT/rec/NAME (local
frame capture for before/after videos). usage: python3 -I serve_local.py ROOT PORT"""
import functools
import http.server
import os
import re
import sys

root, port = sys.argv[1], int(sys.argv[2])


class Handler(http.server.SimpleHTTPRequestHandler):
    def end_headers(self):
        self.send_header("Cache-Control", "no-store")
        super().end_headers()

    def do_POST(self):
        m = re.fullmatch(r"/rec/([A-Za-z0-9_.-]+)", self.path)
        if not m:
            self.send_error(404)
            return
        n = int(self.headers.get("Content-Length", 0))
        os.makedirs(os.path.join(root, "rec"), exist_ok=True)
        with open(os.path.join(root, "rec", m.group(1)), "wb") as fh:
            fh.write(self.rfile.read(n))
        self.send_response(204)
        self.end_headers()

    def log_message(self, *a):
        pass


http.server.ThreadingHTTPServer(("127.0.0.1", port), functools.partial(Handler, directory=root)).serve_forever()
