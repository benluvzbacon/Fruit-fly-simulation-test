#!/usr/bin/env python3
"""Web server for the FlyWire fruit-fly simulation.

Stdlib-only HTTP server that:
  * serves the viewer in ``web/`` (same origin — no CORS tricks needed);
  * exposes a small JSON API over the live :class:`~fly.engine.Simulation`;
  * runs the connectome simulation continuously in a background thread,
    adapting the number of 1 ms LIF steps per wall-clock slice so the brain
    steps as fast as the hardware allows (reported to the UI).

Run:  ``python3 server.py [--port 8000 --host 0.0.0.0]``
"""
from __future__ import annotations

import argparse
import json
import os
import sqlite3
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlparse, parse_qs

import numpy as np

ROOT = os.path.dirname(os.path.abspath(__file__))
WEB = os.path.join(ROOT, "web")
PROC = os.path.join(ROOT, "data", "flywire", "processed")

from fly.engine import Simulation  # noqa: E402


# ---------------------------------------------------------------- state
class SimRunner(threading.Thread):
    """Background loop stepping the brain as fast as the duty cycle allows.

    The old loop aimed each compute slice at ~95 ms per 100 ms wall — i.e. it
    pinned a full CPU core at ~100 % and starved the browser on the same
    machine (page felt "laggy").  `duty` caps the compute fraction: with the
    default 0.55 the engine uses at most ~55 % of one core and sleeps out the
    rest, keeping the whole desktop responsive.
    """

    def __init__(self, duty: float = 0.55):
        super().__init__(daemon=True)
        self.sim = Simulation()
        self.lock = threading.RLock()
        self.running = True
        self.paused = False
        self.duty = float(np.clip(duty, 0.15, 0.95))
        self.steps_this_slice = 0
        self.slice_cost_ms = 5.0
        self.target_steps = 30
        self.start_wall = time.time()

    def run(self):
        while self.running:
            if self.paused:
                time.sleep(0.05)
                continue
            t0 = time.time()
            with self.lock:
                try:
                    self.sim.step(self.target_steps)
                except Exception as e:  # keep the server alive
                    self.sim.events.append(f"engine error: {e!r}")
                    print("engine error:", repr(e), flush=True)
                    time.sleep(0.2)
            cost = max(1.0, (time.time() - t0) * 1000.0)
            # aim each compute slice at ~55 ms of simulation...
            self.target_steps = int(np.clip(
                self.target_steps * 55.0 / cost, 6, 120))
            # ...then sleep out the remainder of the duty window
            idle_ms = cost * (1.0 - self.duty) / self.duty
            time.sleep(min(0.25, max(0.004, idle_ms / 1000.0)))

    def snapshot(self, topk: int = 4000) -> dict:
        with self.lock:
            s = self.sim.snapshot(topk)
            s["target_steps"] = self.target_steps
            s["sim_speed_x"] = round(self.target_steps / 100.0 * self.duty / 0.55, 2)
            s["wall_time_s"] = round(time.time() - self.start_wall, 1)
        return s

    def command(self, body: dict) -> dict:
        with self.lock:
            return self._command(body)

    def _command(self, b: dict) -> dict:
        sim = self.sim
        a = b.get("action")
        if a == "pause":
            self.paused = True
            sim.paused = True
        elif a == "resume":
            self.paused = False
            sim.paused = False
        elif a == "reset":
            sim.__init__()
            sim.events.append("simulation reset")
        elif a == "flash":
            sim.world.flash(float(b.get("level", 1.6)))
            sim.events.append("bright flash (looming startle)")
        elif a == "toggle_light":
            sim.world.light["on"] = not sim.world.light["on"]
            sim.events.append(f"light {'on' if sim.world.light['on'] else 'off'}")
        elif a == "stimulate":
            r = sim.stimulate(int(b["root_id"]), float(b.get("mv", 30.0)))
            sim.events.append(f"stimulated root {b['root_id']}" if r.get("ok") else r["error"])
            return r
        elif a == "drop_food":
            x, y = float(b.get("x", 300)), float(b.get("y", 300))
            sim.world.sources.append(dict(x=x, y=y, z=0.0, r=150.0,
                                          kind="food", amount=1.0, taste="sugar"))
            sim.events.append(f"sugar source dropped at ({x:.0f}, {y:.0f})")
        elif a == "drop_bitter":
            x, y = float(b.get("x", 300)), float(b.get("y", 300))
            sim.world.sources.append(dict(x=x, y=y, z=0.0, r=140.0,
                                          kind="danger", amount=1.0, taste="bitter"))
            sim.events.append(f"bitter/aversive source dropped at ({x:.0f}, {y:.0f})")
        elif a == "clear_sources":
            sim.world.sources.clear()
            sim.events.append("cleared all odor/taste sources")
        elif a == "set_hunger":
            sim.hunger = float(np.clip(b.get("value", 0.5), 0.0, 1.0))
        elif a == "set_duty":
            RUNNER.duty = float(np.clip(b.get("value", 0.55), 0.15, 0.95))
            sim.events.append(f"brain duty → {RUNNER.duty:.2f}")
        elif a == "teleport_fly":
            sim.body.pos = np.array([float(b["x"]), float(b["y"]), float(b.get("z", 0))], dtype=np.float32)
            sim.events.append(f"fly teleported to ({b['x']:.0f}, {b['y']:.0f})")
        else:
            return {"error": f"unknown action {a!r}"}
        return {"ok": True}


class NeuronDB:
    def __init__(self, path=os.path.join(PROC, "neurons.sqlite")):
        self.db = sqlite3.connect(path, check_same_thread=False)
        self.db.row_factory = sqlite3.Row
        self.lock = threading.Lock()

    def neuron(self, idx: int) -> dict | None:
        with self.lock:
            r = self.db.execute("SELECT * FROM neurons WHERE idx=?", (idx,)).fetchone()
        return dict(r) if r else None

    def search(self, q: str, limit: int = 20) -> list[dict]:
        with self.lock:
            if q.isdigit() and len(q) > 10:
                rows = self.db.execute(
                    "SELECT idx, root_id, cell_type, super_class, class FROM neurons WHERE root_id=?",
                    (int(q),)).fetchall()
            else:
                rows = self.db.execute(
                    "SELECT idx, root_id, cell_type, super_class, class FROM neurons "
                    "WHERE cell_type LIKE ? OR labels LIKE ? LIMIT ?",
                    (f"%{q}%", f"%{q}%", limit)).fetchall()
        return [dict(r) for r in rows]


RUNNER = SimRunner(duty=float(os.environ.get("SIM_DUTY", "0.55")))
DB = NeuronDB()
with open(os.path.join(PROC, "meta.json"), encoding="utf-8") as _mf:
    META = json.load(_mf)


# ---------------------------------------------------------------- API
def api_frame(args: dict) -> dict | bytes:
    topk = min(20000, max(200, int(args.get("topk", ["4000"])[0])))
    return RUNNER.snapshot(topk)


def api_meta(_args) -> dict:
    m = dict(META)
    m["paused"] = RUNNER.paused
    return m


_LAYOUT_CACHE: bytes | None = None


def api_layout(_args) -> bytes:
    """Binary layout: JSON header line, then raw arrays — BUILT ONCE, then cached.

    Header: {"n": N, "super_class": [...], "side": [...], "nt": [...]}
    Body: pos float32[N,3], super_class int16[N], side int16[N], nt int16[N]

    The header line is padded so every typed array starts 8-byte aligned —
    `new Float32Array(buf, offset, …)` throws RangeError on an unaligned
    offset, which is exactly what a 275-byte header produced.
    """
    global _LAYOUT_CACHE
    if _LAYOUT_CACHE is not None:
        return _LAYOUT_CACHE
    con = RUNNER.sim.con
    pos = np.ascontiguousarray(con.pos, dtype=np.float32)
    sup = con.super_class.astype(np.int16)
    side = con.side.astype(np.int16)
    nt = con.nt.astype(np.int16)
    header = json.dumps({"n": con.N, "super_class": META["codebooks"]["super_class"],
                         "side": META["codebooks"]["side"], "nt": META["codebooks"]["nt"]})
    pad = (-(len(header) + 1)) % 8          # header + "\n" lands on an 8-byte boundary
    body = (header + " " * pad + "\n").encode() + pos.tobytes() + sup.tobytes() + side.tobytes() + nt.tobytes()
    _LAYOUT_CACHE = body
    return body


def api_neuron(args: dict) -> dict:
    idx = int(args["idx"][0])
    con = RUNNER.sim.con
    info = DB.neuron(idx)
    if info is None:
        return {"error": "unknown idx"}
    t, w, neu = con.targets(idx)
    order = np.argsort(-np.abs(w))
    out_n = []
    rate = RUNNER.sim.net.rate
    v = RUNNER.sim.net.v
    for j in order[:25]:
        i2 = int(t[j])
        nm = con.neuropils[int(neu[j])] if int(neu[j]) < len(con.neuropils) else ""
        out_n.append({"idx": i2, "root_id": int(con.root_id[i2]),
                      "w": float(w[j]), "nt_conn": con.name_book("nt", i2),
                      "cell_type": (DB.neuron(i2) or {}).get("cell_type", ""),
                      "super_class": con.name_book("super_class", i2),
                      "neuropil": nm})
    s, wi = con.sources(idx)
    order = np.argsort(-np.abs(wi))
    in_n = []
    for j in order[:25]:
        i2 = int(s[j])
        in_n.append({"idx": i2, "root_id": int(con.root_id[i2]),
                     "w": float(wi[j]), "nt_conn": con.name_book("nt", i2),
                     "cell_type": (DB.neuron(i2) or {}).get("cell_type", ""),
                     "super_class": con.name_book("super_class", i2)})
    info.update({
        "live": {"rate": float(rate[idx]), "v": float(v[idx])},
        "out_n": out_n, "in_n": in_n,
    })
    return info


def api_trace(args: dict) -> dict:
    root = int(args["root"][0])
    depth = int(args.get("depth", ["3"])[0])
    return RUNNER.sim.trace(root, depth=depth)


def api_search(args: dict) -> dict:
    return {"results": DB.search(args.get("q", [""])[0])}


def command(body: dict) -> dict:
    return RUNNER.command(body)


API = {
    "/api/frame": api_frame, "/api/meta": api_meta, "/api/layout": api_layout,
    "/api/neuron": api_neuron, "/api/trace": api_trace, "/api/search": api_search,
}


# ---------------------------------------------------------------- http
class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, fmt, *a):
        # keep an access log: when a browser can't load assets, the 404s/errors
        # show up here and explain themselves
        sys.stderr.write("[%s] %s\n" % (time.strftime("%H:%M:%S"), fmt % a))

    def _send(self, code, body=b"", ctype="text/plain", extra=None):
        if isinstance(body, str):
            body = body.encode()
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        for k, v in (extra or {}).items():
            self.send_header(k, v)
        self.end_headers()
        try:
            if body:
                self.wfile.write(body)
        except (BrokenPipeError, ConnectionResetError):
            pass

    def do_GET(self):
        u = urlparse(self.path)
        if u.path == "/":
            return self._file("index.html")
        if u.path.startswith("/api/"):
            fn = API.get(u.path)
            if not fn:
                return self._send(404, json.dumps({"error": "no such api"}), "application/json")
            try:
                out = fn(parse_qs(u.query))
                if isinstance(out, bytes):
                    return self._send(200, out, "application/octet-stream")
                return self._send(200, json.dumps(out, default=str), "application/json")
            except Exception as e:  # noqa: BLE001
                return self._send(500, json.dumps({"error": repr(e)}), "application/json")
        # static viewer files — resolve the URL path WITHOUT platform
        # surprises: os.path.normpath("/a/b") on Windows yields "\\a\\b",
        # whose backslash survives lstrip("/") and then anchors the path
        # outside the web root (the 404-on-Windows bug).  Strip the leading
        # slash FIRST, then rebuild from /-separated components.
        rel = u.path.lstrip("/")
        if "\\" in rel or rel.startswith("..") or "/../" in rel:
            return self._send(403, "forbidden")
        safe = os.path.join(*rel.split("/")) if rel else "index.html"
        full = os.path.join(WEB, safe)
        if os.path.commonpath((os.path.abspath(full), WEB)) != WEB:
            return self._send(403, "forbidden")
        return self._file(safe)

    def do_POST(self):
        u = urlparse(self.path)
        if u.path == "/api/command":
            n = int(self.headers.get("Content-Length", 0))
            try:
                body = json.loads(self.rfile.read(n) or b"{}")
            except json.JSONDecodeError:
                return self._send(400, '{"error":"bad json"}', "application/json")
            try:
                return self._send(200, json.dumps(command(body)), "application/json")
            except Exception as e:  # noqa: BLE001
                return self._send(500, json.dumps({"error": repr(e)}), "application/json")
        return self._send(404, "not found")

    def _file(self, rel: str):
        path = os.path.join(WEB, rel)
        if not os.path.isfile(path):
            return self._send(404, "not found")
        ext = os.path.splitext(path)[1]
        ctype = {
            ".html": "text/html; charset=utf-8", ".js": "text/javascript; charset=utf-8",
            ".css": "text/css", ".json": "application/json", ".png": "image/png",
            ".jpg": "image/jpeg", ".svg": "image/svg+xml",
        }.get(ext, "application/octet-stream")
        with open(path, "rb") as f:
            self._send(200, f.read(), ctype)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--duty", type=float, default=0.55,
                    help="max CPU duty of the brain thread, 0.15-0.95 "
                         "(default 0.55 keeps the desktop smooth; raise on a fast machine)")
    args = ap.parse_args()
    RUNNER.duty = float(np.clip(args.duty, 0.15, 0.95))
    # the banner prints unicode (→); keep it safe on a cp1252 Windows console
    for _s in (sys.stdout, sys.stderr):
        try:
            _s.reconfigure(encoding="utf-8", errors="replace")
        except (AttributeError, ValueError):
            pass
    RUNNER.start()
    srv = ThreadingHTTPServer((args.host, args.port), Handler)
    print(f"FlyWire fruit-fly simulation → http://{args.host}:{args.port}")
    print(f"  brain: {RUNNER.sim.con.N:,} real v783 neurons, "
          f"{len(RUNNER.sim.con.out_idx):,} real connections")
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
