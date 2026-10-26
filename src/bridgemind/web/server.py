"""Serve the built card table and real analysis APIs on the trusted LAN."""

import argparse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import mimetypes
import os
import shutil
from pathlib import Path
import threading
from urllib.parse import unquote, urlsplit

from bridgemind.dds import SolverError
from .analysis import AnalysisService
from .updates import apk_file, latest_release


class LanServer(ThreadingHTTPServer):
    daemon_threads = True
    allow_reuse_address = True

    def __init__(self, address, service, web_root, origins=(), update_root=Path(".build/releases/android")):
        self.service = service
        self.web_root = Path(web_root).resolve()
        self.update_root = Path(update_root).resolve()
        host, port = address
        self.origins = {f"http://{h}:{p}" for h in (host, "localhost", "127.0.0.1") for p in (port, 5173)}
        self.origins.update({"http://localhost", "https://localhost", *origins})
        self.hosts = {host, "localhost", "127.0.0.1"}
        self.slots = threading.BoundedSemaphore(24)
        super().__init__(address, Handler)

    def process_request(self, request, client_address):
        if not self.slots.acquire(blocking=False):
            self.shutdown_request(request)
            return
        try:
            super().process_request(request, client_address)
        except BaseException:
            self.slots.release()
            raise

    def process_request_thread(self, request, client_address):
        try:
            super().process_request_thread(request, client_address)
        finally:
            self.slots.release()


class Handler(BaseHTTPRequestHandler):
    server_version = "BridgeMind/1"

    def setup(self):
        super().setup()
        self.connection.settimeout(20)

    def allowed(self):
        host = urlsplit("http://" + self.headers.get("Host", "")).hostname
        if host not in self.server.hosts:
            self.json(400, {"error": "Invalid Host"})
            return False
        origin = self.headers.get("Origin")
        if origin and origin not in self.server.origins:
            self.json(403, {"error": "Origin is not allowed"})
            return False
        return True

    def cors(self):
        origin = self.headers.get("Origin")
        if origin in self.server.origins:
            self.send_header("Access-Control-Allow-Origin", origin)
            self.send_header("Vary", "Origin")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.send_header("Access-Control-Allow-Private-Network", "true")

    def json(self, code, value):
        data = json.dumps(value, ensure_ascii=False, allow_nan=False).encode("utf-8")
        self.send_response(code)
        self.cors()
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.end_headers()
        try:
            self.wfile.write(data)
        except (BrokenPipeError, ConnectionResetError):
            pass

    def do_OPTIONS(self):
        if self.allowed():
            self.send_response(204)
            self.cors()
            self.send_header("Content-Length", "0")
            self.end_headers()

    def do_GET(self):
        if not self.allowed():
            return
        path = urlsplit(self.path).path
        if path == "/api/v1/models":
            self.json(200, self.server.service.model_catalog())
            return
        if path == "/api/v1/updates/android":
            try:
                self.json(200, latest_release(self.server.update_root))
            except FileNotFoundError:
                self.json(404, {"error": "No Android release published"})
            except (OSError, ValueError, TypeError):
                self.json(503, {"error": "Android release is invalid"})
            return
        if path.startswith("/updates/"):
            try:
                prefix = "/updates/android/"
                if not path.startswith(prefix):
                    raise FileNotFoundError()
                target = apk_file(self.server.update_root, path[len(prefix):])
                with target.open("rb") as stream:
                    self.send_response(200)
                    self.send_header("Content-Type", "application/vnd.android.package-archive")
                    self.send_header("Content-Length", str(os.fstat(stream.fileno()).st_size))
                    self.send_header("Cache-Control", "public, max-age=31536000, immutable")
                    self.send_header("X-Content-Type-Options", "nosniff")
                    self.end_headers()
                    shutil.copyfileobj(stream, self.wfile, length=65536)
            except FileNotFoundError:
                self.json(404, {"error": "Update package not found"})
            except (BrokenPipeError, ConnectionResetError):
                pass
            return
        if path == "/api/v1/health":
            self.json(200, {**self.server.service.health(), "processId": os.getpid()})
            return
        if path.startswith("/api/"):
            self.json(404, {"error": "Unknown API"})
            return
        target = (self.server.web_root / (unquote(path).lstrip("/") or "index.html")).resolve()
        if not target.is_relative_to(self.server.web_root) or not target.is_file():
            self.json(404, {"error": "File not found"})
            return
        data = target.read_bytes()
        mime = {".js": "application/javascript", ".css": "text/css", ".svg": "image/svg+xml"}.get(target.suffix)
        mime = mime or mimetypes.guess_type(str(target))[0] or "application/octet-stream"
        self.send_response(200)
        self.send_header("Content-Type", mime)
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "no-cache" if target.name == "index.html" else "public, max-age=3600")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.end_headers()
        try:
            self.wfile.write(data)
        except (BrokenPipeError, ConnectionResetError):
            pass

    def do_POST(self):
        if not self.allowed():
            return
        routes = {"/api/v1/analysis/ai": "ai", "/api/v1/analysis/dds": "dds", "/api/v1/bidding/ai": "bidding"}
        source = routes.get(urlsplit(self.path).path)
        if source is None:
            self.json(404, {"error": "Unknown API"})
            return
        if self.headers.get_content_type() != "application/json":
            self.json(415, {"error": "Content-Type must be application/json"})
            return
        try:
            length = int(self.headers.get("Content-Length", "0"))
            if not 0 < length <= 131072 or self.headers.get("Transfer-Encoding"):
                self.json(413, {"error": "Request must be 1..131072 bytes"})
                return
            data = self.rfile.read(length)
            if len(data) != length:
                raise ValueError("Incomplete request body")
            request = json.loads(data)
            self.json(200, self.server.service.analyze(request, source))
        except (ValueError, TypeError, KeyError) as error:
            self.json(400, {"error": str(error)})
        except SolverError as error:
            code = 504 if error.code == "WAIT_TIMEOUT" else 503
            self.json(code, {"error": "DDS 服务暂不可用，请稍后重试", "code": error.code})
        except TimeoutError:
            self.json(429, {"error": "分析服务正忙，请稍后重试"})
        except Exception:
            import traceback
            traceback.print_exc()
            self.json(503, {"error": "分析服务暂不可用，请查看电脑上的服务日志"})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--solver-port", type=int, default=18765)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--bidding-checkpoint", type=Path, default=Path("runs/sl-pilot/last.pt"))
    parser.add_argument("--web-root", type=Path, default=Path("frontend/dist"))
    parser.add_argument("--update-root", type=Path, default=Path(".build/releases/android"))
    parser.add_argument("--origin", action="append", default=[])
    args = parser.parse_args()
    policy, name = None, None
    if args.checkpoint:
        import torch
        from bridgemind.cardplay.policy import NeuralPlayPolicy
        torch.set_num_threads(4)
        policy = NeuralPlayPolicy(args.checkpoint, device="cpu", stochastic=False)
        name = f"{args.checkpoint.parent.name} · 纯网络"
    bidder, bidding_name = None, None
    if args.bidding_checkpoint and args.bidding_checkpoint.is_file():
        from bridgemind.training.policy import BiddingPolicy
        bidder = BiddingPolicy(args.bidding_checkpoint, device="cpu")
        bidding_name = args.bidding_checkpoint.parent.name
    from bridgemind.training.runner import sha256
    service = AnalysisService(policy, model_name=name, model_sha256=sha256(args.checkpoint) if args.checkpoint else None,
                              bidder=bidder, bidding_name=bidding_name, solver_port=args.solver_port)
    from .models import ModelRegistry
    service.models = ModelRegistry(service, Path.cwd())
    server = LanServer((args.host, args.port), service, args.web_root, args.origin, args.update_root)
    print(json.dumps({"url": f"http://{args.host}:{args.port}", "processId": os.getpid(),
                      "model": name, "webRoot": str(args.web_root.resolve())}, ensure_ascii=False), flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
