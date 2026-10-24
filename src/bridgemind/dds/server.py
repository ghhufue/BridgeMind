"""Local DDS service: python -m bridgemind.dds.server --workers 8."""

import argparse
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import socket
import socketserver
import threading

from .errors import SolverError
from .protocol import recv_message, send_message
from .service import ServiceConfig, SolverService, _timeout


class _Handler(socketserver.BaseRequestHandler):
    def handle(self):
        write_lock = threading.Lock()
        slots = threading.BoundedSemaphore(16)
        self.request.settimeout(30)

        def answer(message):
            response = {"schema_version": 1, "request_id": message.get("request_id")}
            try:
                response["result"] = self.server.dispatch(message)
            except SolverError as error:
                response["error"] = error.to_dict()
            except (TypeError, ValueError, KeyError) as error:
                response["error"] = SolverError("INVALID_ARGUMENT", f"Invalid method parameters: {error}").to_dict()
            except Exception:
                response["error"] = SolverError("INTERNAL_ERROR", "Service request failed").to_dict()
            try:
                with write_lock:
                    send_message(self.request, response)
            except (OSError, SolverError):
                pass
            finally:
                slots.release()

        try:
            while True:
                message = recv_message(self.request)
                slots.acquire()
                try:
                    self.server.control_pool.submit(answer, message)
                except RuntimeError:
                    slots.release()
                    return
        except (OSError, EOFError, SolverError):
            return


class SolverServer(socketserver.ThreadingTCPServer):
    allow_reuse_address = True
    daemon_threads = True

    def __init__(self, service, port=18765):
        self.service = service
        self.control_pool = ThreadPoolExecutor(max_workers=16, thread_name_prefix="dds-control")
        self.connections = threading.BoundedSemaphore(32)
        super().__init__(("127.0.0.1", port), _Handler)

    def process_request(self, request, client_address):
        if not self.connections.acquire(blocking=False):
            self.shutdown_request(request)
            return
        try:
            super().process_request(request, client_address)
        except BaseException:
            self.connections.release()
            raise

    def process_request_thread(self, request, client_address):
        try:
            super().process_request_thread(request, client_address)
        finally:
            self.connections.release()

    def dispatch(self, message):
        if set(message) != {"schema_version", "request_id", "client_id", "method", "params"}:
            raise SolverError("INVALID_ARGUMENT", "Invalid request envelope")
        if type(message["schema_version"]) is not int or message["schema_version"] != 1:
            raise SolverError("UNSUPPORTED_SCHEMA", "Only schema_version=1 is supported")
        for key in ("request_id", "client_id"):
            if not isinstance(message[key], str) or not 1 <= len(message[key]) <= 128:
                raise SolverError("INVALID_ARGUMENT", f"Invalid {key}")
        params, method = message["params"], message["method"]
        if not isinstance(params, dict):
            raise SolverError("INVALID_ARGUMENT", "params must be an object")
        if method in ("get_info", "get_stats") and not params:
            return getattr(self.service, method)()
        methods = {"submit": {"tasks", "scheduling", "request_id"}, "get_job": {"job_id"},
                   "wait": {"job_id", "wait_timeout_ms"}, "cancel": {"job_id", "item_ids"}}
        if not isinstance(method, str) or method not in methods or set(params) - methods[method]:
            raise SolverError("INVALID_ARGUMENT", "Unknown method or parameters")
        if method == "wait":
            params = dict(params)
            timeout = _timeout(params.get("wait_timeout_ms", 1000), "wait_timeout_ms")
            params["wait_timeout_ms"] = min(timeout * 1000, 1000)
        return getattr(self.service, method)(**params, client_id=message["client_id"])

    def server_close(self):
        super().server_close()
        self.control_pool.shutdown(wait=True, cancel_futures=True)


def main(argv=None):
    parser = argparse.ArgumentParser(description="BridgeMind local DDS service")
    parser.add_argument("--port", type=int, default=18765)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--library")
    parser.add_argument("--queue-capacity", type=int, default=1024)
    parser.add_argument("--cache-mb", type=int, default=32)
    parser.add_argument("--tt-default-mb", type=int, default=64)
    parser.add_argument("--tt-max-mb", type=int, default=128)
    parser.add_argument("--ready-file", help="Write endpoint JSON after successful startup")
    args = parser.parse_args(argv)
    try:
        config = ServiceConfig(worker_count=args.workers, library_path=args.library,
                               max_queued_tasks=args.queue_capacity, cache_max_bytes=args.cache_mb * 1024 * 1024,
                               tt_default_mb=args.tt_default_mb, tt_max_mb=args.tt_max_mb)
        with SolverService(config) as service, SolverServer(service, args.port) as server:
            ready = {"host": "127.0.0.1", "port": server.server_address[1], **service.get_info()}
            if args.ready_file:
                path = Path(args.ready_file)
                path.parent.mkdir(parents=True, exist_ok=True)
                temporary = path.with_suffix(path.suffix + ".tmp")
                temporary.write_text(json.dumps(ready), encoding="utf-8")
                temporary.replace(path)
            print(json.dumps(ready), flush=True)
            try:
                server.serve_forever(poll_interval=0.1)
            except KeyboardInterrupt:
                pass
        return 0
    except (SolverError, OSError) as error:
        parser.exit(1, f"DDS service startup failed: {error}\n")


if __name__ == "__main__":
    raise SystemExit(main())

