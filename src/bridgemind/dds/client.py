"""Thread-safe multiplexed client. No DDS library is loaded in the caller."""

from concurrent.futures import Future, TimeoutError as FutureTimeout
from dataclasses import asdict, dataclass
import socket
import threading
import time
import uuid

from .errors import SolverError
from .models import PositionResult, TableResult
from .protocol import recv_message, send_message
from .service import _timeout


@dataclass(frozen=True)
class Options:
    wait_timeout_ms: float = 30000
    queue_timeout_ms: float | None = None
    priority: str = "training"
    state_id: str | None = None


class SolverWaitTimeout(SolverError):
    def __init__(self, job_id):
        super().__init__("WAIT_TIMEOUT", "Stopped waiting; the solver job may still be running")
        self.job_id = job_id


class SolverClient:
    def __init__(self, host="127.0.0.1", port=18765, *, client_id=None, connect_timeout=5):
        if host != "127.0.0.1":
            raise SolverError("INVALID_ARGUMENT", "v1 only supports 127.0.0.1")
        self.client_id = client_id or uuid.uuid4().hex
        self._pending = {}
        self._lock = threading.Lock()
        self._write_lock = threading.Lock()
        self._closed = False
        try:
            self._socket = socket.create_connection((host, port), timeout=connect_timeout)
            self._socket.settimeout(None)
        except OSError as error:
            raise SolverError("SERVICE_UNAVAILABLE", f"Cannot connect to DDS service: {error}", retryable=True) from error
        self._reader = threading.Thread(target=self._read_loop, name="dds-client-reader", daemon=True)
        self._reader.start()

    def _read_loop(self):
        try:
            while True:
                message = recv_message(self._socket)
                if message.get("schema_version") != 1:
                    raise SolverError("UNSUPPORTED_SCHEMA", "Unsupported response schema")
                with self._lock:
                    future = self._pending.pop(message.get("request_id"), None)
                if future is not None:
                    if "error" in message:
                        future.set_exception(SolverError.from_dict(message["error"]))
                    else:
                        future.set_result(message["result"])
        except (OSError, EOFError, SolverError, KeyError, TypeError, ValueError) as error:
            self._fail_pending(SolverError("SERVICE_UNAVAILABLE", f"DDS connection ended: {error}", retryable=True))

    def _fail_pending(self, error):
        with self._lock:
            self._closed = True
            pending, self._pending = self._pending, {}
        for future in pending.values():
            future.set_exception(error)

    def _rpc(self, method, params=None):
        request_id, future = uuid.uuid4().hex, Future()
        with self._lock:
            if self._closed:
                raise SolverError("SERVICE_UNAVAILABLE", "DDS client is closed", retryable=True)
            self._pending[request_id] = future
        try:
            with self._write_lock:
                send_message(self._socket, {"schema_version": 1, "request_id": request_id,
                                             "client_id": self.client_id, "method": method, "params": params or {}})
            return future.result(timeout=10)
        except FutureTimeout as error:
            raise SolverError("SERVICE_UNAVAILABLE", "DDS control request timed out", retryable=True) from error
        except OSError as error:
            raise SolverError("SERVICE_UNAVAILABLE", f"DDS connection failed: {error}", retryable=True) from error
        finally:
            with self._lock:
                self._pending.pop(request_id, None)

    def submit(self, tasks, *, scheduling=None, request_id=None):
        return self._rpc("submit", {"tasks": tasks, "scheduling": scheduling,
                                    "request_id": request_id})

    def get_job(self, job_id):
        return self._rpc("get_job", {"job_id": job_id})

    def wait(self, job_id, wait_timeout_ms=30000):
        seconds = _timeout(wait_timeout_ms, "wait_timeout_ms")
        deadline = time.monotonic() + seconds
        while True:
            remaining = max(0, (deadline - time.monotonic()) * 1000)
            result = self._rpc("wait", {"job_id": job_id, "wait_timeout_ms": min(remaining, 1000)})
            if result["complete"] or time.monotonic() >= deadline:
                return result

    def cancel(self, job_id, item_ids=None):
        return self._rpc("cancel", {"job_id": job_id, "item_ids": item_ids})

    def get_info(self):
        return self._rpc("get_info")

    def get_stats(self):
        return self._rpc("get_stats")

    def _solve(self, operation, payloads, options):
        options = asdict(options) if isinstance(options, Options) else dict(options or {})
        allowed = {"wait_timeout_ms", "queue_timeout_ms", "priority", "state_id"}
        if set(options) - allowed:
            raise SolverError("INVALID_ARGUMENT", "Unknown solver options")
        wait_ms = options.pop("wait_timeout_ms", 30000)
        _timeout(wait_ms, "wait_timeout_ms")
        state_id = options.pop("state_id", None)
        tasks = [{"item_id": str(index), "operation": operation,
                  "payload": payload.to_dict() if hasattr(payload, "to_dict") else payload,
                  "state_id": state_id} for index, payload in enumerate(payloads)]
        submitted = self.submit(tasks, scheduling=options)
        result = self.wait(submitted["job_id"], wait_ms)
        if not result["complete"]:
            raise SolverWaitTimeout(submitted["job_id"])
        return result["items"]

    def solve_tables(self, deals, options=None):
        return self._solve("table", deals, options)

    def solve_positions(self, positions, options=None):
        return self._solve("position", positions, options)

    def _single(self, operation, payload, options, result_type):
        item = self._solve(operation, [payload], options)[0]
        if item["status"] != "succeeded":
            raise SolverError.from_dict(item.get("error", SolverError("CANCELLED", "Task was cancelled").to_dict()))
        return result_type.from_dict(item["result"])

    def solve_table(self, deal, options=None):
        return self._single("table", deal, options, TableResult)

    def solve_position(self, position, options=None):
        return self._single("position", position, options, PositionResult)

    def close(self):
        self._fail_pending(SolverError("SERVICE_UNAVAILABLE", "DDS client closed", retryable=True))
        try:
            self._socket.shutdown(socket.SHUT_RDWR)
        except OSError:
            pass
        self._socket.close()
        if threading.current_thread() is not self._reader:
            self._reader.join(timeout=2)

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()

