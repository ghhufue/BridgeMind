"""Bounded solver scheduling, one native context per persistent worker."""

from collections import Counter, OrderedDict, deque
from dataclasses import dataclass
import copy
import hashlib
import json
import math
import os
import threading
import time
import uuid

from .errors import SolverError
from .models import Deal, Position
from .native import NativeDdsBackend

TERMINAL = {"succeeded", "failed", "cancelled", "expired"}
PRIORITIES = {"interactive": 0, "training": 1, "offline": 2}


def _json(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _timeout(value, field, allow_none=False):
    if value is None and allow_none:
        return None
    if isinstance(value, bool) or not isinstance(value, (float, int)) or not math.isfinite(value) or value < 0:
        raise SolverError("INVALID_ARGUMENT", f"{field} must be a finite nonnegative number", field)
    return value / 1000


@dataclass(frozen=True)
class ServiceConfig:
    worker_count: int = 8
    max_queued_tasks: int = 1024
    max_batch_items: int = 256
    tt_default_mb: int = 64
    tt_max_mb: int = 128
    cache_max_bytes: int = 32 * 1024 * 1024
    max_retained_jobs: int = 4096
    job_retention_seconds: float = 300
    priority_aging_seconds: float = 5
    library_path: str | None = None

    def __post_init__(self):
        for key in ("worker_count", "max_queued_tasks", "max_batch_items", "max_retained_jobs"):
            if type(getattr(self, key)) is not int or getattr(self, key) < 1:
                raise SolverError("INVALID_ARGUMENT", f"Invalid {key}")
        if self.worker_count > 256:
            raise SolverError("INVALID_ARGUMENT", "worker_count must be <=256")
        if type(self.cache_max_bytes) is not int or self.cache_max_bytes < 0:
            raise SolverError("INVALID_ARGUMENT", "Invalid cache_max_bytes")
        if type(self.tt_default_mb) is not int or type(self.tt_max_mb) is not int or not 32 <= self.tt_default_mb <= self.tt_max_mb <= 4096:
            raise SolverError("INVALID_ARGUMENT", "Large TT memory must satisfy 32 <= default <= max <= 4096 MiB")
        for key in ("job_retention_seconds", "priority_aging_seconds"):
            value = getattr(self, key)
            if isinstance(value, bool) or not isinstance(value, (float, int)) or not math.isfinite(value) or value <= 0:
                raise SolverError("INVALID_ARGUMENT", f"Invalid {key}")


class SolverService:
    def __init__(self, config=None, *, backend_factory=None):
        self.config = config or ServiceConfig()
        cfg = self.config
        self._factory = backend_factory or (lambda: NativeDdsBackend(cfg.library_path, cfg.tt_default_mb, cfg.tt_max_mb))
        self._condition = threading.Condition()
        self._jobs = OrderedDict()
        self._requests = {}
        self._work = {}
        self._cache = OrderedDict()
        self._cache_bytes = 0
        self._counters = Counter()
        self._latencies = deque(maxlen=2048)
        self._closed = False
        self._startup_errors = []
        self._info = None
        self.instance_id = uuid.uuid4().hex
        self._threads = []
        ready = []
        for index in range(cfg.worker_count):
            event = threading.Event()
            thread = threading.Thread(target=self._worker, args=(event,), name=f"dds-worker-{index}", daemon=True)
            self._threads.append(thread)
            ready.append(event)
            thread.start()
        if not all(event.wait(30) for event in ready) or self._startup_errors:
            self.close()
            raise SolverError("SERVICE_UNAVAILABLE", "DDS worker startup failed: " + "; ".join(self._startup_errors))

    def _fail(self, item, error, status="failed"):
        item["status"], item["error"] = status, error.to_dict()
        item["_finished"] = time.monotonic()
        self._counters["failed" if status == "failed" else status] += 1

    def _public(self, item):
        return copy.deepcopy({k: v for k, v in item.items() if not k.startswith("_")})

    def _expire_locked(self):
        now = time.monotonic()
        changed = False
        for key, work in list(self._work.items()):
            if work["status"] != "queued":
                continue
            for item in work["items"]:
                if item["status"] == "queued" and item["_deadline"] is not None and now >= item["_deadline"]:
                    item["queue_ms"] = (now - item["_queued"]) * 1000
                    self._fail(item, SolverError("QUEUE_EXPIRED", "Task expired before execution", retryable=True), "expired")
                    changed = True
            if not any(i["status"] == "queued" for i in work["items"]):
                del self._work[key]
        if changed:
            self._condition.notify_all()

    def _prune_locked(self):
        now = time.monotonic()
        for job_id, job in list(self._jobs.items()):
            complete = all(i["status"] in TERMINAL for i in job["items"])
            if complete:
                job.setdefault("finished", max(i["_finished"] for i in job["items"]))
                if now - job["finished"] >= self.config.job_retention_seconds:
                    del self._jobs[job_id]
                    self._requests.pop((job["owner"], job["request_id"]), None)

    def submit(self, tasks, *, request_id=None, client_id="local", scheduling=None, schema_version=1):
        if type(schema_version) is not int or schema_version != 1:
            raise SolverError("UNSUPPORTED_SCHEMA", "Only schema_version=1 is supported")
        if not isinstance(tasks, list) or not 1 <= len(tasks) <= self.config.max_batch_items:
            raise SolverError("INVALID_ARGUMENT", "Invalid task batch size")
        if not isinstance(client_id, str) or not 1 <= len(client_id) <= 128:
            raise SolverError("INVALID_ARGUMENT", "Invalid client_id")
        request_id = uuid.uuid4().hex if request_id is None else request_id
        if not isinstance(request_id, str) or not 1 <= len(request_id) <= 128:
            raise SolverError("INVALID_ARGUMENT", "Invalid request_id")
        scheduling = scheduling if scheduling is not None else {}
        if not isinstance(scheduling, dict) or set(scheduling) - {"priority", "queue_timeout_ms"}:
            raise SolverError("INVALID_ARGUMENT", "Unknown scheduling fields")
        priority = scheduling.get("priority", "training")
        if not isinstance(priority, str) or priority not in PRIORITIES:
            raise SolverError("INVALID_ARGUMENT", "Invalid priority")
        timeout = _timeout(scheduling.get("queue_timeout_ms"), "queue_timeout_ms", allow_none=True)
        ids = [t.get("item_id") if isinstance(t, dict) else None for t in tasks]
        if any(not isinstance(i, str) or not 1 <= len(i) <= 128 for i in ids) or len(set(ids)) != len(ids):
            raise SolverError("INVALID_ARGUMENT", "item_id must be unique nonempty strings")
        try:
            fingerprint = _json({"tasks": tasks, "scheduling": scheduling})
        except (ValueError, TypeError) as error:
            raise SolverError("INVALID_ARGUMENT", "Tasks must be JSON serializable") from error
        with self._condition:
            if self._closed:
                raise SolverError("SERVICE_UNAVAILABLE", "Service is closed", retryable=True)
            self._expire_locked()
            self._prune_locked()
            previous = self._requests.get((client_id, request_id))
            if previous:
                job = self._jobs[previous]
                if job["fingerprint"] != fingerprint:
                    raise SolverError("IDEMPOTENCY_CONFLICT", "request_id was used with different contents")
                return copy.deepcopy(job["submission"])
            if len(self._jobs) >= self.config.max_retained_jobs:
                for old_id, old in list(self._jobs.items()):
                    if all(i["status"] in TERMINAL for i in old["items"]):
                        del self._jobs[old_id]
                        self._requests.pop((old["owner"], old["request_id"]), None)
                        break
                else:
                    raise SolverError("QUEUE_FULL", "Job retention capacity is full", retryable=True)
            job_id = uuid.uuid4().hex
            items = []
            for task in tasks:
                now = time.monotonic()
                item = {"item_id": task["item_id"], "state_id": task.get("state_id"),
                        "status": "queued", "cache_hit": False, "coalesced": False,
                        "queue_ms": None, "solve_ms": None, "search_nodes": None,
                        "schema_version": 1, "solver_version": self._info["version"],
                        "solver_commit": self._info.get("commit"),
                        "_queued": now, "_deadline": None if timeout is None else now + timeout,
                        "_priority": PRIORITIES[priority]}
                items.append(item)
                try:
                    if set(task) - {"item_id", "operation", "payload", "state_id"}:
                        raise SolverError("INVALID_ARGUMENT", "Unknown task fields")
                    if item["state_id"] is not None and (not isinstance(item["state_id"], str) or len(item["state_id"]) > 256):
                        raise SolverError("INVALID_ARGUMENT", "Invalid state_id")
                    operation = task.get("operation")
                    if operation == "table":
                        payload = Deal.from_dict(task.get("payload"))
                    elif operation == "position":
                        payload = Position.from_dict(task.get("payload"))
                    else:
                        raise SolverError("INVALID_ARGUMENT", "Unknown operation")
                    key = hashlib.sha256(_json([1, self._info["identity"], operation, payload.to_dict()]).encode()).hexdigest()
                    item["_key"] = key
                    if key in self._cache:
                        value, _ = self._cache[key]
                        self._cache.move_to_end(key)
                        item.update(status="succeeded", result=copy.deepcopy(value), cache_hit=True, queue_ms=0, _finished=now)
                        self._counters["cache_hits"] += 1
                        continue
                    work = self._work.get(key)
                    if work:
                        item["coalesced"] = True
                        self._counters["coalesced"] += 1
                        if work["status"] == "running":
                            item.update(status="running", queue_ms=0)
                        work["items"].append(item)
                        continue
                    if timeout == 0:
                        raise SolverError("QUEUE_EXPIRED", "Task expired before execution", retryable=True)
                    if sum(w["status"] == "queued" for w in self._work.values()) >= self.config.max_queued_tasks:
                        raise SolverError("QUEUE_FULL", "Task queue is full", retryable=True)
                    self._work[key] = {"status": "queued", "operation": operation, "payload": payload, "items": [item]}
                except SolverError as error:
                    self._fail(item, error, "expired" if error.code == "QUEUE_EXPIRED" else "failed")
            submission = {"job_id": job_id, "service_instance_id": self.instance_id,
                          "items": [self._public(i) for i in items]}
            self._jobs[job_id] = {"owner": client_id, "request_id": request_id, "fingerprint": fingerprint,
                                  "items": items, "submission": copy.deepcopy(submission)}
            self._requests[(client_id, request_id)] = job_id
            self._counters["submitted"] += len(items)
            self._condition.notify_all()
            return submission

    def _job(self, job_id, client_id):
        job = self._jobs.get(job_id)
        if not job or job["owner"] != client_id:
            raise SolverError("JOB_NOT_FOUND", "Job not found for this client")
        return job

    def _snapshot(self, job_id, job):
        return {"job_id": job_id, "service_instance_id": self.instance_id,
                "complete": all(i["status"] in TERMINAL for i in job["items"]),
                "items": [self._public(i) for i in job["items"]]}

    def get_job(self, job_id, *, client_id="local"):
        with self._condition:
            self._expire_locked()
            self._prune_locked()
            return self._snapshot(job_id, self._job(job_id, client_id))

    def wait(self, job_id, wait_timeout_ms=30000, *, client_id="local"):
        timeout = _timeout(wait_timeout_ms, "wait_timeout_ms")
        deadline = time.monotonic() + timeout
        with self._condition:
            job = self._job(job_id, client_id)
            while True:
                self._expire_locked()
                snapshot = self._snapshot(job_id, job)
                remaining = deadline - time.monotonic()
                if snapshot["complete"] or remaining <= 0:
                    return snapshot
                self._condition.wait(min(remaining, 0.05))

    def cancel(self, job_id, item_ids=None, *, client_id="local"):
        with self._condition:
            job = self._job(job_id, client_id)
            if item_ids is not None and (not isinstance(item_ids, list) or any(not isinstance(i, str) for i in item_ids)):
                raise SolverError("INVALID_ARGUMENT", "item_ids must be an array of strings")
            selected = {i["item_id"] for i in job["items"]} if item_ids is None else set(item_ids)
            if selected - {i["item_id"] for i in job["items"]}:
                raise SolverError("INVALID_ARGUMENT", "Unknown item_id")
            results = []
            for item in job["items"]:
                if item["item_id"] not in selected:
                    continue
                disposition = "already_terminal"
                if item["status"] == "queued":
                    item["status"] = "cancelled"
                    item["_finished"] = time.monotonic()
                    self._counters["cancelled"] += 1
                    disposition = "cancelled"
                elif item["status"] == "running":
                    disposition = "already_running"
                results.append({"item_id": item["item_id"], "disposition": disposition})
            self._expire_locked()
            return {"job_id": job_id, "items": results}

    def _cache_result(self, key, result):
        size = len(_json(result).encode()) + len(key)
        if size > self.config.cache_max_bytes:
            return
        while self._cache and self._cache_bytes + size > self.config.cache_max_bytes:
            _, (_, old_size) = self._cache.popitem(last=False)
            self._cache_bytes -= old_size
        self._cache[key] = (result, size)
        self._cache_bytes += size

    def _worker(self, ready):
        backend = None
        try:
            backend = self._factory()
            with self._condition:
                if self._info is not None and self._info["identity"] != backend.info["identity"]:
                    raise SolverError("SERVICE_UNAVAILABLE", "Worker solver identity mismatch")
                self._info = dict(backend.info)
            ready.set()
            while True:
                with self._condition:
                    self._expire_locked()
                    queued = [(key, work) for key, work in self._work.items() if work["status"] == "queued"]
                    if self._closed:
                        return
                    if not queued:
                        self._condition.wait()
                        continue
                    now = time.monotonic()
                    def rank(pair):
                        return min(i["_priority"] - (now - i["_queued"]) / self.config.priority_aging_seconds
                                   for i in pair[1]["items"] if i["status"] == "queued")
                    key, work = min(queued, key=rank)
                    work["status"] = "running"
                    for item in work["items"]:
                        if item["status"] == "queued":
                            item.update(status="running", queue_ms=(now - item["_queued"]) * 1000)
                start = time.perf_counter()
                result, failure = None, None
                try:
                    result = (backend.solve_table(work["payload"]) if work["operation"] == "table"
                              else backend.solve_position(work["payload"])).to_dict()
                except SolverError as error:
                    failure = error
                except Exception as error:
                    failure = SolverError("NATIVE_SOLVER_ERROR", f"Solver invocation failed: {error}")
                elapsed = (time.perf_counter() - start) * 1000
                with self._condition:
                    if result is not None:
                        self._cache_result(key, result)
                    for item in work["items"]:
                        if item["status"] != "running":
                            continue
                        item["solve_ms"] = elapsed
                        if failure:
                            self._fail(item, failure)
                        else:
                            item.update(status="succeeded", result=copy.deepcopy(result), search_nodes=result.get("search_nodes"), _finished=time.monotonic())
                            self._counters["succeeded"] += 1
                        self._latencies.append((item["queue_ms"], elapsed))
                    self._work.pop(key, None)
                    self._counters["native_calls"] += 1
                    self._condition.notify_all()
        except Exception as error:
            with self._condition:
                self._startup_errors.append(str(error))
                if ready.is_set():
                    self._closed = True
                    for work in self._work.values():
                        for item in work["items"]:
                            if item["status"] not in TERMINAL:
                                self._fail(item, SolverError("SERVICE_UNAVAILABLE", "Worker stopped", retryable=True))
                    self._work.clear()
                self._condition.notify_all()
        finally:
            ready.set()
            if backend:
                backend.close()

    def get_info(self):
        return {"schema_version": 1, "solver_version": self._info["version"],
                "solver_commit": self._info.get("commit"), "solver_identity": self._info["identity"],
                "service_instance_id": self.instance_id, "process_id": os.getpid(), "operations": ["table", "position"],
                "worker_count": self.config.worker_count, "max_batch_items": self.config.max_batch_items,
                "tt_default_mb": self.config.tt_default_mb, "tt_max_mb": self.config.tt_max_mb,
                "max_queued_tasks": self.config.max_queued_tasks,
                "hard_cancellation": False}

    def get_stats(self):
        with self._condition:
            self._expire_locked()
            self._prune_locked()
            def percentile(index, p):
                values = sorted(pair[index] for pair in self._latencies)
                return values[min(len(values) - 1, math.ceil(len(values) * p) - 1)] if values else None
            return {**dict(self._counters), "cache_hits": self._counters["cache_hits"],
                    "queued": sum(w["status"] == "queued" for w in self._work.values()),
                    "running": sum(w["status"] == "running" for w in self._work.values()),
                    "retained_jobs": len(self._jobs), "cache_entries": len(self._cache),
                    "cache_serialized_bytes": self._cache_bytes, "closed": self._closed,
                    "queue_ms_p95": percentile(0, 0.95), "queue_ms_p99": percentile(0, 0.99),
                    "solve_ms_p95": percentile(1, 0.95), "latency_sample_count": len(self._latencies)}

    def close(self):
        with self._condition:
            self._closed = True
            for key, work in list(self._work.items()):
                if work["status"] == "queued":
                    for item in work["items"]:
                        if item["status"] == "queued":
                            self._fail(item, SolverError("SERVICE_UNAVAILABLE", "Service is closing", retryable=True))
                    del self._work[key]
            self._condition.notify_all()
        for thread in self._threads:
            if thread is not threading.current_thread():
                thread.join()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()
