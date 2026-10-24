import threading
import time

import pytest

from bridgemind.dds import SolverService, ServiceConfig, SolverError
from bridgemind.dds.models import TableResult


class GateBackend:
    info = {"version": "test", "commit": "test", "identity": "gate"}

    def __init__(self, started, release, calls):
        self.started, self.release, self.calls = started, release, calls

    def solve_table(self, deal):
        self.calls.append(threading.get_ident())
        self.started.set()
        assert self.release.wait(5), "test gate not released"
        return TableResult(tuple((0, 0, 0, 0) for _ in range(5)))

    def close(self):
        pass


def task(deal, item_id="a"):
    return {"item_id": item_id, "operation": "table", "payload": deal.to_dict()}


def test_cache_partial_failure_and_idempotency(deal):
    with SolverService(ServiceConfig(worker_count=2)) as service:
        tasks = [task(deal), {"item_id": "bad", "operation": "table", "payload": {"hands": {}}}]
        first = service.submit(tasks, request_id="same", client_id="owner")
        assert service.submit(tasks, request_id="same", client_id="owner") == first
        result = service.wait(first["job_id"], 10000, client_id="owner")
        assert [x["status"] for x in result["items"]] == ["succeeded", "failed"]
        second = service.submit([task(deal)])
        cached = service.wait(second["job_id"], 1000)
        assert cached["items"][0]["cache_hit"]
        with pytest.raises(SolverError) as error:
            service.submit([task(deal)], request_id="same", client_id="owner")
        assert error.value.code == "IDEMPOTENCY_CONFLICT"
        with pytest.raises(SolverError):
            service.get_job(first["job_id"], client_id="another")


def test_queue_capacity_expiry_and_cancelling_shared_work(deal):
    started, release, calls = threading.Event(), threading.Event(), []
    factory = lambda: GateBackend(started, release, calls)
    config = ServiceConfig(worker_count=1, max_queued_tasks=1)
    with SolverService(config, backend_factory=factory) as service:
        first = service.submit([task(deal)])
        assert started.wait(2)
        shared = service.submit([task(deal, "shared")])
        assert service.cancel(first["job_id"])["items"][0]["disposition"] == "already_running"
        # Different seat allocation forces a distinct cache key.
        rotated = deal.to_dict()
        rotated["hands"]["N"], rotated["hands"]["E"] = rotated["hands"]["E"], rotated["hands"]["N"]
        waiting = service.submit([{"item_id": "waiting", "operation": "table", "payload": rotated}], scheduling={"queue_timeout_ms": 1})
        overflow = service.submit([{"item_id": "overflow", "operation": "table", "payload": rotated}])
        # Same queued computation coalesces even when the queue is full.
        assert overflow["items"][0]["coalesced"]
        time.sleep(0.02)
        release.set()
        assert service.wait(shared["job_id"], 2000)["items"][0]["status"] == "succeeded"
        assert service.wait(waiting["job_id"], 2000)["items"][0]["status"] == "expired"
        assert service.wait(overflow["job_id"], 2000)["items"][0]["status"] == "succeeded"
        assert len(calls) == 2


def test_queued_cancel_does_not_cancel_other_subscriber(deal):
    started, release, calls = threading.Event(), threading.Event(), []
    with SolverService(ServiceConfig(worker_count=1), backend_factory=lambda: GateBackend(started, release, calls)) as service:
        running = service.submit([task(deal)])
        assert started.wait(2)
        swapped = deal.to_dict()
        swapped["hands"]["N"], swapped["hands"]["S"] = swapped["hands"]["S"], swapped["hands"]["N"]
        pending = {"item_id": "x", "operation": "table", "payload": swapped}
        a, b = service.submit([pending]), service.submit([pending])
        assert service.cancel(a["job_id"])["items"][0]["disposition"] == "cancelled"
        release.set()
        assert service.wait(a["job_id"], 2000)["items"][0]["status"] == "cancelled"
        assert service.wait(b["job_id"], 2000)["items"][0]["status"] == "succeeded"
        assert service.wait(running["job_id"], 2000)["complete"]


def test_queue_full_rejects_distinct_work_and_wait_timeout_preserves_job(deal):
    started, release, calls = threading.Event(), threading.Event(), []
    with SolverService(ServiceConfig(worker_count=1, max_queued_tasks=1), backend_factory=lambda: GateBackend(started, release, calls)) as service:
        a = service.submit([task(deal)])
        assert started.wait(2)
        rotated = deal.to_dict()
        rotated["hands"]["N"], rotated["hands"]["E"] = rotated["hands"]["E"], rotated["hands"]["N"]
        b = service.submit([{"item_id": "b", "operation": "table", "payload": rotated}])
        another = deal.to_dict()
        another["hands"]["N"], another["hands"]["W"] = another["hands"]["W"], another["hands"]["N"]
        rejected = service.submit([{"item_id": "c", "operation": "table", "payload": another}])
        assert rejected["items"][0]["error"]["code"] == "QUEUE_FULL"
        assert not service.wait(a["job_id"], 0)["complete"]
        assert service.get_stats()["running"] == 1
        release.set()
        assert service.wait(a["job_id"], 2000)["complete"]
        assert service.wait(b["job_id"], 2000)["complete"]


def test_retention_does_not_break_retry_at_capacity(deal):
    with SolverService(ServiceConfig(worker_count=1, max_retained_jobs=1)) as service:
        first = service.submit([task(deal)], request_id="retry")
        assert service.wait(first["job_id"], 5000)["complete"]
        assert service.submit([task(deal)], request_id="retry") == first


def test_distinct_random_deals_match_serial_backend():
    import random
    from bridgemind.dds import Deal, NativeDdsBackend
    from bridgemind.dds.models import CARDS, SEATS
    rng = random.Random(309)
    deals = []
    for _ in range(12):
        cards = list(CARDS)
        rng.shuffle(cards)
        deals.append(Deal({seat: cards[i * 13:(i + 1) * 13] for i, seat in enumerate(SEATS)}))
    with NativeDdsBackend(tt_default_mb=32, tt_max_mb=64) as backend:
        expected = [backend.solve_table(deal).tricks for deal in deals]
    with SolverService(ServiceConfig(worker_count=4, tt_default_mb=32, tt_max_mb=64)) as service:
        submission = service.submit([task(deal, str(i)) for i, deal in enumerate(deals)])
        items = service.wait(submission["job_id"], 15000)["items"]
        assert all(item["status"] == "succeeded" for item in items)
        assert [tuple(tuple(row) for row in item["result"]["tricks"]) for item in items] == expected
        assert service.get_stats()["native_calls"] == 12


def test_job_retention_starts_when_computation_completes(deal):
    with SolverService(ServiceConfig(worker_count=1, job_retention_seconds=0.02)) as service:
        job = service.submit([task(deal)])
        assert service.wait(job["job_id"], 5000)["complete"]
        time.sleep(0.04)
        with pytest.raises(SolverError) as error:
            service.get_job(job["job_id"])
        assert error.value.code == "JOB_NOT_FOUND"


@pytest.mark.parametrize("request_id", ["", 0, False, []])
def test_invalid_business_request_id_is_not_silently_replaced(deal, request_id):
    with SolverService(ServiceConfig(worker_count=1)) as service:
        with pytest.raises(SolverError) as error:
            service.submit([task(deal)], request_id=request_id)
        assert error.value.code == "INVALID_ARGUMENT"

