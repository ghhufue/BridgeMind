"""Small reproducible end-to-end benchmark, excluding service startup."""

import argparse
import ctypes as ct
import json
import os
from pathlib import Path
import random
import signal
import subprocess
import sys
import tempfile
import time

from bridgemind.dds import Deal, SolverClient
from bridgemind.dds.models import CARDS, SEATS


def process_memory(pid):
    if os.name != "nt":
        return None
    class Memory(ct.Structure):
        _fields_ = [("cb", ct.c_ulong), ("faults", ct.c_ulong)] + [
            (name, ct.c_size_t) for name in ("peak_ws", "ws", "peak_paged", "paged", "peak_nonpaged", "nonpaged", "pagefile", "peak_pagefile")]
    kernel = ct.WinDLL("kernel32", use_last_error=True)
    kernel.OpenProcess.argtypes = [ct.c_ulong, ct.c_int, ct.c_ulong]
    kernel.OpenProcess.restype = ct.c_void_p
    kernel.CloseHandle.argtypes = [ct.c_void_p]
    handle = kernel.OpenProcess(0x410, False, pid)
    if not handle:
        return None
    try:
        value = Memory()
        value.cb = ct.sizeof(value)
        fn = ct.WinDLL("psapi").GetProcessMemoryInfo
        fn.argtypes = [ct.c_void_p, ct.POINTER(Memory), ct.c_ulong]
        if not fn(handle, ct.byref(value), ct.sizeof(value)):
            return None
        return {"working_set_mib": round(value.ws / 1024**2, 1),
                "peak_working_set_mib": round(value.peak_ws / 1024**2, 1)}
    finally:
        kernel.CloseHandle(handle)


def run(workers, deals):
    with tempfile.TemporaryDirectory(prefix="bridgemind-benchmark-") as folder:
        ready = Path(folder) / "ready.json"
        process = subprocess.Popen([
            sys.executable, "-m", "bridgemind.dds.server", "--port", "0", "--workers", str(workers),
            "--cache-mb", "0", "--tt-default-mb", "32", "--tt-max-mb", "64", "--ready-file", str(ready)],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        endpoint = None
        try:
            deadline = time.monotonic() + 30
            while not ready.exists():
                if process.poll() is not None:
                    raise RuntimeError(process.communicate()[1])
                if time.monotonic() >= deadline:
                    raise TimeoutError("Service did not become ready")
                time.sleep(0.025)
            endpoint = json.loads(ready.read_text())
            with SolverClient(port=endpoint["port"]) as client:
                start = time.perf_counter()
                items = client.solve_tables(deals, {"wait_timeout_ms": 120000})
                elapsed = time.perf_counter() - start
                if any(item["status"] != "succeeded" for item in items):
                    raise RuntimeError("Benchmark solve failed")
                results = [item["result"]["tricks"] for item in items]
                metrics = {"workers": workers, "boards": len(deals), "seconds": round(elapsed, 4),
                           "boards_per_second": round(len(deals) / elapsed, 2),
                           "memory": process_memory(endpoint["process_id"]), "stats": client.get_stats()}
                return metrics, results
        finally:
            if endpoint:
                try:
                    os.kill(endpoint["process_id"], signal.SIGTERM)
                except OSError:
                    pass
            if process.poll() is None:
                process.terminate()
            process.communicate(timeout=10)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--boards", type=int, default=32)
    parser.add_argument("--workers", type=int, nargs="+", default=[1, 4])
    parser.add_argument("--output", default=".build/validation/benchmark.json")
    args = parser.parse_args()
    if not 1 <= args.boards <= 256:
        parser.error("--boards must be in 1..256")
    rng = random.Random(20261025)
    deals = []
    for _ in range(args.boards):
        cards = list(CARDS)
        rng.shuffle(cards)
        deals.append(Deal({seat: cards[i*13:(i+1)*13] for i, seat in enumerate(SEATS)}))
    expected, report = None, []
    for workers in args.workers:
        metrics, results = run(workers, deals)
        if expected is not None and results != expected:
            raise RuntimeError("Serial and parallel results differ")
        expected = results
        report.append(metrics)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
