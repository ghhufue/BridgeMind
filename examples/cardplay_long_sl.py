"""Detached, time-bounded SL worker. Status and checkpoints stay in its run directory."""

import argparse
from datetime import datetime, timedelta, timezone
import json
import os
from pathlib import Path
import time
import traceback

from bridgemind.cardplay.__main__ import parser as training_parser
from bridgemind.cardplay.model import read_checkpoint
from bridgemind.cardplay.runner import train_supervised
from bridgemind.training.runner import provenance, sha256

ROOT = Path(__file__).resolve().parents[1]


def process_running(pid):
    if os.name == "nt":
        import ctypes
        from ctypes import wintypes
        kernel = ctypes.WinDLL("kernel32", use_last_error=True)
        kernel.OpenProcess.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
        kernel.OpenProcess.restype = wintypes.HANDLE
        kernel.GetExitCodeProcess.argtypes = [wintypes.HANDLE, ctypes.POINTER(wintypes.DWORD)]
        kernel.CloseHandle.argtypes = [wintypes.HANDLE]
        handle = kernel.OpenProcess(0x1000, False, pid)
        if not handle:
            return ctypes.get_last_error() != 87  # Access denied is conservatively treated as alive.
        try:
            code = wintypes.DWORD()
            return not kernel.GetExitCodeProcess(handle, ctypes.byref(code)) or code.value == 259
        finally:
            kernel.CloseHandle(handle)
    try:
        os.kill(pid, 0)
        return True
    except ProcessLookupError:
        return False


def write_status(path, record):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(record, indent=2, allow_nan=False), encoding="utf-8")
    temporary.replace(path)


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--run", required=True)
    cli.add_argument("--seconds", type=int, default=1200)
    cli.add_argument("--resume", action="store_true")
    args = cli.parse_args()
    if args.seconds < 1:
        cli.error("--seconds must be positive")
    directory = Path(args.run).resolve()
    directory.mkdir(parents=True, exist_ok=True)
    status_path = directory/"status.json"
    if status_path.exists():
        previous = json.loads(status_path.read_text())
        if not args.resume:
            raise ValueError("Run exists; use --resume or a fresh directory")
        if previous["state"] == "running" and process_running(previous["process_id"]):
            raise ValueError("The recorded training worker is still running")
    started = datetime.now(timezone.utc)
    initial = ROOT/"runs/cardplay-expanded-v1/sl-42/best.pt"
    options = ["train-sl", "--run", str(directory), "--train", str(ROOT/"data/cardplay-expanded-v1/train"),
               "--valid", str(ROOT/"data/cardplay-expanded-v1/valid"), "--steps", "1000000",
               "--max-seconds", str(args.seconds), "--batch-size", "64", "--eval-every", "1000",
               "--eval-states", "8192", "--seed", "42", "--lr", "0.00003", "--device", "cuda"]
    options += ["--resume"] if args.resume else ["--init", str(initial)]
    status = {"state": "running", "phase": "SL training", "process_id": os.getpid(),
              "started_at_utc": started.isoformat(),
              "expected_finish_utc": (started+timedelta(seconds=args.seconds)).isoformat(),
              "time_budget_seconds": args.seconds, "training_arguments": options,
              "initial_model": str(initial), "initial_model_sha256": sha256(initial),
              "provenance": provenance(), "test_set_used": False}
    write_status(status_path, status)
    tick = time.monotonic()
    try:
        train_supervised(training_parser().parse_args(options))
        last = read_checkpoint(directory/"last.pt")
        best = read_checkpoint(directory/"best.pt")
        events = [json.loads(line) for line in (directory/"metrics.jsonl").read_text().splitlines()]
        status.update(state="completed", phase="saved", elapsed_seconds=time.monotonic()-tick,
                      finished_at_utc=datetime.now(timezone.utc).isoformat(), last_step=last["metadata"]["step"],
                      best_step=best["metadata"]["step"], best_validation=best["metadata"]["validation"],
                      last_validation=next(row["validation"] for row in reversed(events) if "validation" in row),
                      last_model_sha256=sha256(directory/"last.pt"), best_model_sha256=sha256(directory/"best.pt"))
        write_status(directory/"report.json", status)
        write_status(status_path, status)
    except BaseException as error:
        status.update(state="failed", elapsed_seconds=time.monotonic()-tick,
                      finished_at_utc=datetime.now(timezone.utc).isoformat(), error=f"{type(error).__name__}: {error}",
                      traceback=traceback.format_exc())
        write_status(status_path, status)
        raise


if __name__ == "__main__":
    os.chdir(ROOT)
    main()
