import socket
import os
import signal
import struct
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor

import pytest

from bridgemind.dds import SolverClient, SolverError
from bridgemind.dds.protocol import recv_message


def test_fragmented_frames_and_oversized_frame():
    a, b = socket.socketpair()
    try:
        payload = b'{"ok":true}'
        frame = struct.pack('!I', len(payload)) + payload
        with ThreadPoolExecutor(1) as pool:
            reader = pool.submit(recv_message, b)
            for byte in frame:
                a.sendall(bytes([byte]))
            assert reader.result(2) == {"ok": True}
        a.sendall(struct.pack('!I', 8 * 1024 * 1024 + 1))
        with pytest.raises(SolverError):
            recv_message(b)
    finally:
        a.close()
        b.close()


def test_real_subprocess_client_batches_and_parallel_calls(deal, tmp_path):
    ready = tmp_path / "ready.json"
    endpoint = None
    process = subprocess.Popen([sys.executable, "-m", "bridgemind.dds.server", "--port", "0", "--workers", "2", "--ready-file", str(ready)], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    try:
        import json
        import time
        for _ in range(200):
            if ready.exists():
                break
            if process.poll() is not None:
                pytest.fail(process.communicate()[1])
            time.sleep(0.025)
        assert ready.exists()
        endpoint = json.loads(ready.read_text())
        assert endpoint["process_id"] > 0
        with SolverClient(port=endpoint["port"]) as client:
            assert client.get_info()["solver_version"] == "3.1.0"
            with ThreadPoolExecutor(4) as pool:
                results = list(pool.map(lambda _: client.solve_table(deal), range(8)))
            assert all(result == results[0] for result in results)
            batch = client.solve_tables([deal, {"hands": {}}])
            assert [x["status"] for x in batch] == ["succeeded", "failed"]
            position = {"remaining_hands": {"N": [], "E": ["SK"], "S": ["SQ"], "W": ["SJ"]}, "trump": "NT", "trick_leader": "N", "current_trick": ["SA"]}
            assert client.solve_position(position).best_tricks == 0
            assert client.get_stats()["cache_hits"] > 0
            import importlib.util
            if importlib.util.find_spec("pyspiel") is not None:
                from bridgemind.engine import BridgeBiddingEnv
                env = BridgeBiddingEnv(client, deal)
                for call in ("1NT", "P", "P", "P"):
                    env.apply_call(call)
                assert env.result()["returns"] == [-50, 50, -50, 50]
    finally:
        # A Windows venv redirector can have a different PID from Python itself.
        if endpoint:
            try:
                os.kill(endpoint["process_id"], signal.SIGTERM)
            except OSError:
                pass
        if process.poll() is None:
            process.terminate()
        process.communicate(timeout=10)


def test_wait_timeout_returns_recoverable_job_and_disconnect(deal):
    import threading
    from bridgemind.dds import Options, ServiceConfig, SolverService, SolverWaitTimeout
    from bridgemind.dds.server import SolverServer
    from test_service import GateBackend

    started, release, calls = threading.Event(), threading.Event(), []
    with SolverService(ServiceConfig(worker_count=1), backend_factory=lambda: GateBackend(started, release, calls)) as service:
        with SolverServer(service, 0) as server:
            thread = threading.Thread(target=server.serve_forever, daemon=True)
            thread.start()
            try:
                with SolverClient(port=server.server_address[1]) as client:
                    with pytest.raises(SolverWaitTimeout) as error:
                        client.solve_table(deal, Options(wait_timeout_ms=1))
                    release.set()
                    assert client.wait(error.value.job_id, 2000)["complete"]
                    client.close()
                    with pytest.raises(SolverError) as closed:
                        client.get_info()
                    assert closed.value.code == "SERVICE_UNAVAILABLE"
            finally:
                release.set()
                server.shutdown()
                thread.join(timeout=2)


@pytest.mark.parametrize("payload", [b'[]', b'{"x":NaN}', b'\xff', b'{'])
def test_invalid_json_frames_rejected(payload):
    a, b = socket.socketpair()
    try:
        a.sendall(struct.pack('!I', len(payload)) + payload)
        with pytest.raises(SolverError):
            recv_message(b)
    finally:
        a.close()
        b.close()

