"""Length-prefixed, bounded JSON frames over a persistent socket."""

import json
import struct

from .errors import SolverError

MAX_FRAME_BYTES = 8 * 1024 * 1024


def _read_exact(sock, length):
    data = bytearray()
    while len(data) < length:
        chunk = sock.recv(length - len(data))
        if not chunk:
            raise EOFError("Solver connection closed")
        data.extend(chunk)
    return bytes(data)


def recv_message(sock):
    length = struct.unpack("!I", _read_exact(sock, 4))[0]
    if not 0 < length <= MAX_FRAME_BYTES:
        raise SolverError("INVALID_ARGUMENT", "Frame exceeds the allowed size")
    try:
        def reject_constant(value):
            raise ValueError(f"Invalid JSON constant {value}")
        message = json.loads(_read_exact(sock, length).decode("utf-8"), parse_constant=reject_constant)
    except (UnicodeError, ValueError, RecursionError) as error:
        raise SolverError("INVALID_ARGUMENT", "Invalid JSON frame") from error
    if not isinstance(message, dict):
        raise SolverError("INVALID_ARGUMENT", "Message must be an object")
    return message


def send_message(sock, message):
    data = json.dumps(message, separators=(",", ":"), ensure_ascii=True, allow_nan=False).encode("utf-8")
    if not 0 < len(data) <= MAX_FRAME_BYTES:
        raise SolverError("INVALID_ARGUMENT", "Frame exceeds the allowed size")
    sock.sendall(struct.pack("!I", len(data)) + data)

