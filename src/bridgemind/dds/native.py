"""ctypes binding to DDS's context-scoped C ABI; no global thread pool calls."""

import ctypes as ct
import hashlib
import json
import os
from pathlib import Path
import threading

from .errors import SolverError
from .models import CARD_ORDER, RANKS, SEATS, STRAINS, Deal, Move, Position, PositionResult, TableResult

DDS_SUITS = "SHDC"
DDS_STRAINS = ("S", "H", "D", "C", "NT")
_libraries = {}
_library_lock = threading.Lock()


class _Deal(ct.Structure):
    _fields_ = [("trump", ct.c_int), ("first", ct.c_int),
                ("current_suits", ct.c_int * 3), ("current_ranks", ct.c_int * 3),
                ("cards", (ct.c_uint * 4) * 4)]


class _TableDeal(ct.Structure):
    _fields_ = [("cards", (ct.c_uint * 4) * 4)]


class _Table(ct.Structure):
    _fields_ = [("tricks", (ct.c_int * 4) * 5)]


class _Future(ct.Structure):
    _fields_ = [("nodes", ct.c_int), ("cards", ct.c_int)] + [
        (key, ct.c_int * 13) for key in ("suit", "rank", "equals", "score")]


class _Info(ct.Structure):
    _fields_ = [(key, ct.c_int) for key in ("major", "minor", "patch")] + [
        ("version_string", ct.c_char * 10)] + [
        (key, ct.c_int) for key in ("system", "bits", "compiler", "constructor", "cores", "threading", "threads")
    ] + [("thread_sizes", ct.c_char * 128), ("system_string", ct.c_char * 1024)]


def default_library_path():
    configured = os.environ.get("BRIDGEMIND_DDS_LIBRARY")
    if configured:
        return Path(configured).resolve()
    return Path(__file__).resolve().parents[3] / "artifacts/dds/windows-x64/dds.dll"


def _load(path):
    path = Path(path).resolve()
    with _library_lock:
        if str(path) in _libraries:
            return _libraries[str(path)]
        try:
            dll = ct.CDLL(str(path))
            signatures = {
                "dds_c_create_solvercontext": ([ct.c_int, ct.c_int, ct.c_int], ct.c_void_p),
                "dds_c_destroy_solvercontext": ([ct.c_void_p], None),
                "dds_c_calc_dd_table": ([ct.c_void_p, ct.POINTER(_TableDeal), ct.POINTER(_Table)], ct.c_int),
                "dds_c_solve_board": ([ct.c_void_p, ct.POINTER(_Deal), ct.c_int, ct.c_int, ct.c_int, ct.POINTER(_Future)], ct.c_int),
                "GetDDSInfo": ([ct.POINTER(_Info)], None),
                "ErrorMessage": ([ct.c_int, ct.c_char_p], None),
            }
            for name, (args, result) in signatures.items():
                function = getattr(dll, name)
                function.argtypes, function.restype = args, result
            info = _Info()
            dll.GetDDSInfo(ct.byref(info))
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            commit = None
            manifest_path = path.parent / "manifest.json"
            if manifest_path.exists():
                manifest = json.loads(manifest_path.read_text(encoding="utf-8-sig"))
                if any(f["name"] == path.name and f["sha256"] == digest for f in manifest.get("files", [])):
                    commit = manifest.get("commit")
            metadata = {"version": info.version_string.decode("ascii"), "commit": commit,
                        "identity": digest, "bits": info.bits, "library_path": str(path)}
        except (OSError, AttributeError, ValueError, KeyError) as error:
            raise SolverError("SERVICE_UNAVAILABLE", f"Cannot load DDS library {path}: {error}") from error
        _libraries[str(path)] = (dll, metadata)
        return dll, metadata


def _fill_cards(target, hands):
    for index, seat in enumerate(SEATS):
        for card in hands[seat]:
            target[index][DDS_SUITS.index(card[0])] |= 1 << (RANKS.index(card[1]) + 2)


class NativeDdsBackend:
    def __init__(self, library_path=None, tt_default_mb=64, tt_max_mb=128):
        for name in ("DDS_TT_DEFAULT_MB", "DDS_TT_LIMIT_MB"):
            if os.environ.get(name):
                raise SolverError("INVALID_ARGUMENT", f"Unset {name}; use managed context memory options")
        # DDS 3.1.0's Large TT can fault below its allocation granularity.
        # Keep this wrapper on the verified >=32 MiB configuration range.
        if type(tt_default_mb) is not int or type(tt_max_mb) is not int or not 32 <= tt_default_mb <= tt_max_mb <= 4096:
            raise SolverError("INVALID_ARGUMENT", "Invalid transposition table memory configuration")
        self._dll, self.info = _load(library_path or default_library_path())
        self._owner = threading.get_ident()
        self._ctx = self._dll.dds_c_create_solvercontext(1, tt_default_mb, tt_max_mb)
        if not self._ctx:
            raise SolverError("SERVICE_UNAVAILABLE", "DDS context allocation failed")

    def _check_thread(self):
        if self._owner != threading.get_ident():
            raise SolverError("INVALID_ARGUMENT", "DDS context must stay on its owning thread")
        if not self._ctx:
            raise SolverError("SERVICE_UNAVAILABLE", "DDS context is closed")

    def _check_code(self, code):
        if code != 1:
            message = ct.create_string_buffer(256)
            self._dll.ErrorMessage(code, message)
            raise SolverError("NATIVE_SOLVER_ERROR", message.value.decode("utf-8", errors="replace"), native_code=code)

    def solve_table(self, deal):
        self._check_thread()
        deal = Deal.from_dict(deal)
        native, result = _TableDeal(), _Table()
        _fill_cards(native.cards, deal.hands)
        self._check_code(self._dll.dds_c_calc_dd_table(self._ctx, ct.byref(native), ct.byref(result)))
        values = tuple(tuple(result.tricks[DDS_STRAINS.index(s)][seat] for seat in range(4)) for s in STRAINS)
        if any(not 0 <= value <= 13 for row in values for value in row):
            raise SolverError("NATIVE_SOLVER_ERROR", "DDS returned an invalid trick table")
        return TableResult(values)

    def solve_position(self, position):
        self._check_thread()
        position = Position.from_dict(position)
        native, result = _Deal(), _Future()
        _fill_cards(native.cards, position.remaining_hands)
        native.trump = DDS_STRAINS.index(position.trump)
        native.first = SEATS.index(position.trick_leader)
        for index, card in enumerate(position.current_trick):
            native.current_suits[index] = DDS_SUITS.index(card[0])
            native.current_ranks[index] = RANKS.index(card[1]) + 2
        self._check_code(self._dll.dds_c_solve_board(self._ctx, ct.byref(native), -1, 3, 1, ct.byref(result)))
        values = {}
        if not 1 <= result.cards <= 13:
            raise SolverError("NATIVE_SOLVER_ERROR", "Invalid DDS candidate count")
        for index in range(result.cards):
            if not 0 <= result.suit[index] < 4 or not 2 <= result.rank[index] <= 14 or not 0 <= result.score[index] <= position.remaining_tricks:
                raise SolverError("NATIVE_SOLVER_ERROR", "Invalid DDS candidate value")
            ranks = {result.rank[index]} | {r for r in range(2, 15) if result.equals[index] & (1 << r)}
            for rank in ranks:
                values[DDS_SUITS[result.suit[index]] + RANKS[rank - 2]] = result.score[index]
        if set(values) != set(position.legal_cards()):
            raise SolverError("NATIVE_SOLVER_ERROR", "DDS candidates do not match legal cards")
        moves = tuple(Move(card, score) for card, score in sorted(values.items(), key=lambda pair: (-pair[1], CARD_ORDER[pair[0]])))
        best = moves[0].tricks_for_evaluated_side
        return PositionResult(position.to_play, "NS" if SEATS.index(position.to_play) % 2 == 0 else "EW",
                              position.remaining_tricks, best,
                              tuple(move.card for move in moves if move.tricks_for_evaluated_side == best),
                              moves, result.nodes)

    def close(self):
        if self._ctx:
            self._check_thread()
            self._dll.dds_c_destroy_solvercontext(self._ctx)
            self._ctx = None

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()

