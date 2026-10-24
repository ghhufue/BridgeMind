"""Public, immutable solver data independent of DDS/OpenSpiel numbering."""

from collections.abc import Mapping
from dataclasses import asdict, dataclass

from .errors import SolverError

SEATS = ("N", "E", "S", "W")
STRAINS = ("C", "D", "H", "S", "NT")
RANKS = "23456789TJQKA"
CARDS = tuple(suit + rank for suit in STRAINS[:4] for rank in RANKS)
CARD_ORDER = {card: i for i, card in enumerate(CARDS)}


def _invalid(message, field=None, code="INVALID_ARGUMENT"):
    raise SolverError(code, message, field)


def _object(value, keys):
    if not isinstance(value, Mapping) or set(value) != set(keys):
        _invalid(f"Expected exactly these fields: {', '.join(keys)}")


def _cards(value, field):
    if not isinstance(value, (list, tuple)):
        _invalid("Cards must be an array", field)
    if any(not isinstance(c, str) or c not in CARD_ORDER for c in value):
        _invalid("Invalid card; expected e.g. SA, HT, C2", field)
    return tuple(sorted(value, key=CARD_ORDER.__getitem__))


def _hands(value):
    if not isinstance(value, Mapping) or set(value) != set(SEATS):
        _invalid("Hands must contain exactly N, E, S, W", "hands")
    normalized = tuple(_cards(value[seat], f"hands.{seat}") for seat in SEATS)
    flat = [card for hand in normalized for card in hand]
    if len(set(flat)) != len(flat):
        _invalid("Duplicate card", "hands")
    return normalized


@dataclass(frozen=True, init=False)
class Deal:
    _hands: tuple

    def __init__(self, hands):
        normalized = _hands(hands)
        if any(len(hand) != 13 for hand in normalized):
            _invalid("Each initial hand must contain 13 cards", "hands")
        object.__setattr__(self, "_hands", normalized)

    @property
    def hands(self):
        return dict(zip(SEATS, self._hands))

    def to_dict(self):
        return {"hands": {s: list(h) for s, h in zip(SEATS, self._hands)}}

    @classmethod
    def from_dict(cls, value):
        if isinstance(value, cls):
            return value
        _object(value, ("hands",))
        return cls(value["hands"])


@dataclass(frozen=True, init=False)
class Position:
    _hands: tuple
    trump: str
    trick_leader: str
    current_trick: tuple

    def __init__(self, remaining_hands, trump, trick_leader, current_trick=()):
        hands = _hands(remaining_hands)
        if trump not in STRAINS or trick_leader not in SEATS:
            _invalid("Invalid trump or trick_leader")
        _cards(current_trick, "current_trick")
        trick = tuple(current_trick)
        if len(trick) > 3:
            _invalid("Current trick must contain 0..3 cards", "current_trick")
        flat = [c for hand in hands for c in hand] + list(trick)
        if len(flat) != len(set(flat)):
            _invalid("Duplicate card in position", "current_trick")
        total = len(flat)
        if total == 0 or total > 52 or total % 4:
            _invalid("Position card count must represent 1..13 tricks", code="INVALID_POSITION")
        n = total // 4
        leader = SEATS.index(trick_leader)
        played = {(leader + i) % 4 for i in range(len(trick))}
        if any(len(hand) != n - (i in played) for i, hand in enumerate(hands)):
            _invalid("Hand count inconsistent with current trick", code="INVALID_POSITION")
        for offset, card in enumerate(trick[1:], 1):
            hand = hands[(leader + offset) % 4]
            if card[0] != trick[0][0] and any(c[0] == trick[0][0] for c in hand):
                _invalid("Current trick contains a failure to follow suit", code="INVALID_POSITION")
        for name, value in (("_hands", hands), ("trump", trump),
                            ("trick_leader", trick_leader), ("current_trick", trick)):
            object.__setattr__(self, name, value)

    @property
    def remaining_hands(self):
        return dict(zip(SEATS, self._hands))

    @property
    def to_play(self):
        return SEATS[(SEATS.index(self.trick_leader) + len(self.current_trick)) % 4]

    @property
    def remaining_tricks(self):
        return (sum(map(len, self._hands)) + len(self.current_trick)) // 4

    def legal_cards(self):
        hand = self.remaining_hands[self.to_play]
        follow = tuple(c for c in hand if self.current_trick and c[0] == self.current_trick[0][0])
        return follow or hand

    def to_dict(self):
        return {"remaining_hands": {s: list(h) for s, h in zip(SEATS, self._hands)},
                "trump": self.trump, "trick_leader": self.trick_leader,
                "current_trick": list(self.current_trick)}

    @classmethod
    def from_dict(cls, value):
        if isinstance(value, cls):
            return value
        if not isinstance(value, Mapping):
            _invalid("Position must be an object")
        _object(value, ("remaining_hands", "trump", "trick_leader", "current_trick")
                if "current_trick" in value else ("remaining_hands", "trump", "trick_leader"))
        return cls(**value)


@dataclass(frozen=True)
class TableResult:
    tricks: tuple
    strains: tuple = STRAINS
    declarers: tuple = SEATS
    exact: bool = True
    evaluation_model: str = "double_dummy"

    def to_dict(self):
        return asdict(self)

    @classmethod
    def from_dict(cls, value):
        return cls(tuple(tuple(row) for row in value["tricks"]))


@dataclass(frozen=True)
class Move:
    card: str
    tricks_for_evaluated_side: int


@dataclass(frozen=True)
class PositionResult:
    to_play: str
    evaluated_side: str
    remaining_tricks: int
    best_tricks: int
    best_cards: tuple
    moves: tuple
    search_nodes: int | None = None
    exact: bool = True
    evaluation_model: str = "double_dummy"

    def to_dict(self):
        return asdict(self)

    @classmethod
    def from_dict(cls, value):
        return cls(value["to_play"], value["evaluated_side"], value["remaining_tricks"],
                   value["best_tricks"], tuple(value["best_cards"]),
                   tuple(Move(**m) for m in value["moves"]), value.get("search_nodes"))


def parse_pbn(value):
    if not isinstance(value, str):
        _invalid("PBN must be a string", "pbn")
    tokens = value.split()
    if len(tokens) != 4 or len(tokens[0]) < 2 or tokens[0][1] != ":" or tokens[0][0] not in SEATS:
        _invalid("Expected PBN initial seat and four hands", "pbn")
    first = SEATS.index(tokens[0][0])
    tokens[0] = tokens[0][2:]
    hands = {}
    for offset, token in enumerate(tokens):
        suits = token.split(".")
        if len(suits) != 4:
            _invalid("PBN hands require four suit segments", "pbn")
        cards = [suit + rank for suit, ranks in zip("SHDC", suits) for rank in ranks if rank != "-"]
        if any("-" in ranks and ranks != "-" for ranks in suits):
            _invalid("Invalid PBN void", "pbn")
        hands[SEATS[(first + offset) % 4]] = cards
    return Deal(hands)


def format_pbn(deal, first_seat="N"):
    deal = Deal.from_dict(deal)
    if first_seat not in SEATS:
        _invalid("Invalid first_seat")
    first = SEATS.index(first_seat)
    tokens = []
    for offset in range(4):
        hand = deal.hands[SEATS[(first + offset) % 4]]
        tokens.append(".".join("".join(r for r in reversed(RANKS) if s + r in hand) for s in "SHDC"))
    return first_seat + ":" + " ".join(tokens)

