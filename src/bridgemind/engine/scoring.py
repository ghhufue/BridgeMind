"""Duplicate bridge raw scoring, independently checked against OpenSpiel.

Rules reference: third_party/open_spiel/open_spiel/games/bridge/bridge_scoring.cc
(Apache-2.0, DeepMind Technologies). This is separate from the DDS service.
"""

from dataclasses import asdict, dataclass

from bridgemind.dds.models import SEATS, STRAINS


@dataclass(frozen=True)
class Contract:
    level: int
    strain: str
    declarer: str
    double_status: int = 0

    def __post_init__(self):
        if type(self.level) is not int or not 1 <= self.level <= 7 or self.strain not in STRAINS or self.declarer not in SEATS:
            raise ValueError("Invalid contract")
        if type(self.double_status) is not int or self.double_status not in (0, 1, 2):
            raise ValueError("Invalid double_status")

    def to_dict(self):
        return asdict(self)


def score_contract(contract, declarer_tricks, vulnerable):
    if type(declarer_tricks) is not int or not 0 <= declarer_tricks <= 13 or type(vulnerable) is not bool:
        raise ValueError("Invalid tricks or vulnerability")
    difference = declarer_tricks - 6 - contract.level
    doubled = contract.double_status
    if difference < 0:
        down = -difference
        if not doubled:
            return -down * (100 if vulnerable else 50)
        penalty = 200 + 300 * (down - 1) if vulnerable else 100 + 200 * min(down - 1, 2) + 300 * max(down - 3, 0)
        return -penalty * (2 if doubled == 2 else 1)
    unit = 20 if contract.strain in ("C", "D") else 30
    points = (contract.level * unit + (10 if contract.strain == "NT" else 0)) * (1 << doubled)
    bonus = (500 if vulnerable else 300) if points >= 100 else 50
    if contract.level == 6:
        bonus += 750 if vulnerable else 500
    elif contract.level == 7:
        bonus += 1500 if vulnerable else 1000
    if doubled:
        bonus += 50 * (1 << (doubled - 1))
        overtricks = difference * (200 if vulnerable else 100) * (1 << (doubled - 1))
    else:
        overtricks = difference * unit
    return points + bonus + overtricks

