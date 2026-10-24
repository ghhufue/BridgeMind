import random
from concurrent.futures import ThreadPoolExecutor

import pytest

from bridgemind.dds import Deal, Position, SolverError, NativeDdsBackend, format_pbn, parse_pbn


def test_table_matches_upstream_and_public_strain_order(backend, deal):
    result = backend.solve_table(deal)
    assert result.tricks == ((7, 5, 7, 5), (5, 7, 5, 7), (6, 6, 6, 6), (5, 8, 5, 8), (6, 6, 6, 6))
    assert result.strains == ("C", "D", "H", "S", "NT")
    assert result.exact
    assert backend.info["version"] == "3.1.0"


def test_pbn_roundtrip_with_rotated_first_seat(deal):
    assert parse_pbn(format_pbn(deal, first_seat="E")) == deal


def test_partial_trick_keeps_side_to_play_perspective(backend):
    position = Position({"N": [], "E": ["SK"], "S": ["SQ"], "W": ["SJ"]}, "NT", "N", ["SA"])
    result = backend.solve_position(position)
    assert (result.to_play, result.evaluated_side, result.remaining_tricks) == ("E", "EW", 1)
    assert result.best_tricks == 0
    assert result.best_cards == ("SK",)
    assert result.moves[0].tricks_for_evaluated_side == 0


def test_equivalent_cards_expanded_and_single_choice_is_exact(backend):
    position = Position({"N": ["SA", "SK"], "E": ["SQ", "SJ"], "S": ["ST", "S9"], "W": ["S8", "S7"]}, "NT", "N")
    result = backend.solve_position(position)
    assert result.best_cards == ("SK", "SA")
    assert len(result.moves) == 2
    assert {m.tricks_for_evaluated_side for m in result.moves} == {2}


@pytest.mark.parametrize("changes", [
    {"N": ["SA"]},
    {"N": ["ZZ"] * 13, "E": [], "S": [], "W": []},
])
def test_invalid_deals_rejected(changes):
    with pytest.raises(SolverError):
        Deal(changes)


def test_invalid_partial_trick_counts_and_revoke_rejected():
    with pytest.raises(SolverError, match="count"):
        Position({"N": ["SK"], "E": ["SQ"], "S": ["SJ"], "W": []}, "NT", "N", ["SA"])
    with pytest.raises(SolverError, match="follow"):
        Position({"N": ["SK"], "E": ["SQ"], "S": ["SJ", "S9"], "W": ["S8", "S7"]}, "NT", "N", ["SA", "H2"])


def test_unsafe_small_large_tt_configuration_rejected():
    with pytest.raises(SolverError):
        NativeDdsBackend(tt_default_mb=4, tt_max_mb=16)


def test_environment_cannot_override_managed_memory(monkeypatch):
    monkeypatch.setenv("DDS_TT_DEFAULT_MB", "4")
    with pytest.raises(SolverError, match="DDS_TT_DEFAULT_MB"):
        NativeDdsBackend()


@pytest.mark.parametrize("pbn, expected", [
    ("E:QJT5432.T.6.QJ82 .J97543.K7532.94 87.A62.QJT4.AT75 AK96.KQ8.A98.K63",
     ((6, 7, 6, 7), (8, 3, 8, 3), (10, 2, 10, 2), (4, 9, 4, 9), (9, 3, 9, 3))),
    ("N:73.QJT.AQ54.T752 QT6.876.KJ9.AQ84 5.A95432.7632.K6 AKJ9842.K.T8.J93",
     ((3, 9, 3, 9), (8, 4, 8, 4), (9, 4, 9, 4), (3, 10, 3, 10), (4, 8, 4, 8))),
])
def test_remaining_upstream_tables(backend, pbn, expected):
    assert backend.solve_table(parse_pbn(pbn)).tricks == expected


def test_context_cannot_be_used_from_another_thread(backend, deal):
    with ThreadPoolExecutor(1) as pool:
        with pytest.raises(SolverError, match="thread"):
            pool.submit(backend.solve_table, deal).result()


def _winner(leader, cards, trump):
    led = cards[0][0]
    rank = "23456789TJQKA"
    def strength(card):
        return (2 if card[0] == trump else 1 if card[0] == led else 0, rank.index(card[1]))
    return (leader + max(range(4), key=lambda i: strength(cards[i]))) % 4


def _legal(hand, trick):
    follow = [card for card in hand if trick and card[0] == trick[0][0]]
    return follow or list(hand)


def _oracle(hands, leader, trick, trump, side):
    if not any(hands):
        return 0
    seat = (leader + len(trick)) % 4
    values = []
    for card in _legal(hands[seat], trick):
        next_hands = [list(h) for h in hands]
        next_hands[seat].remove(card)
        next_trick = trick + [card]
        if len(next_trick) == 4:
            winner = _winner(leader, next_trick, trump)
            value = int(winner % 2 == side) + _oracle(next_hands, winner, [], trump, side)
        else:
            value = _oracle(next_hands, leader, next_trick, trump, side)
        values.append(value)
    return (max if seat % 2 == side else min)(values)


def test_random_endings_against_independent_minimax(backend):
    rng = random.Random(174)
    deck = [s + r for s in "CDHS" for r in "23456789TJQKA"]
    for index in range(24):
        cards = rng.sample(deck, 8)
        hands = [cards[i * 2:i * 2 + 2] for i in range(4)]
        leader = rng.randrange(4)
        trick = []
        for offset in range(index % 4):
            seat = (leader + offset) % 4
            card = rng.choice(_legal(hands[seat], trick))
            hands[seat].remove(card)
            trick.append(card)
        trump = rng.choice(["C", "D", "H", "S", "NT"])
        position = Position(dict(zip("NESW", hands)), trump, "NESW"[leader], trick)
        result = backend.solve_position(position)
        actor = (leader + len(trick)) % 4
        expected = {}
        for card in _legal(hands[actor], trick):
            next_hands = [list(h) for h in hands]
            next_hands[actor].remove(card)
            next_trick = trick + [card]
            if len(next_trick) == 4:
                winner = _winner(leader, next_trick, trump)
                score = int(winner % 2 == actor % 2) + _oracle(next_hands, winner, [], trump, actor % 2)
            else:
                score = _oracle(next_hands, leader, next_trick, trump, actor % 2)
            expected[card] = score
        assert {m.card: m.tricks_for_evaluated_side for m in result.moves} == expected

