import pytest

from bridgemind.dds import NativeDdsBackend, parse_pbn


# DDS v3.1.0 examples/hands.cpp, Apache-2.0, kept in third_party/dds/LICENSE.
PBN = "N:QJ6.K652.J85.T98 873.J97.AT764.Q4 K5.T83.KQ9.A7652 AT942.AQ4.32.KJ3"


@pytest.fixture
def deal():
    return parse_pbn(PBN)


@pytest.fixture
def backend():
    with NativeDdsBackend(tt_default_mb=32, tt_max_mb=64) as backend:
        yield backend

