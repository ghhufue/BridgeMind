"""New random deals sent to the independent DDS service, never to the actor."""

import numpy as np

from bridgemind.dds import Options, SolverError
from .data import deal_from_ownership


def generate_tables(solver, count, *, seed=42, batch_size=32, progress=None):
    if count < 1 or batch_size < 1:
        raise ValueError("Positive board count and batch size required")
    rng = np.random.default_rng(seed)
    owners = np.stack([rng.permutation(np.repeat(np.arange(4, dtype=np.uint8), 13)) for _ in range(count)])
    tables = np.empty((count, 5, 4), dtype=np.uint8)
    for start in range(0, count, batch_size):
        deals = [deal_from_ownership(row) for row in owners[start:start+batch_size]]
        results = solver.solve_tables(deals, Options(wait_timeout_ms=120000, priority="offline"))
        if len(results) != len(deals):
            raise ValueError("Incomplete DDS batch response")
        for index, item in enumerate(results, start):
            if item["status"] != "succeeded":
                raise SolverError.from_dict(item["error"])
            tables[index] = item["result"]["tricks"]
        if progress:
            progress(min(start + batch_size, count), count)
    return owners, tables
