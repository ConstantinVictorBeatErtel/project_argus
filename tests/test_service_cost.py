"""Latency service-cost construction."""

from __future__ import annotations

import numpy as np
from scipy.sparse import csr_matrix

from src.optimization.milp import propagation_latency_cost


def test_propagation_latency_cost_divides_range_by_signal_speed() -> None:
    ranges = csr_matrix(np.array([[300.0, 0.0], [0.0, 600.0]], dtype=float))
    costs = propagation_latency_cost(ranges, speed_of_light_km_s=300.0)

    assert costs.toarray().tolist() == [[1.0, 0.0], [0.0, 2.0]]
