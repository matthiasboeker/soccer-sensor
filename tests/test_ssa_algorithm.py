import numpy as np  # type: ignore
from singular_spectrum_analysis.ssa_algorithm import (
    get_trajectory_matrix,
    diagonal_averaging,
)


test_ts = np.array([1, 2, 3, 4, 5, 5, 4])


def test_transformation():
    assert [1.0, 2.0, 3.0, 4., 5.0, 5.0, 4.] == diagonal_averaging(
        get_trajectory_matrix(test_ts, lag=2)
    ).tolist()
