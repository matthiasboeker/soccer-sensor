from typing import Dict

import numpy as np   # type: ignore
import pandas as pd  # type: ignore


class ContainsNAN(Exception):
    """The trajectory matrix contains NAN values."""


class NotReversible(Exception):
    """The sum of the elementary matrices are not equal to trajectory matrix"""


def contains_nan(array: np.ndarray):
    return np.isnan(np.sum(array))


def get_trajectory_matrix(time_series: np.ndarray, lag: int) -> np.ndarray:
    if isinstance(time_series, pd.Series):
        time_series = time_series.to_numpy()
    k = len(time_series) - lag + 1  # The number of columns in the trajectory matrix.
    return np.column_stack([time_series[i:i + lag] for i in range(0, k)])


def diagonal_averaging(matrix: np.ndarray) -> np.ndarray:
    """Averages the anti-diagonals of the given elementary matrix, X_i, and returns a time series."""
    reversed_matrix = matrix[::-1]
    return np.array([reversed_matrix.diagonal(i).mean() for i in range(-matrix.shape[0]+1, matrix.shape[1])])


def apply_svd(trajectory: np.ndarray) -> Dict[str, np.ndarray]:
    if contains_nan(trajectory):
        raise ContainsNAN
    u, s, v = np.linalg.svd(trajectory)
    return {"U": u, "Sigma": s, "V": v}


def calculate_elementary_matrices(svd: Dict[str, np.ndarray], rank, trajectory_matrix: np.ndarray) -> np.ndarray:
    v_transpose = svd["V"].T
    elementary_matrices = np.array([svd["Sigma"][i] * np.outer(svd["U"][:, i], v_transpose[:, i]) for i in range(0, rank)])
    if not np.allclose(trajectory_matrix, elementary_matrices.sum(axis=0), atol=1e-10):
        raise NotReversible
    return elementary_matrices


class SSA:
    def __init__(self, trajectory_matrix, svd: Dict[str, np.ndarray], elementary_matrices: np.ndarray):
        self.trajectory_matrix = trajectory_matrix
        self.svd = svd,
        self.elementary_matrices = elementary_matrices

    @classmethod
    def transform(cls, time_series: np.ndarray, lag: int):
        trajectory_matrix = get_trajectory_matrix(time_series, lag)
        rank = np.linalg.matrix_rank(trajectory_matrix)
        svd = apply_svd(trajectory_matrix)
        elementary_matrices = calculate_elementary_matrices(svd, rank, trajectory_matrix)
        return cls(trajectory_matrix, svd, elementary_matrices)

    def get_relative_contributions(self) -> np.ndarray:
        return self.svd["Sigma"] ** 2 / (self.svd["Sigma"] ** 2).sum()
