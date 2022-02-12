from typing import Dict, List

import numpy as np   # type: ignore
import pandas as pd  # type: ignore
import matplotlib.pyplot as plt

class ContainsNAN(Exception):
    """The trajectory matrix contains NAN values."""


class NotReversible(Exception):
    """The sum of the elementary matrices are not equal to trajectory matrix"""

def get_missing_matrix(matrix):
    return 1 * np.isnan(matrix)

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


def reconstruct(matrices: List[np.ndarray]) -> np.ndarray:
    return sum(diagonal_averaging(matrix) for matrix in matrices)


def apply_svd(trajectory: np.ndarray) -> Dict[str, np.ndarray]:
    if contains_nan(trajectory):
        raise ContainsNAN
    u, s, v = np.linalg.svd(trajectory)
    return {"U": u, "Sigma": s, "V": v}


def calculate_elementary_matrices(svd: Dict[str, np.ndarray], trajectory_matrix: np.ndarray) -> np.ndarray:
    v_transpose = svd["V"].T
    rank = np.linalg.matrix_rank(trajectory_matrix)
    elementary_matrices = np.array([svd["Sigma"][i] * np.outer(svd["U"][:, i], v_transpose[:, i]) for i in range(0, rank)])
    if not np.allclose(trajectory_matrix, elementary_matrices.sum(axis=0), atol=1e-10):
        raise NotReversible
    return elementary_matrices


def fit_matrix(svd, trajectory_matrix, q):
    elementary_matrices = calculate_elementary_matrices(svd, trajectory_matrix)
    if not np.allclose(trajectory_matrix, elementary_matrices.sum(axis=0), atol=1e-10):
        raise NotReversible
    return sum(elementary_matrices[i] for i in range(0, q))


def impute(missing_matrix, trajectory, fitted_matrix):
    return np.nan_to_num(trajectory) + np.multiply(missing_matrix, fitted_matrix)


def get_norm_dist_matrix(time_series):
    """Extent for different distributions"""
    mean = np.nanmean(time_series)
    var = np.nanvar(time_series)
    return np.array([np.random.normal(mean, var, 1)[0] if np.isnan(point) else point for point in time_series])


def square_distance_eigenvalues(old_matrix, new_matrix):
    difference = np.square(old_matrix - new_matrix)
    return np.sum(np.sum(difference, axis=1))


def em_iterations(trajectory, missing_matrix, order, lag, threshold, total_iterations):
    loss = []
    i = 1
    diff = 100
    X = maximise(trajectory, lag)
    old_matrix = X
    while i < total_iterations or diff > threshold:
        if np.mod(i, 50) == 0:
            print(f"Iteration{i}")
        X, fitted_matrix = expectation_step(X, trajectory, missing_matrix, order)
        diff = square_distance_eigenvalues(old_matrix, fitted_matrix)
        loss.append(diff)
        old_matrix = fitted_matrix
        X = maximise(X, lag)
        i = i+1

    return X, loss


def expectation_step(matrix_iteration, trajectory, missing_matrix, order_q: int):
    svd = apply_svd(matrix_iteration)
    fitted_matrix = fit_matrix(svd, matrix_iteration, order_q)
    return impute(missing_matrix, trajectory, fitted_matrix), fitted_matrix


def maximise(iteration_matrix: np.ndarray, lag:int):
    iteration_ts = diagonal_averaging(iteration_matrix)
    return get_trajectory_matrix(get_norm_dist_matrix(iteration_ts), lag)


class SSA:
    def __init__(self, trajectory_matrix, loss):
        self.trajectory_matrix = trajectory_matrix
        self.loss = loss

    @classmethod
    def fit(cls, time_series: np.ndarray, lag: int, order: int, threshold, total_iterations):
        trajectory_matrix = get_trajectory_matrix(time_series, lag)
        missing_matrix = get_missing_matrix(trajectory_matrix)
        fitted_trajectory, loss = em_iterations(trajectory_matrix, missing_matrix,order, lag, threshold, total_iterations)
        return cls(fitted_trajectory, loss)
