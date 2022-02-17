from typing import Any, Dict, List

import numpy as np  # type: ignore
import pandas as pd  # type: ignore


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
    return np.column_stack([time_series[i : i + lag] for i in range(0, k)])


def diagonal_averaging(matrix: np.ndarray) -> np.ndarray:
    """Averages the anti-diagonals of the given elementary matrix, X_i, and returns a time series."""
    reversed_matrix = matrix[::-1]
    return np.array(
        [
            reversed_matrix.diagonal(i).mean()
            for i in range(-matrix.shape[0] + 1, matrix.shape[1])
        ]
    )


def reconstruct(matrices: List[np.ndarray]) -> np.ndarray:
    return sum(diagonal_averaging(matrix) for matrix in matrices)


def apply_svd(trajectory: np.ndarray) -> Dict[str, np.ndarray]:
    if contains_nan(trajectory):
        raise ContainsNAN
    u, s, v = np.linalg.svd(trajectory)
    return {"U": u, "Sigma": s, "V": v}


def calculate_elementary_matrices(
    svd: Dict[str, np.ndarray], trajectory_matrix: np.ndarray
) -> np.ndarray:
    v_transpose = svd["V"].T
    rank = np.linalg.matrix_rank(trajectory_matrix)
    elementary_matrices = np.array(
        [
            svd["Sigma"][i] * np.outer(svd["U"][:, i], v_transpose[:, i])
            for i in range(0, rank)
        ]
    )
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
    return np.array(
        [
            np.random.normal(mean, var, 1)[0] if np.isnan(point) else point
            for point in time_series
        ]
    )


def get_multinomial_dist_matrix(time_series):
    """Extent for different distributions"""
    unique, counts = np.unique(np.rint(time_series[~np.isnan(time_series)]), return_counts=True)
    return np.array(
        [
            unique[np.argmax(np.random.multinomial(100, counts/sum(counts)))] if np.isnan(point) else point
            for point in time_series
        ]
    )


def square_distance_eigenvalues(old_matrix, new_matrix):
    difference = np.square(old_matrix - new_matrix)
    return np.sum(np.sum(difference, axis=1))


def em_iterations(trajectory, missing_matrix, rank_determination, lag, tolerance, total_iterations):
    loss = []
    eigen_values = []
    chosen_rank = -1
    i = 0
    diff = 100
    X = maximise(trajectory, lag)
    old_matrix = X
    while (i < total_iterations) and (diff > tolerance):
        if np.mod(i, 50) == 0:
            print(f"Iteration{i}")
        X, fitted_matrix, eigen_values, chosen_rank = expectation_step(
            X, trajectory, missing_matrix, rank_determination
        )
        diff = square_distance_eigenvalues(old_matrix, fitted_matrix)
        loss.append(diff)
        old_matrix = fitted_matrix
        X = maximise(X, lag)
        i += 1

    return X, loss, eigen_values, chosen_rank


def determine_rank(eigenvalues, threshold):
    explained_variance = (eigenvalues ** 2).cumsum() / (eigenvalues ** 2).sum()
    return np.where(
        explained_variance == min(explained_variance, key=lambda x: abs(x - threshold))
    )[0][0]


def expectation_step(matrix_iteration, trajectory, missing_matrix, rank_determination: Dict[str, Any]):

    svd = apply_svd(matrix_iteration)
    if "threshold" in rank_determination.keys():
        rank = determine_rank(svd["Sigma"], rank_determination["threshold"])
    else:
        rank = rank_determination["order"]
    cum_contribution = (svd["Sigma"] ** 2).cumsum() / (svd["Sigma"] ** 2).sum()
    fitted_matrix = fit_matrix(svd, matrix_iteration, rank)
    return (
        impute(missing_matrix, trajectory, fitted_matrix),
        fitted_matrix,
        cum_contribution,
        rank,
    )


def maximise(iteration_matrix: np.ndarray, lag: int):
    iteration_ts = diagonal_averaging(iteration_matrix)
    return get_trajectory_matrix(get_multinomial_dist_matrix(iteration_ts), lag)


class SSA:
    def __init__(self, trajectory_matrix, loss, cum_contribution, rank):
        self.trajectory_matrix = trajectory_matrix
        self.loss = loss
        self.cum_contribution = cum_contribution
        self.rank = rank

    @classmethod
    def fit(
        cls, trajectory_matrix: np.ndarray, lag: int, rank_determine: Dict[str, Any], tolerance: float, total_iterations
    ):
        missing_matrix = get_missing_matrix(trajectory_matrix)
        fitted_trajectory, loss, cum_contribution, rank = em_iterations(
            trajectory_matrix, missing_matrix, rank_determine, lag, tolerance, total_iterations
        )
        return cls(fitted_trajectory, loss, cum_contribution, rank)

    @classmethod
    def transform_fit(
            cls, time_series: np.ndarray, lag: int, rank_determine: Dict[str, Any], tolerance: float, total_iterations
    ):
        trajectory_matrix = get_trajectory_matrix(time_series, lag)
        return cls.fit(trajectory_matrix, lag, rank_determine, tolerance, total_iterations)