from typing import Dict, Tuple, Union

import numpy as np  # type: ignore
import pandas as pd  # type: ignore
from singular_spectrum_analysis.ssa_algorithm import contains_nan, ContainsNAN, NotReversible, get_missing_matrix, calculate_elementary_matrices


def binary_distance(signal: np.ndarray) -> np.ndarray:
    distance_signal = []
    delta = 0
    for index, value in enumerate(signal):
        if signal[index] == 0:
            delta = index - index - 1
            distance_signal.append(delta)
        else:
            delta = index - index - 1 + delta
            distance_signal.append(delta)
    return np.array(distance_signal)


def rev_binary_distance(signal: np.ndarray) -> np.ndarray:
    distance_signal = []
    delta = 0
    index = len(signal) - 1
    while index >= 0:
        if signal[index] == 0:
            delta = index - index - 1
            distance_signal.append(delta)
        else:
            delta = index - index - 1 + delta
            distance_signal.append(delta)

        index -= 1
    return np.flip(np.array(distance_signal))


def forward_backward_fill(signal: pd.Series) -> pd.Series:
    signal_bw = signal.bfill()
    return signal_bw.ffill()


def get_missing_distance(array: np.ndarray) -> np.ndarray:
    rev_dist = rev_binary_distance(array)
    dist = binary_distance(array)
    return rev_dist*dist


def get_binary_missing(signal: np.ndarray) -> np.ndarray:
    return np.isnan(signal)


def calculate_decay(x_bin: np.ndarray, x_t: np.ndarray, distance: np.ndarray, mean: float) -> np.ndarray:
    return (1-x_bin)*x_t + x_bin*(distance*x_t + (1-distance)*mean)


def decay_filling(binary_signal: np.ndarray, signal: pd.Series, distance: np.ndarray, mean: float) -> np.ndarray:
    return np.array(
        [calculate_decay(x_bin, x_t, dist, mean) for x_bin, x_t, dist in
         zip(binary_signal.tolist(), signal.tolist(), distance.tolist())])


def neg_exp_relu(array: np.ndarray) -> np.ndarray:
    return np.array([np.exp(-max(0.0, x)) for x in array])


def normalise(signal: np.ndarray) -> np.ndarray:
    return np.array([(max(signal) - value) / (max(signal) - min(signal)) for value in signal])


def decay_fill_signal(signal: np.ndarray, scaling_func=normalise) -> np.ndarray:
    mean = np.nanmean(signal)[0]
    fill_df = forward_backward_fill(pd.Series(signal))
    binary_missing = get_binary_missing(signal)
    missing_distance = get_missing_distance(binary_missing)
    scaled_missing_distance = scaling_func(missing_distance)
    return decay_filling(binary_missing, fill_df, scaled_missing_distance, mean)


def apply_decay_filling(matrix):
    return np.apply_along_axis(decay_fill_signal, 0, matrix)


def apply_svd(matrix: np.ndarray) -> Dict[str, np.ndarray]:
    if contains_nan(matrix):
        raise ContainsNAN
    u, s, v = np.linalg.svd(matrix, full_matrices=False)
    return {"U": u, "Sigma": s, "V": v}


def fit_matrix(svd: Dict[str, np.ndarray], matrix: np.ndarray, q: int) -> Union[np.ndarray, int]:
    elementary_matrices = calculate_elementary_matrices(svd, matrix)
    if not np.allclose(matrix, elementary_matrices.sum(axis=0), atol=1e-10):
        raise NotReversible
    return sum(elementary_matrices[i] for i in range(0, q))


def impute(init_matrix: np.ndarray, missing_matrix: np.ndarray, fitted_matrix: np.ndarray) -> np.ndarray:
    return np.nan_to_num(init_matrix) + np.multiply(missing_matrix, fitted_matrix)


def factorise(matrix: np.ndarray, missing_matrix: np.ndarray, rank: int, fill_with_decay: bool) -> Tuple[np.ndarray, np.ndarray]:
    if fill_with_decay:
        X = apply_decay_filling(matrix)
    else:
        X = np.ma.MaskedArray(np.nan_to_num(matrix, nan=np.nanmean(matrix)), mask=missing_matrix)
    svd = apply_svd(X)
    eigenvalues = np.log(svd["Sigma"] ** 2)
    X = fit_matrix(svd, X, rank)
    return (
        impute(matrix, missing_matrix, X),
        eigenvalues,
    )


class TsSVD:
    def __init__(self, matrix: np.ndarray, eigenvalues: np.ndarray):
        self.matrix = matrix
        self.eigenvalues = eigenvalues

    @classmethod
    def fit(
        cls, matrix: np.ndarray, rank: int, fill_with_decay=True
    ):
        missing_matrix = get_missing_matrix(matrix)
        imputed_matrix, eigenvalues = factorise(
            matrix, missing_matrix, rank, fill_with_decay)
        return cls(imputed_matrix, eigenvalues)
