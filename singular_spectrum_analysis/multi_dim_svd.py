from typing import Dict, Tuple, Union, List

import numpy as np  # type: ignore
import pandas as pd  # type: ignore
from statsmodels.stats.diagnostic import acorr_ljungbox
from singular_spectrum_analysis.ssa_algorithm import contains_nan, ContainsNAN, NotReversible, get_missing_matrix, \
    calculate_elementary_matrices, square_distance_eigenvalues


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
    mean = np.nanmean(signal)
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


def test_for_white_noise(elementary_matrix, alpha, lags):
    significants = []
    for i in range(0, elementary_matrix.shape[1]):
        _, p_value_lags = acorr_ljungbox(np.array(elementary_matrix[:,i]), lags=lags, return_df=False)
        is_significant = any(p_value > alpha for p_value in p_value_lags)
        significants.append(is_significant)
    return any(is_sig for is_sig in significants)


def eigenspace_decay_filling(elementary_matrices, missing_matrix):
    nan_matrix = missing_matrix.astype("float")
    nan_matrix[nan_matrix == 1] = np.nan
    return [
        apply_decay_filling(nan_matrix+elementary_matrices[i])
        for i in range(0, len(elementary_matrices))
    ]

def determine_rank(elementary_matrices, lags):
    matrices_nr = len(elementary_matrices)
    is_white_noise: bool = False
    index = 0 #matrices_nr - 1
    while index < matrices_nr  and is_white_noise==False:
        is_white_noise = test_for_white_noise(elementary_matrices[index], alpha=0.05, lags=lags)
        index += 1
    return index


def fit_matrix(svd: Dict[str, np.ndarray], matrix: np.ndarray, missing_matrix, rank) -> Union[np.ndarray, int]:
    elementary_matrices = calculate_elementary_matrices(svd, matrix)
    if not np.allclose(matrix, elementary_matrices.sum(axis=0), atol=1e-10):
        raise NotReversible
    #reverse_sigma = (1/(len(svd["Sigma"])-q))*sum(svd["Sigma"][i] for i in range(q, len(svd["Sigma"])))
    #elementary_matrices_imputed = eigenspace_decay_filling(elementary_matrices, missing_matrix)
    return sum(elementary_matrices[i] for i in range(0, rank)), rank
    #sum(elementary_matrices[i]*(svd["Sigma"][i]-reverse_sigma)/svd["Sigma"][i] for i in range(0, q))


def impute(init_matrix: np.ndarray, missing_matrix: np.ndarray, fitted_matrix: np.ndarray) -> np.ndarray:
    return np.nan_to_num(init_matrix) + np.multiply(missing_matrix, fitted_matrix)


def factorise(
        matrix: np.ndarray,
        missing_matrix: np.ndarray,
        rank: int,
        threshold,
        tolerance) -> Tuple[np.ndarray, np.ndarray, List[float], int]:
    diff = 100
    i = 0
    stored_values = []
    old_matrix = np.zeros((matrix.shape[0], matrix.shape[1]))+1
    X = apply_decay_filling(matrix)
    while i < threshold and tolerance < diff:
        parameters = {}
        M = np.tile(np.mean(X,axis=0),(len(X),1))
        svd = apply_svd(X-M)
        eigenvalues = np.log(svd["Sigma"] ** 2)
        fitted_res_matrix, rank = fit_matrix(svd, X-M, missing_matrix, rank)
        fitted_matrix = fitted_res_matrix + M
        diff = square_distance_eigenvalues(fitted_matrix, old_matrix)
        old_matrix = fitted_matrix

        X = impute(matrix, missing_matrix, fitted_matrix)
        i += 1
        parameters["loss"] = diff
        parameters["matrix"] = X
        parameters["eigenvalues"] = eigenvalues
        parameters["rank"] = rank
        stored_values.append(parameters)

    loss = [iteration["loss"] for iteration in stored_values]
    best_index = np.argmin(loss)
    best_matrix = [iteration["matrix"] for iteration in stored_values][best_index]
    best_eigenvalues = [iteration["eigenvalues"] for iteration in stored_values][best_index]
    best_rank = [iteration["rank"] for iteration in stored_values][best_index]

    return (
        best_matrix,
        best_eigenvalues,
        loss,
        best_rank,
    )


class TsSVD:
    def __init__(self, matrix: np.ndarray, eigenvalues: np.ndarray, loss: List[float], rank: int):
        self.matrix = matrix
        self.eigenvalues = eigenvalues
        self.loss = loss
        self.rank = rank

    @classmethod
    def fit(
        cls, matrix: np.ndarray, rank: int, threshold=100, tolerance=0.1):
        missing_matrix = get_missing_matrix(matrix)
        imputed_matrix, eigenvalues, loss, rank = factorise(
            matrix, missing_matrix, rank, threshold, tolerance)
        return cls(imputed_matrix, eigenvalues, loss, rank)
