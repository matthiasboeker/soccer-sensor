from typing import Dict, Tuple

import numpy as np  # type: ignore
import pandas as pd  # type: ignore
from singular_spectrum_analysis.ssa_algorithm import contains_nan, ContainsNAN, NotReversible, get_missing_matrix, calculate_elementary_matrices


def apply_svd(matrix: np.ndarray) -> Dict[str, np.ndarray]:
    if contains_nan(matrix):
        raise ContainsNAN
    u, s, v = np.linalg.svd(matrix)
    return {"U": u, "Sigma": s, "V": v}


def fit_matrix(svd: Dict[str, np.ndarray], trajectory_matrix: np.ndarray, q: int) -> np.ndarray:
    elementary_matrices = calculate_elementary_matrices(svd, trajectory_matrix)
    if not np.allclose(trajectory_matrix, elementary_matrices.sum(axis=0), atol=1e-10):
        raise NotReversible
    return sum(elementary_matrices[i] for i in range(0, q))


def impute(init_matrix: np.ndarray, missing_matrix: np.ndarray, fitted_matrix: np.ndarray) -> np.ndarray:
    return np.nan_to_num(init_matrix) + np.multiply(missing_matrix, fitted_matrix)

def factorise(matrix: np.ndarray, missing_matrix: np.ndarray, rank: int) -> Tuple[np.ndarray, np.ndarray]:
    X = np.ma.MaskedArray(np.nan_to_num(matrix), mask=missing_matrix, fill_value=np.nanmedian(matrix))
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
        cls, matrix: np.ndarray, rank: int
    ) -> TsSVD:
        missing_matrix = get_missing_matrix(matrix)
        imputed_matrix, eigenvalues = factorise(
            matrix, missing_matrix, rank)
        return cls(imputed_matrix, eigenvalues)
