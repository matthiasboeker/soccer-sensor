import pandas as pd  # type: ignore


class DifferentLength(Exception):
    pass


def spearman_corr(series_x: pd.Series, series_y: pd.Series) -> float:
    if len(series_x) != len(series_y):
        raise DifferentLength
    else:
        nr_observations = len(series_x)
    sum_over_squares = sum((x - y) ** 2 for x, y in zip(series_x, series_y))
    return 1 - 6 * sum_over_squares / nr_observations * (nr_observations ** 2 - 1)
