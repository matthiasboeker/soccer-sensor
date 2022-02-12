from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator

from preprocessing.data_loader import generate_teams
from singular_spectrum_analysis.ssa_algorithm import SSA, diagonal_averaging

params = {
    "lag": 31,
    "order": 10,
    "threshold": 0.1,
    "total_iterations": 250
}


def create_evaluation_plots(real_series, imputed_series):

    fig, axs = plt.subplots(nrows=len(imputed_series), ncols=1, figsize=(15, 10))

    for real_series, imputed_series, ax in zip(real_series, imputed_series, axs.ravel()):
        ax.plot(np.rint(diagonal_averaging(imputed_series)), label="Rounded Imputed Series")
        ax.plot(diagonal_averaging(imputed_series), label="Imputed Series")
        ax.plot(real_series, label="True")
        ax.title(f"Readiness of a Player with Missing Data")
        ax.tick_params(labelrotation=75, labelsize=9)
        ax.xaxis.set_major_locator(MultipleLocator(30))
        ax.legend()
        ax.xlabel("Time")
        ax.ylabel("Readiness Scale")
        plt.savefig(Path(__file__).parent.parent / "evaluation_images" / "evaluation_plots.png")


def main():

    path_to_folder = Path(__file__).parent.parent / "input"
    path_to_data = [[path_to_folder / "rosenborg-women_a_2020.xlsx", path_to_folder / "input" / "rosenborg-women_a_2021.xlsx"],
                    [path_to_folder / "vifwomen_a_2020.xlsx", path_to_folder / "vifwomen_a_2021.xlsx"]]

    teams = generate_teams(path_to_data, ["VI", "Rosenborg"])
    players = teams["Rosenborg"].players
    print("Players loaded")

    fitted_series = [SSA.fit(player.readiness.to_numpy(), params["lag"], params["order"], params["threshold"],
                             params["total_iterations"]) for _, player in players.items()]

    real_series = [player.readiness.to_numpy() for _, player in players.items()]

    create_evaluation_plots(real_series, fitted_series)
