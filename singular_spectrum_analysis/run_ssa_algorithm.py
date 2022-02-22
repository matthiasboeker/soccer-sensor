from pathlib import Path
import numpy as np  # type: ignore
import matplotlib.pyplot as plt  # type: ignore
from matplotlib.ticker import MultipleLocator  # type: ignore

from preprocessing.data_loader import generate_teams
from singular_spectrum_analysis.ssa_algorithm import SSA, diagonal_averaging

params = {"lag": 31, "rank": {"order": 13}, "tolerance": 0.01, "total_iterations": 200}


def create_evaluation_plots(real_series, imputed_series):

    fig, axs = plt.subplots(
        nrows=int(np.ceil(len(imputed_series) / 4)), ncols=4, figsize=(50, 25)
    )
    for imputed, ax in zip(imputed_series, axs.ravel()):
        ax.plot(imputed.loss)
        ax.set_title("Loss")
        ax.set_xlabel("Iterations")
        ax.set_ylabel("Loss")
    plt.savefig(Path(__file__).parent.parent / "evaluation_images" / "loss.png")

    fig, axs = plt.subplots(
        nrows=int(np.ceil(len(imputed_series) / 4)), ncols=4, figsize=(50, 25)
    )
    for imputed, ax in zip(imputed_series, axs.ravel()):
        ax.plot(imputed.cum_contribution)
        ax.axvline(x=imputed.rank, linewidth=1, color="red")
        ax.set_title("Relative Importance")
        ax.set_xlabel("Dimensions")
        ax.set_ylabel("Cumulative Contribution (%)")
    plt.savefig(Path(__file__).parent.parent / "evaluation_images" / "eigen_values.png")

    fig, axs = plt.subplots(nrows=len(imputed_series), ncols=1, figsize=(100, 50))
    for real, imputed, ax in zip(real_series, imputed_series, axs.ravel()):
        ax.plot(diagonal_averaging(imputed.trajectory_matrix), label="Imputed Series")
        ax.plot(real, label="True")
        ax.set_title("Readiness of a Player with Missing Data")
        ax.tick_params(labelrotation=75, labelsize=9)
        ax.xaxis.set_major_locator(MultipleLocator(30))
        ax.legend()
        ax.set_xlabel("Time")
        ax.set_ylabel("Readiness Scale")
    plt.savefig(
        Path(__file__).parent.parent / "evaluation_images" / "evaluation_plots.png"
    )


def main():

    path_to_folder = Path(__file__).parent.parent / "input"
    path_to_data = [
        [
            path_to_folder / "rosenborg-women_a_2020.xlsx",
            path_to_folder / "rosenborg-women_a_2021.xlsx",
        ],
        [
            path_to_folder / "vifwomen_a_2020.xlsx",
            path_to_folder / "vifwomen_a_2021.xlsx",
        ],
    ]

    teams = generate_teams(path_to_data, ["VI", "Rosenborg"])
    players = teams["Rosenborg"].players
    print("Players loaded")

    fitted_series = [
        SSA.transform_fit(
            player.readiness.to_numpy(),
            params["lag"],
            params["rank"],
            params["tolerance"],
            params["total_iterations"],
        )
        for _, player in players.items()
    ]

    real_series = [player.readiness.to_numpy() for _, player in players.items()]

    create_evaluation_plots(real_series, fitted_series)


if __name__ == "__main__":
    main()
