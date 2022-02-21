from typing import Dict
from pathlib import Path
import numpy as np  # type: ignore
import pandas as pd  # type: ignore
import matplotlib.pyplot as plt  # type: ignore
from matplotlib.ticker import MultipleLocator  # type: ignore

from preprocessing.data_loader import generate_teams, SoccerPlayer
from singular_spectrum_analysis.multi_dim_svd import TsSVD

params = {"rank": 13, "tolerance": 0.01, "total_iterations": 100}


def merge_ts_to_df(players: Dict[str, SoccerPlayer]):
    return pd.DataFrame([player.stress for player in players.values()])


def create_evaluation_plots(real_series, imputed_series, eigenvalues):
    fig, ax = plt.subplots(figsize=(20, 15))
    ax.plot(eigenvalues)
    ax.set_title("Relative Importance")
    ax.set_xlabel("Dimensions")
    ax.set_ylabel("Eigenvalues")
    plt.savefig(Path(__file__).parent.parent / "evaluation_images_multivariate" / "eigenvalues.png")

    fig, axs = plt.subplots(nrows=len(imputed_series), ncols=1, figsize=(100, 50))
    for real, imputed, ax in zip(real_series, imputed_series, axs.ravel()):
        ax.plot(imputed, label="Imputed", color="black")
        ax.plot(real, label="True", color="red")
        ax.set_title("Readiness of a Player with Missing Data")
        ax.tick_params(labelrotation=75, labelsize=9)
        ax.xaxis.set_major_locator(MultipleLocator(30))
        ax.legend()
        ax.set_xlabel("Time")
        ax.set_ylabel("Readiness Scale")
    plt.savefig(
        Path(__file__).parent.parent / "evaluation_images_multivariate" / "evaluation_plots.png"
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
    players_team_a = teams["Rosenborg"].players
    players_team_b = teams["VI"].players


    print("Players loaded")

    both_teams = {**players_team_a, **players_team_b}
    teams = merge_ts_to_df(both_teams)
    team_a_SVD = TsSVD.fit(teams.to_numpy(), params["rank"])
    imputed_series = [team_a_SVD.matrix[i,:] for i in range(0, team_a_SVD.matrix.shape[0])]
    real = [player.stress for player in both_teams.values()]
    create_evaluation_plots(real, imputed_series, team_a_SVD.eigenvalues)
    print("Plots created.")



if __name__ == "__main__":
    main()