from pathlib import Path
import numpy as np  # type: ignore
import matplotlib.pyplot as plt  # type: ignore

from preprocessing.data_loader import generate_teams
from singular_spectrum_analysis.ssa_algorithm import SSA, diagonal_averaging
from forecasting.data_loading import get_dataloader
from forecasting.lstm import FlatLSTM, nn, torch, training, testing


params = {"lag": 31, "rank": {"order": 13}, "tolerance": 0.05, "total_iterations": 100}


def evaluation_graphs(y_ssa, y, preds_ssa, preds):
    fig, ax = plt.subplots(figsize=(20, 15))
    ax.plot(preds, color="orange", label="Predictions")
    #ax.plot(preds_ssa, color="purple", label="SAA Predictions")
    ax.plot(y_ssa, color="darkgreen", label="SSA Labels")
    #ax.plot(y, color="red", label="Labels")
    ax.set_title("LSTM Predictions")
    ax.set_xlabel("Time")
    ax.set_ylabel("Scale")
    ax.legend()
    plt.savefig(Path(__file__).parent.parent / "evaluation_images" / "predictions.png")


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

    train_ts = players["0"].readiness[:500]
    test_ts = players["0"].readiness[500:]
    ssa_train_ts = diagonal_averaging(fitted_series[0].trajectory_matrix)[:500]
    ssa_test_ts = diagonal_averaging(fitted_series[0].trajectory_matrix)[500:]

    train_loaders = [get_dataloader(ts,
                                  sequence_length=7,
                                  prediction_length=7,
                                  bach_size=10,
                                  ) for ts in [train_ts, ssa_train_ts]]

    test_loaders = [get_dataloader(ts,
                                  sequence_length=7,
                                  prediction_length=7,
                                  bach_size=10,
                                  ) for ts in [test_ts, ssa_test_ts]]

    model = FlatLSTM(n_hidden=1, sequence_length=7)
    criterion = nn.MSELoss()
    optimizer = torch.optim.LBFGS(model.parameters(), lr=0.8)

    models = [training(5, loader, optimizer, model, criterion) for loader in train_loaders]

    y, predictions = testing(test_loaders[0], models[0], criterion)
    y_ssa, predictions_ssa = testing(test_loaders[1], models[1], criterion)
    evaluation_graphs(y_ssa,  y,  predictions_ssa, predictions)


if __name__ == "__main__":
    main()
