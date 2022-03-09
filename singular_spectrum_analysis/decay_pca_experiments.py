import json
from random import shuffle
from pathlib import Path
import numpy as np  # type: ignore
import pandas as pd  # type: ignore
import matplotlib.pyplot as plt  # type: ignore
from matplotlib.ticker import MultipleLocator  # type: ignore

from fancyimpute import KNN, SoftImpute
from sklearn.metrics import mean_absolute_error
from statsmodels.imputation.bayes_mi import BayesGaussMI

from singular_spectrum_analysis.multi_dim_svd import TsSVD

from contextlib import contextmanager
import sys, os

@contextmanager
def suppress_stdout():
    with open(os.devnull, "w") as devnull:
        old_stdout = sys.stdout
        sys.stdout = devnull
        try:
            yield
        finally:
            sys.stdout = old_stdout

def evaluate(data, params, iterations=1, random_value=0.5):
    pca_results = []
    knn_results = []
    bayes_results = []
    softimpute_results = []

    for iteration in range(0, iterations):
        softimpute_result = {}
        knn_result = {}
        pca_result = {}
        bayes_result = {}
        rows = []
        for row in data:
            edge_nans = [np.nan for _ in range(100)]
            missing_middle = [np.nan if
                              random_value < np.random.normal(1, 1, 1)[0] else 0
                              for _ in range(data.shape[0]-200)]

            mask = [edge_nans, missing_middle, edge_nans]
            shuffle(mask)
            flat_list = [item for sublist in mask for item in sublist]
            rows.append(np.array(data[row].to_numpy() + flat_list))

        missing_df = pd.DataFrame(rows).T

        pca = TsSVD.fit(missing_df.to_numpy(),
                                  rank=params["rank"],
                                  tolerance=params["tolerance"],
                                  threshold=params["threshold"])
        with suppress_stdout():
            knn = KNN().fit_transform(missing_df)
            #bayesimpute = BayesGaussMI(missing_df)
            softimpute = SoftImpute(init_fill_method="mean").fit_transform(missing_df)

        pca_result["loss"] = pca.loss
        pca_result["eigenvalues"] = pca.eigenvalues

        pca_result["data"] = pca.matrix
        softimpute_result["data"] = softimpute
        knn_result["data"] = knn
        #bayes_result["data"] = bayesimpute

        pca_result["mae"] = mean_absolute_error(data, pca.matrix)
        softimpute_result["mae"] = mean_absolute_error(data, softimpute)
        knn_result["mae"] = mean_absolute_error(data, knn)
        #bayes_result["mae"] = mean_absolute_error(data, bayesimpute)

        pca_result["rank"] = pca.rank

        #bayes_results.append(bayes_result)
        pca_results.append(pca_result)
        softimpute_results.append(softimpute_result)
        knn_results.append(knn_result)

    return {"pca_res": pca_results, "softimpute_res": softimpute_results, "knn_res": knn_results,} #"bayes_res": bayes_results}


def create_evaluation_plots(name, real, knn_res, softimpute_res, pca_res):
    fig, ax = plt.subplots(figsize=(20, 15))
    ax.plot(pca_res[0]["eigenvalues"])
    ax.set_title("Relative Importance")
    ax.set_xlabel("Dimensions")
    ax.set_ylabel("Eigenvalues")
    plt.savefig(Path(__file__).parent.parent / "iterative_pca_experiments" / f"Data: '{name}' eigenvalues.png")

    fig, ax = plt.subplots(figsize=(20, 15))
    ax.plot(pca_res[0]["loss"])
    ax.set_title("Loss")
    ax.set_xlabel("Iterations")
    ax.set_ylabel("Loss")
    plt.savefig(Path(__file__).parent.parent / "iterative_pca_experiments" / f"Data: '{name}' loss.png")

    fig, (ax2, ax3, ax4) = plt.subplots(3, 1, figsize=(10, 20))
    ax2.scatter(real.iloc[:, 0], pd.Series(pca_res[0]["data"][:, 0]), label="Decay", facecolors='none',
                edgecolors='black')
    ax2.set_xlabel("Real")
    ax2.set_ylabel("Imputed")
    ax2.set_title(f'Correlation: {np.corrcoef(real.iloc[:, 0], pca_res[0]["data"][:, 0])[0, 1]}')
    ax2.legend()
    ax3.scatter(real.iloc[:, 0], softimpute_res[0]["data"][:, 0], label="SoftImpute", facecolors='none', edgecolors='black')
    ax3.set_xlabel("Real")
    ax3.set_ylabel("Imputed")
    ax3.set_title(f'Correlation: {np.corrcoef(real.iloc[:, 0], softimpute_res[0]["data"][:, 0])[0, 1]}')
    ax3.legend()
    ax4.scatter(real.iloc[:, 0], knn_res[0]["data"][:, 0], label="KNN", facecolors='none', edgecolors='black')
    ax4.set_xlabel("Real")
    ax4.set_ylabel("Imputed")
    ax4.set_title(f'Correlation: {np.corrcoef(real.iloc[:, 0], knn_res[0]["data"][:, 0])[0, 1]}')
    ax4.legend()
    plt.savefig(Path(__file__).parent.parent / "iterative_pca_experiments" / f"Data:'{name}' correlations.png")


def main():

    path_to_folder = Path(__file__).parent.parent / "notebooks"
    experiments = {}
    experiments["climate"] = pd.read_csv(path_to_folder / "DailyDelhiClimateTrain.csv", engine='python', sep=None).iloc[:1400, :]
    experiments["motor_activity"] = pd.read_csv(path_to_folder / "activity.csv", engine='python', sep=None).iloc[:1000, :]
    experiments["cycling"] = pd.read_csv(path_to_folder / "cycling_data.csv", sep=";")
    experiments["basketball"] = pd.read_csv(path_to_folder/ "acc.csv", sep=";").iloc[:5000, :]
    #experiments["smartphone_activity"] = pd.read_csv(path_to_folder / "final_X_test.csv", engine='python', sep=None).iloc[:1400, :]

    params = {"rank": 3, "tolerance": 0.1, "threshold": 100}

    for name, experiment in experiments.items():
        print(f"{name}, Dimensions: {experiment.shape}")
        results = evaluate(experiment, params, random_value=1)
        create_evaluation_plots(name, experiment, results["knn_res"], results["softimpute_res"], results["pca_res"])
        res = {}
        res["name"] = f"MAE of {name}"
        for key,  result in results.items():
            res[key] = np.nanmean([list_item["mae"] for list_item in result])
        res["pca_rank"] = results["pca_res"][0]["rank"]

        out_file = open(Path(__file__).parent.parent / "iterative_pca_experiments" / f"{name} results.json", "w")
        json.dump(res, out_file, indent=1)
        out_file.close()


if __name__ == "__main__":
    main()