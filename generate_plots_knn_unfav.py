import os
import pandas as pd
import matplotlib.pyplot as plt

RESULTS_FILE = "results/experiment_knn_unfav/aggregated_results.csv"
BASE_DIR = "results/experiment_knn_unfav"


def create_experiment_folder():
    """
    crea automaticamente experiment_1, experiment_2, ...
    """
    i = 1
    while os.path.exists(os.path.join(BASE_DIR, f"experiment_{i}")):
        i += 1

    folder = os.path.join(BASE_DIR, f"experiment_{i}")
    os.makedirs(folder)

    return folder


def plot_time_vs_points(df, folder):

    subset = df[(df["dimension"] == 784) & (df["k"] == 5)]

    for dist in subset["distance"].unique():
        data = subset[subset["distance"] == dist]
        plt.plot(data["n_points"], data["mean"], label=dist)

    plt.xlabel("Numero punti")
    plt.ylabel("Tempo medio (ms)")
    plt.title("Tempo vs Numero punti (MNIST)")
    plt.legend()

    plt.savefig(os.path.join(folder, "tempo_vs_punti.png"), dpi=300, bbox_inches="tight")
    plt.clf()


def plot_time_vs_dimension(df, folder):

    subset = df[(df["n_points"] == 5000) & (df["k"] == 5)]

    for dist in subset["distance"].unique():
        data = subset[subset["distance"] == dist]
        plt.plot(data["dimension"], data["mean"], label=dist)

    plt.xlabel("Dimensione")
    plt.ylabel("Tempo medio (ms)")
    plt.title("Tempo vs Dimensione (MNIST)")
    plt.legend()

    plt.savefig(os.path.join(folder, "tempo_vs_dimensione.png"), dpi=300, bbox_inches="tight")
    plt.clf()


def generate_plots_knn_unfav():
    df = pd.read_csv(RESULTS_FILE)

    experiment_folder = create_experiment_folder()

    plot_time_vs_points(df, experiment_folder)
    plot_time_vs_dimension(df, experiment_folder)

    print(f"Grafici salvati in: {experiment_folder}")
