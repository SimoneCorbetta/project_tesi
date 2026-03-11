import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

RESULTS_FILE = "results/experiment_knn_fav/aggregated_results.csv"
BASE_DIR = "results/experiment_knn_fav"


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


def plot_time_vs_k(df, folder):

    plt.style.use("seaborn-v0_8")

    # ciclo su ogni dimensione presente nel dataframe
    for dim in df["dimension"].unique():

        subset = df[df["dimension"] == dim]

        distances = subset["distance"].unique()
        k_values = sorted(subset["k"].unique())

        x = np.arange(len(k_values))
        width = 0.25

        for i, dist in enumerate(distances):

            data = subset[subset["distance"] == dist].sort_values("k")

            plt.bar(
                x + i * width,
                data["mean_time"],
                width=width,
                label=dist
            )

        plt.xlabel("Valore di k")
        plt.ylabel("Tempo medio (ms)")
        plt.title(f"Tempo vs k (dimensione = {dim})")
        plt.xticks(x + width, k_values)
        plt.legend()

        plt.savefig(
            os.path.join(folder, f"tempo_vs_k--dim_{dim}.png"),
            dpi=300,
            bbox_inches="tight"
        )

        plt.clf()  # pulisce il grafico per il prossimo


def plot_time_vs_dimension(df, folder):

    plt.style.use("seaborn-v0_8")

    for k_val in df["k"].unique():

        subset = df[df["k"] == k_val]

        dims = sorted(subset["dimension"].unique())
        distances = subset["distance"].unique()

        x = np.arange(len(dims))
        width = 0.25

        for i, dist in enumerate(distances):

            data = subset[subset["distance"] == dist].sort_values("dimension")

            plt.bar(
                x + i * width,
                data["mean_time"],
                width=width,
                label=dist
            )

        plt.xlabel("Dimensione")
        plt.ylabel("Tempo medio (ms)")
        plt.title(f"Tempo vs Dimensione (k = {k_val})")
        plt.xticks(x + width, dims)
        plt.legend()

        plt.savefig(
            os.path.join(folder, f"tempo_vs_dimensione--k_{k_val}.png"),
            dpi=300,
            bbox_inches="tight"
        )

        plt.clf()


def plot_k_vs_accuratezza(df, folder):

    plt.style.use("seaborn-v0_8")

    for dim in df["dimension"].unique():

        subset = df[df["dimension"] == dim]

        k_vals = sorted(subset["k"].unique())
        distances = subset["distance"].unique()

        x = np.arange(len(k_vals))
        width = 0.25

        for i, dist in enumerate(distances):

            data = subset[subset["distance"] == dist].sort_values("k")

            plt.bar(
                x + i * width,
                data["accuracy"],
                width=width,
                label=dist
            )

        plt.xlabel("Valori k")
        plt.ylabel("Accuratezza")
        plt.title(f"k vs Accuratezza (dimensione = {dim})")
        plt.xticks(x + width, k_vals)
        plt.legend()

        plt.savefig(
            os.path.join(folder, f"k_vs_accuracy--dim_{dim}.png"),
            dpi=300,
            bbox_inches="tight"
        )

        plt.clf()


def generate_plots_knn_fav():
    df = pd.read_csv(RESULTS_FILE)

    experiment_folder = create_experiment_folder()

    plot_time_vs_k(df, experiment_folder)
    plot_time_vs_dimension(df, experiment_folder)
    plot_k_vs_accuratezza(df, experiment_folder)

    print(f"Grafici salvati in: {experiment_folder}")
