import os
import pandas as pd
import matplotlib.pyplot as plt

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

    subset = df[df["dimension"] == 4]

    for dist in subset["distance"].unique():
        data = subset[subset["distance"] == dist]
        plt.plot(data["k"], data["mean_time"], label=dist)

    plt.style.use("seaborn-v0_8")
    plt.xlabel("Valore di k")
    plt.ylabel("Tempo medio (ms)")
    plt.title("Tempo vs k (Iris)")
    plt.legend()
    plt.savefig(os.path.join(folder, "tempo_vs_valorik.png"), dpi=300, bbox_inches="tight")
    plt.clf()   # pulisce il grafico per il prossimo plot


def plot_time_vs_dimension(df, folder):

    subset = df[df["k"] == 5]

    for dist in subset["distance"].unique():
        data = subset[subset["distance"] == dist]
        plt.plot(data["dimension"], data["mean_time"], label=dist)

    plt.style.use("seaborn-v0_8")
    plt.xlabel("Dimensione")
    plt.ylabel("Tempo medio (ms)")
    plt.title("Tempo vs Dimensione (Iris)")
    plt.legend()
    plt.savefig(os.path.join(folder, "tempo_vs_dimensione.png"), dpi=300, bbox_inches="tight")
    plt.clf()   # pulisce il grafico per il prossimo plot


def plot_k_vs_accuratezza(df, folder):

    subset = df[df["dimension"] == 4]

    for dist in subset["distance"].unique():
        data = subset[subset["distance"] == dist]
        plt.plot(data["k"], data["accuracy"], marker="o", label=dist)

    plt.style.use("seaborn-v0_8")
    plt.xlabel("Valori k")
    plt.ylabel("Accuratezza")
    plt.title("Valori k vs Accuratezza(Iris)")
    plt.legend()
    plt.savefig(os.path.join(folder, "valorik_vs_accuratezza.png"), dpi=300, bbox_inches="tight")
    plt.clf()   # pulisce il grafico per il prossimo plot


def generate_plots_knn_fav():
    df = pd.read_csv(RESULTS_FILE)

    experiment_folder = create_experiment_folder()

    plot_time_vs_k(df, experiment_folder)
    plot_time_vs_dimension(df, experiment_folder)
    plot_k_vs_accuratezza(df, experiment_folder)

    print(f"Grafici salvati in: {experiment_folder}")
