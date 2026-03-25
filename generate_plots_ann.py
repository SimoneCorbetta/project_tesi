import pandas as pd
import matplotlib.pyplot as plt
import os
import numpy as np


RESULTS_FILE = "results/experiment_ann/aggregated_results.csv"
BASE_DIR = "results/experiment_ann"


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


def load_data():
    return pd.read_csv(RESULTS_FILE)


def plot_time_vs_accuracy(df, folder):

    plt.style.use("seaborn-v0_8")

    grouped = df.groupby("efSearch").mean().reset_index()

    x = np.arange(len(grouped["efSearch"]))

    plt.figure()

    plt.bar(
        x,
        grouped["accuracy"]
    )

    plt.xlabel("Search Time (s)")
    plt.ylabel("Accuracy")
    plt.title("Time vs Accuracy")

    plt.xticks(x, [f"{t:.4f}" for t in grouped["search_time"]], rotation=45)

    plt.savefig(
        os.path.join(folder, "time_vs_accuracy.png"),
        dpi=300,
        bbox_inches="tight"
    )

    plt.clf()


def plot_efsearch_vs_accuracy(df, folder):

    plt.style.use("seaborn-v0_8")

    grouped = df.groupby("efSearch").mean().reset_index()

    x = np.arange(len(grouped["efSearch"]))

    plt.figure()

    plt.bar(
        x,
        grouped["accuracy"]
    )

    plt.xlabel("efSearch")
    plt.ylabel("Accuracy")
    plt.title("efSearch vs Accuracy")

    plt.xticks(x, grouped["efSearch"])

    plt.savefig(
        os.path.join(folder, "efsearch_vs_accuracy.png"),
        dpi=300,
        bbox_inches="tight"
    )

    plt.clf()


def plot_efsearch_vs_time(df, folder):

    plt.style.use("seaborn-v0_8")

    grouped = df.groupby("efSearch").mean().reset_index()

    x = np.arange(len(grouped["efSearch"]))

    plt.figure()

    plt.bar(
        x,
        grouped["search_time"]
    )

    plt.xlabel("efSearch")
    plt.ylabel("Search Time (s)")
    plt.title("efSearch vs Search Time")

    plt.xticks(x, grouped["efSearch"])

    plt.savefig(
        os.path.join(folder, "efsearch_vs_time.png"),
        dpi=300,
        bbox_inches="tight"
    )

    plt.clf()


def plot_recall_vs_time(df, folder):

    plt.style.use("seaborn-v0_8")

    grouped = df.groupby("efSearch").mean().reset_index()

    plt.figure()

    # grafico a linea (meglio per curve ANN)
    plt.plot(
        grouped["search_time"],
        grouped["accuracy"],
        marker='o'
    )

    plt.xlabel("Search Time (s)")
    plt.ylabel("Recall (Accuracy)")
    plt.title("Recall vs Search Time")

    # opzionale: annota i punti con efSearch
    for i, ef in enumerate(grouped["efSearch"]):
        plt.annotate(
            str(ef),
            (grouped["search_time"][i], grouped["accuracy"][i])
        )

    plt.savefig(
        os.path.join(folder, "recall_vs_time.png"),
        dpi=300,
        bbox_inches="tight"
    )

    plt.clf()


def generate_plots_ann():

    df = load_data()

    experiment_folder = create_experiment_folder()

    plot_time_vs_accuracy(df, experiment_folder)
    plot_efsearch_vs_accuracy(df, experiment_folder)
    plot_efsearch_vs_time(df, experiment_folder)
    plot_recall_vs_time(df, experiment_folder)

    print(f"Grafici salvati in: {experiment_folder}")
