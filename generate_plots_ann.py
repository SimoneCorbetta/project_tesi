import pandas as pd
import matplotlib.pyplot as plt
import os

# modificare codice
    #1 -> grafici plot e non bar
    #2 -> verificare i dati che vengono salvati e come vengono salvati
    #3 -> qui mostrano i grafici invece li voglio salvare in una cartella per gli esperimenti

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

#modifica
def plot_time_vs_accuracy(df, folder):

    grouped = df.groupby("efSearch").mean().reset_index()

    plt.figure()

    plt.plot(grouped["search_time"], grouped["accuracy"], marker='o')

    plt.xlabel("Search Time (s)")
    plt.ylabel("Accuracy")
    plt.title("Time vs Accuracy")

    plt.show()

#modifica
def plot_efsearch_vs_accuracy(df, folder):

    grouped = df.groupby("efSearch").mean().reset_index()

    plt.figure()

    plt.plot(grouped["efSearch"], grouped["accuracy"], marker='o')

    plt.xlabel("efSearch")
    plt.ylabel("Accuracy")
    plt.title("efSearch vs Accuracy")

    plt.show()

#modifica
def plot_efsearch_vs_time(df, folder):

    grouped = df.groupby("efSearch").mean().reset_index()

    plt.figure()

    plt.plot(grouped["efSearch"], grouped["search_time"], marker='o')

    plt.xlabel("efSearch")
    plt.ylabel("Search Time")
    plt.title("efSearch vs Search Time")

    plt.show()

def generate_plots_ann():

    df = load_data()
    experiment_folder = create_experiment_folder()
    plot_time_vs_accuracy(df, experiment_folder)
    plot_efsearch_vs_accuracy(df, experiment_folder)
    plot_efsearch_vs_time(df, experiment_folder)

    print(f"Grafici salvati in: {experiment_folder}")
