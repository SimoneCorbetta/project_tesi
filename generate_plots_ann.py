import pandas as pd
import matplotlib.pyplot as plt

# modificare codice
    #1 -> grafici plot e non bar
    #2 -> verificare i dati che vengono salvati e come vengono salvati
    #3 -> qui mostrano i grafici invece li voglio salvare in una cartella per gli esperimenti

def load_data():

    return pd.read_csv("results_ann.csv")


def plot_time_vs_accuracy(df):

    grouped = df.groupby("efSearch").mean().reset_index()

    plt.figure()

    plt.plot(grouped["search_time"], grouped["accuracy"], marker='o')

    plt.xlabel("Search Time (s)")
    plt.ylabel("Accuracy")
    plt.title("Time vs Accuracy")

    plt.show()


def plot_efsearch_vs_accuracy(df):

    grouped = df.groupby("efSearch").mean().reset_index()

    plt.figure()

    plt.plot(grouped["efSearch"], grouped["accuracy"], marker='o')

    plt.xlabel("efSearch")
    plt.ylabel("Accuracy")
    plt.title("efSearch vs Accuracy")

    plt.show()


def plot_efsearch_vs_time(df):

    grouped = df.groupby("efSearch").mean().reset_index()

    plt.figure()

    plt.plot(grouped["efSearch"], grouped["search_time"], marker='o')

    plt.xlabel("efSearch")
    plt.ylabel("Search Time")
    plt.title("efSearch vs Search Time")

    plt.show()


if __name__ == "__main__":

    df = load_data()

    plot_time_vs_accuracy(df)
    plot_efsearch_vs_accuracy(df)
    plot_efsearch_vs_time(df)