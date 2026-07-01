import pandas as pd
import matplotlib.pyplot as plt
import os
import numpy as np


RESULTS_FILE = "results/aggregated_results.csv"
BASE_DIR = "results/experiment_ann"


def create_experiment_folder():

  i = 1
  while os.path.exists(os.path.join(BASE_DIR, f"experiment_{i}")):
    i += 1

  folder = os.path.join(BASE_DIR, f"experiment_{i}")
  os.makedirs(folder)

  return folder


def load_data():
  return pd.read_csv(RESULTS_FILE)


def get_grouped(df, group_by):
  return df.groupby(group_by, as_index=False).mean(numeric_only=True)


def plot_recall_vs_time(df, folder, group_by):

  plt.style.use("seaborn-v0_8")
  plt.figure(figsize=(8, 6))

  for value in sorted(df[group_by].unique()):

    subset = df[df[group_by] == value]
    grouped = subset.groupby("efSearch", as_index=False).mean(numeric_only=True)

    plt.plot(
      grouped["search_time"],
      grouped["accuracy"],
      marker="o",
      label=f"{group_by}={value}"
    )

  plt.xlabel("Search Time (s)")
  plt.ylabel("Recall")
  plt.title(f"Recall vs Search Time ({group_by})")
  plt.legend()
  plt.grid(True)

  plt.savefig(
    os.path.join(folder, f"recall_vs_time_{group_by}.png"),
    dpi=300,
    bbox_inches="tight"
  )

  plt.close()


def plot_recall_vs_efsearch(df, folder, group_by):

  plt.style.use("seaborn-v0_8")
  plt.figure(figsize=(8, 6))

  for value in sorted(df[group_by].unique()):

    subset = df[df[group_by] == value]
    grouped = subset.groupby("efSearch", as_index=False).mean(numeric_only=True)

    plt.plot(
      grouped["efSearch"],
      grouped["accuracy"],
      marker="o",
      label=f"{group_by}={value}"
    )

  plt.xlabel("efSearch")
  plt.ylabel("Recall")
  plt.title(f"Recall vs efSearch ({group_by})")
  plt.legend()
  plt.grid(True)

  plt.savefig(
    os.path.join(folder, f"recall_vs_efsearch_{group_by}.png"),
    dpi=300,
    bbox_inches="tight"
  )

  plt.close()


def plot_time_vs_efsearch(df, folder, group_by):

  plt.style.use("seaborn-v0_8")
  plt.figure(figsize=(8, 6))

  for value in sorted(df[group_by].unique()):

    subset = df[df[group_by] == value]
    grouped = subset.groupby("efSearch", as_index=False).mean(numeric_only=True)

    plt.plot(
      grouped["efSearch"],
      grouped["search_time"],
      marker="o",
      label=f"{group_by}={value}"
    )

  plt.xlabel("efSearch")
  plt.ylabel("Search Time (s)")
  plt.title(f"Search Time vs efSearch ({group_by})")
  plt.legend()
  plt.grid(True)

  plt.savefig(
    os.path.join(folder, f"time_vs_efsearch_{group_by}.png"),
    dpi=300,
    bbox_inches="tight"
  )

  plt.close()

"""
def plot_avg_accuracy(df, folder, group_by):

    grouped = df.groupby(group_by)["accuracy"].mean().reset_index()

    plt.figure(figsize=(7, 5))

    plt.bar(grouped[group_by].astype(str), grouped["accuracy"])

    plt.xlabel(group_by)
    plt.ylabel("Average Recall")
    plt.title(f"Average Recall vs {group_by}")
    plt.grid(axis="y")

    plt.savefig(
        os.path.join(folder, f"avg_accuracy_{group_by}.png"),
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()


def plot_avg_time(df, folder, group_by):

    grouped = df.groupby(group_by)["search_time"].mean().reset_index()

    plt.figure(figsize=(7, 5))

    plt.bar(grouped[group_by].astype(str), grouped["search_time"])

    plt.xlabel(group_by)
    plt.ylabel("Average Search Time (s)")
    plt.title(f"Average Time vs {group_by}")
    plt.grid(axis="y")

    plt.savefig(
        os.path.join(folder, f"avg_time_{group_by}.png"),
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()
"""

def generate_plots_ann():

  df = load_data()
  folder = create_experiment_folder()

  parameters = ["M", "efConstruction", "k"]

  for p in parameters:

    plot_recall_vs_time(df, folder, p)
    plot_recall_vs_efsearch(df, folder, p)
    plot_time_vs_efsearch(df, folder, p)
    #plot_avg_accuracy(df, folder, p)
    #plot_avg_time(df, folder, p)

  print(f"Grafici salvati in: {folder}")
