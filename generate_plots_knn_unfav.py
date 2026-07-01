import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA

RESULTS_FILE = "results/aggregated_results.csv"
TARGETS_FILE = "results/targets.csv"
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


# ===============================
# TEMPO vs NUMERO PUNTI
# ===============================

def plot_time_vs_points(df, folder):

  plt.style.use("seaborn-v0_8")

  for dim in df["dimension"].unique():
    for k_val in df["k"].unique():

      subset = df[(df["dimension"] == dim) & (df["k"] == k_val)]

      if subset.empty:
        continue

      n_points = sorted(subset["n_points"].unique())
      distances = subset["distance"].unique()

      x = np.arange(len(n_points))
      width = 0.25

      for i, dist in enumerate(distances):

        data = subset[subset["distance"] == dist].sort_values("n_points")

        plt.bar(
          x + i * width,
          data["mean_time"],
          width=width,
          label=dist
        )

      plt.xlabel("Numero punti")
      plt.ylabel("Tempo medio (ms)")
      plt.title(f"Tempo vs Numero punti (dim={dim}, k={k_val})")
      plt.xticks(x + width, n_points)
      plt.legend()

      plt.savefig(
        os.path.join(folder, f"tempo_vs_punti_dim_{dim}_k_{k_val}.png"),
        dpi=300,
        bbox_inches="tight"
      )

      plt.clf()


# ===============================
# TEMPO vs DIMENSIONE
# ===============================

def plot_time_vs_dimension(df, folder):

  plt.style.use("seaborn-v0_8")
  for n in df["n_points"].unique():
    for k_val in df["k"].unique():

      subset = df[(df["n_points"] == n) & (df["k"] == k_val)]

      if subset.empty:
        continue

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
      plt.title(f"Tempo vs Dimensione (n={n}, k={k_val})")
      plt.xticks(x + width, dims)
      plt.legend()

      plt.savefig(
        os.path.join(folder, f"tempo_vs_dimensione_n_{n}_k_{k_val}.png"),
        dpi=300,
        bbox_inches="tight"
      )

      plt.clf()


# ===============================
# K vs ACCURATEZZA
# ===============================

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
      # media accuracy per ogni k
      grouped = data.groupby("k")["accuracy"].mean().reindex(k_vals)

      plt.bar(
        x + i * width,
        grouped.values,
        width=width,
        label=dist
      )

    plt.xlabel("Valori k")
    plt.ylabel("Accuratezza")
    plt.title(f"k vs Accuratezza (dimensione = {dim})")
    plt.xticks(x + width, k_vals)
    plt.legend()

    plt.savefig(
      os.path.join(folder, f"k_vs_accuracy_dim_{dim}.png"),
      dpi=300,
      bbox_inches="tight"
    )

    plt.clf()


# ===============================
# DISTRIBUZIONE CLASSI MNIST
# ===============================

def plot_class_distribution(df_targets, folder):

  plt.style.use("seaborn-v0_8")

  counts = df_targets["target"].value_counts().sort_index()
  plt.figure(figsize=(8, 5))

  plt.bar(
    counts.index,
    counts.values
  )

  plt.xlabel("Classe")
  plt.ylabel("Numero immagini")
  plt.title("Distribuzione classi MNIST")

  plt.xticks(range(10))

  plt.grid(axis="y", linestyle="--", alpha=0.5)

  plt.savefig(
    os.path.join(folder, "mnist_class_distribution.png"),
    dpi=300,
    bbox_inches="tight"
  )

  plt.clf()


# ===============================
# ESEMPI IMMAGINI MNIST
# ===============================

def plot_mnist_examples(df_targets, folder):

  plt.style.use("seaborn-v0_8")

  fig, axes = plt.subplots(2, 5, figsize=(10, 5))
  shown_classes = set()

  idx_plot = 0

  for i in range(len(df_targets)):

    label = int(df_targets.iloc[i]["target"])
    if label not in shown_classes:

      row = idx_plot // 5
      col = idx_plot % 5

      # recupero pixel
      pixels = df_targets.iloc[i].drop("target").values.astype(float)

      image = pixels.reshape(28, 28)
      axes[row, col].imshow(
        image,
        cmap="gray"
      )

      axes[row, col].set_title(f"Classe {label}")

      axes[row, col].axis("off")

      shown_classes.add(label)

      idx_plot += 1

    if len(shown_classes) == 10:
      break

  plt.suptitle("Esempi immagini MNIST")
  plt.savefig(
    os.path.join(folder, "mnist_examples.png"),
    dpi=300,
    bbox_inches="tight"
  )

  plt.clf()


# ===============================
# PCA 2D MNIST
# ===============================

def plot_pca_2d(df_targets, folder):

  plt.style.use("seaborn-v0_8")

  X = df_targets.drop("target", axis=1).values
  y = df_targets["target"].values

  pca = PCA(n_components=2)
  X_pca = pca.fit_transform(X)

  plt.figure(figsize=(10, 7))
  scatter = plt.scatter(
    X_pca[:, 0],
    X_pca[:, 1],
    c=y,
    cmap="tab10",   # 👈 colori distinti per 10 classi MNIST
    s=10,
    alpha=0.7
  )

  plt.xlabel("Prima componente principale")
  plt.ylabel("Seconda componente principale")
  plt.title("PCA 2D - MNIST")

  cbar = plt.colorbar(scatter)
  cbar.set_label("Classe")

  plt.grid(True, linestyle="--", alpha=0.4)

  plt.savefig(
    os.path.join(folder, "mnist_pca_2d.png"),
    dpi=300,
    bbox_inches="tight"
  )

  plt.clf()

# ===============================
# GENERAZIONE GRAFICI
# ===============================

def generate_plots_knn_unfav():

  df = pd.read_csv(RESULTS_FILE)
  df_targets = pd.read_csv(TARGETS_FILE)

  experiment_folder = create_experiment_folder()

  plot_time_vs_points(df, experiment_folder)
  plot_time_vs_dimension(df, experiment_folder)
  plot_k_vs_accuratezza(df, experiment_folder)

  plot_class_distribution(df_targets, experiment_folder)
  plot_mnist_examples(df_targets, experiment_folder)
  plot_pca_2d(df_targets, experiment_folder)

  print(f"Grafici salvati in: {experiment_folder}")
