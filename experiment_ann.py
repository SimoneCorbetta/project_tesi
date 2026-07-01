import numpy as np
import time
import csv

from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split

from algoritm_ann import HNSW


def load_mnist(n_samples=5000):

  print("Loading MNIST dataset...")

  mnist = fetch_openml('mnist_784', version=1)

  X = mnist.data.values.astype(np.float32)

  # normalizzazione
  X /= 255.0

  # prendiamo solo una parte per velocità
  X = X[:n_samples]

  return X


def brute_force_search(data, query, k):

  dists = np.linalg.norm(data - query, axis=1)

  idx = np.argsort(dists)

  return idx[:k]


def recall(true_neighbors, approx_neighbors):

  true_set = set(true_neighbors)
  approx_set = set([i for _, i in approx_neighbors])

  return len(true_set & approx_set) / len(true_set)


def run_experiment():

  N = 5000

  data = load_mnist(N)

  train, queries = train_test_split(
      data,
      test_size=0.1,
      random_state=42
  )

  results = []

  # valori da testare
  M_values = [8, 16, 32]
  efConstruction_values = [100, 200]
  k_values = [5, 25, 50]
  efSearch_values = [10, 20, 50, 100]

  for M in M_values:

    for efConstruction in efConstruction_values:

      print(f"\nBuilding HNSW (M={M}, efConstruction={efConstruction})...")

      hnsw = HNSW(
        M=M,
        efConstruction=efConstruction
      )

      # costruzione indice
      for v in train:
        hnsw.insert(v)

      print("Running queries...")

      for k in k_values:

        for efSearch in efSearch_values:

          total_time = 0
          total_recall = 0

          for q in queries:

            start = time.time()

            approx = hnsw.search(
              q,
              k=k,
              efSearch=efSearch
            )

            total_time += time.time() - start

            true = brute_force_search(train, q, k)

            total_recall += recall(true, approx)

          avg_time = total_time / len(queries)
          avg_recall = total_recall / len(queries)

          results.append([
            M,
            efConstruction,
            k,
            efSearch,
            avg_time,
            avg_recall
          ])

  save_results(results)


def save_results(results):

  with open("results/aggregated_results.csv", "w", newline="") as f:

    writer = csv.writer(f)

    writer.writerow([
      "M",
      "efConstruction",
      "k",
      "efSearch",
      "search_time",
      "accuracy"
    ])

    writer.writerows(results)
 