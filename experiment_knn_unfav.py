import numpy as np
import pandas as pd
import time
from sklearn.datasets import fetch_openml
from knn import *

np.random.seed(42)

# ================================
# CARICAMENTO MNIST
# ================================

print("Caricamento MNIST...")
mnist = fetch_openml('mnist_784', version=1, as_frame=False)
X_full = mnist.data.astype(np.float32) / 255.0  # normalizzazione

raw_results = []

distances = {
    "euclidean": euclidean,
    "cosine": cosine
}

n_points_list = [1000, 5000, 10000]
dimensions = [100, 300, 784]
k_values = [1, 5]
n_queries = 20
repetitions = 5

for n in n_points_list:
    for d in dimensions:

        X = X_full[:n, :d]

        for k in k_values:
            for dist_name, dist_func in distances.items():
                for q_id in range(n_queries):

                    query = X[np.random.randint(0, n)]
                    times = []

                    for r in range(repetitions):

                        start = time.time()
                        knn_exact(X, query, k, dist_func)
                        end = time.time()

                        times.append((end - start) * 1000)

                    for t in times:
                        raw_results.append({
                            "n_points": n,
                            "dimension": d,
                            "k": k,
                            "distance": dist_name,
                            "query_id": q_id,
                            "time_ms": t
                        })

df_raw = pd.DataFrame(raw_results)
df_agg = df_raw.groupby(
    ["n_points", "dimension", "k", "distance"]
)["time_ms"].agg(["mean", "std"]).reset_index()

df_agg.to_csv("results/experiment_knn_unfav/aggregated_results.csv", index=False)

print("Esperimenti MNIST completati.")
