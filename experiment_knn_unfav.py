import numpy as np
import pandas as pd
import time

from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

from algoritm_knn import *


def experiment_knn_unfav():

    np.random.seed(42)

    print("Caricamento MNIST...")

    mnist = fetch_openml('mnist_784', version=1, as_frame=False)

    X_full = mnist.data.astype(np.float32) / 255.0
    y_full = mnist.target.astype(int)

    raw_results = []

    distances = {
        "euclidean": euclidean,
        "cosine": cosine
    }

    n_points_list = [1000, 5000, 10000]
    dimensions = [100, 300, 784]
    k_values = [1, 5]

    repetitions = 5
    n_queries = 20

    for n in n_points_list:
        for d in dimensions:

            X = X_full[:n, :d]
            y = y_full[:n]

            # ==========================
            # TRAIN TEST SPLIT
            # ==========================

            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42
            )

            for k in k_values:
                for dist_name, dist_func in distances.items():

                    times = []
                    y_pred = []

                    # usiamo alcune query dal test set
                    indices = np.random.choice(len(X_test), n_queries, replace=False)

                    for idx in indices:

                        query = X_test[idx]

                        # ripetizioni per il tempo
                        rep_times = []

                        for r in range(repetitions):

                            start = time.time()

                            neighbors = knn_exact(X_train, query, k, dist_func)

                            end = time.time()

                            rep_times.append((end - start) * 1000)

                        times.append(np.mean(rep_times))

                        # ==========================
                        # PREDIZIONE
                        # ==========================

                        labels = [y_train[i[0]] for i in neighbors]

                        pred = max(set(labels), key=labels.count)

                        y_pred.append(pred)

                    # ==========================
                    # ACCURATEZZA
                    # ==========================

                    y_true = y_test[indices]

                    acc = accuracy_score(y_true, y_pred)

                    raw_results.append({
                        "n_points": n,
                        "dimension": d,
                        "k": k,
                        "distance": dist_name,
                        "mean_time": np.mean(times),
                        "std_time": np.std(times),
                        "accuracy": acc
                    })

    df_raw = pd.DataFrame(raw_results)

    df_agg = df_raw.groupby(
    ["n_points", "dimension", "k", "distance"]
    ).agg(
        mean_time=("mean_time", "mean"),
        std_time=("std_time", "std"),
        accuracy=("accuracy", "mean")
    ).reset_index()

    df_agg.columns = [
        "n_points",
        "dimension",
        "k",
        "distance",
        "mean_time",
        "std_time",
        "accuracy"
    ]

    df_agg.to_csv("results/experiment_knn_unfav/aggregated_results.csv", index=False)

    print("Esperimenti MNIST completati.")
