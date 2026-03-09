import numpy as np
import pandas as pd
import time
from sklearn.datasets import load_iris
from algoritm_knn import *
import matplotlib.pyplot as plt

def experiment_knn_fav():
    # Impostiamo il seed per garantire riproducibilità
    np.random.seed(42)

    # ================================
    # CARICAMENTO DATASET IRIS
    # ================================

    iris = load_iris()
    X_full = iris.data  # matrice 150 x 4

    raw_results = []

    # Dizionario che associa nome distanza → funzione
    distances = {
        "euclidean": euclidean,
        "manhattan": manhattan,
        "cosine": cosine
       }

    # Parametri sperimentali, qui i parametri possono essere modificati per fare altre prove con altri dati
    k_values = [1, 3, 5, 10]
    dimensions = [2, 3, 4]   # utilizziamo le prime d feature
    n_queries = 30           # numero di query casuali
    repetitions = 10         # ripetizioni per media statistica

    # ================================
    # CICLO SPERIMENTALE PRINCIPALE
    # ================================

    for d in dimensions:

        # Selezione delle prime d dimensioni
        X = X_full[:, :d]

        for k in k_values:
            for dist_name, dist_func in distances.items():
                for q_id in range(n_queries):

                    # Selezione casuale della query
                    query = X[np.random.randint(0, len(X))]

                    times = []

                    # Ripetiamo più volte per ridurre rumore statistico
                    for r in range(repetitions):

                        start = time.time()

                        # Esecuzione KNN esatto
                        knn_exact(X, query, k, dist_func)

                        end = time.time()

                        # Tempo in millisecondi
                        times.append((end - start) * 1000)

                    # Salvataggio tempi grezzi
                    for t in times:
                        raw_results.append({
                            "dimension": d,
                            "k": k,
                            "distance": dist_name,
                            "query_id": q_id,
                            "time_ms": t
                        })

    # ================================
    # SALVATAGGIO RISULTATI
    # ================================

    df_raw = pd.DataFrame(raw_results)
    # Calcolo media e deviazione standard
    df_agg = df_raw.groupby(
        ["dimension", "k", "distance"]
       )["time_ms"].agg(["mean", "std"]).reset_index()

    df_agg.to_csv("results/experiment_knn_fav/aggregated_results.csv", index=False)

    print("Esperimenti Iris completati.")
