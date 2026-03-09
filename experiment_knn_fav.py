import numpy as np
import pandas as pd
import time
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from algoritm_knn import *

def experiment_knn_fav():
    # Impostiamo il seed per garantire riproducibilità
    np.random.seed(42)

    # ================================
    # CARICAMENTO DATASET IRIS
    # ================================

    iris = load_iris()
    X_full = iris.data  # matrice 150 x 4
    Y_full = iris.target

    X_train, X_test, y_train, y_test = train_test_split(
        X_full, Y_full, test_size=0.3, random_state=42
    )

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

    # ================================
    # CICLO SPERIMENTALE PRINCIPALE
    # ================================

    for d in dimensions:

        # Selezione delle prime d dimensioni
        Xtr = X_train[:, :d]
        Xte = X_test[:, :d]

        for k in k_values:
            for dist_name, dist_func in distances.items():

                times = []
                accuracies = []

                # Ripetiamo più volte per ridurre rumore statistico
                for r in range(len(Xte)):
                    query = Xte[r]

                    start = time.time()

                        # Esecuzione KNN esatto
                    neighbors = knn_exact(Xtr, query, k, dist_func)
                    #print(neighbors)

                    end = time.time()

                    # Tempo in millisecondi
                    times.append((end - start) * 1000)

                    # voto maggioritario
                    labels = [y_train[idx[0]] for idx in neighbors]
                    pred = max(set(labels), key=labels.count)

                    accuracies.append(pred)
                        
                # Calcolo accuratezza
                acc = accuracy_score(y_test, accuracies)

                # Ripetizioni per stabilità del tempo
                for t in times:
                    raw_results.append({
                        "dimension": d,
                        "k": k,
                        "distance": dist_name,
                        "time_ms": t,
                        "accuracy": acc
                    })
    # ================================
    # SALVATAGGIO RISULTATI
    # ================================

    df_raw = pd.DataFrame(raw_results)
    # Calcolo media e deviazione standard
    
    df_agg = df_raw.groupby(
        ["dimension", "k", "distance"]
    ).agg({
        "time_ms": ["mean", "std"],
        "accuracy": "mean"
    }).reset_index()

    df_agg.columns = [
        "dimension",
        "k",
        "distance",
        "mean_time",
        "std_time",
        "accuracy"
    ]

    df_agg.to_csv("results/experiment_knn_fav/aggregated_results.csv", index=False)

    print("Esperimenti Iris completati.")
