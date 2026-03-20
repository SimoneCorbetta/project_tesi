import os

#da modificare, aggiungere anche i test legati all'algoritmo ann
#modifiche riguardo i contenuti e i valori dei risultati riportati per ogni esperimento in ann
folders = [
    "results/experiment_knn_fav",
    "results/experiment_knn_unfav",
    "results/experiment_ann"
]

def clear_results():
    for folder in folders:
        file_path = os.path.join(folder, "aggregated_results.csv")

        with open(file_path, "w") as f:
            f.write("query_point,k,distance,neighbors,time\n")

    print("Tutti i file dei risultati sono stati svuotati.")
