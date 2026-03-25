import os

# array che contiene i percorsi delle cartelle degli esperimenti, divisi per algoritmo e dataset
folders = [
    "results/experiment_knn_fav",
    "results/experiment_knn_unfav",
    "results/experiment_ann"
]


def clear_results():

    for folder in folders:

        file_path = os.path.join(folder, "aggregated_results.csv")

        # controllo sicurezza: crea il file se non esiste
        os.makedirs(folder, exist_ok=True)

        with open(file_path, "w") as f:

            # intestazioni diverse per ANN
            if "experiment_ann" in folder:

                f.write("efSearch,search_time,accuracy\n")

            else:

                f.write("query_point,k,distance,neighbors,time\n")

    print("Tutti i file dei risultati sono stati svuotati.")