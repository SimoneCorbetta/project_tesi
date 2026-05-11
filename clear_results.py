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

def clear_target_file():
    for folder in folders:

        file_path = os.path.join(folder, "targets.csv")

        os.makedirs(folder, exist_ok=True)

        # sovrascrive il file con solo intestazione
        with open(file_path, "w") as f:
            f.write("sepal length (cm),sepal width (cm),petal length (cm),petal width (cm),target\n")

    print("File target.csv svuotato correttamente.")