import os
import sys


def clear_results():
    experiment = sys.argv[1] 
    file_path = os.path.join("./results/", "aggregated_results.csv")

        # controllo sicurezza: crea il file se non esiste
    os.makedirs("./results/", exist_ok=True)

    with open(file_path, "w") as f:

        # intestazioni diverse per ANN
        if "ann" in experiment:
            f.write("M,efConstruction,k,efSearch,search_time,accuracy\n")
        else:
            f.write("query_point,k,distance,neighbors,time\n")
    print("Tutti i file dei risultati sono stati svuotati.")

def clear_target_file():
    experiment = sys.argv[1]  # "iris", "mnist" oppure "ann"
    headers = {
        "iris": "sepal length (cm),sepal width (cm),petal length (cm),petal width (cm),target\n",
        "mnist": "pixel_values,target\n",
        "ann": "pixel_values,target\n"
    }
    if experiment not in headers:
        raise ValueError("Esperimento non valido. Usa 'iris' oppure 'mnist'.")

    file_path = os.path.join("./results/", "targets.csv")
    os.makedirs("./results/", exist_ok=True)
    # sovrascrive il file con solo intestazione
    with open(file_path, "w") as f:
        f.write(headers[experiment])

    print(f"File targets.csv svuotato correttamente per {experiment}.")
