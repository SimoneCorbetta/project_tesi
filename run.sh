#!/bin/bash

python3 clear_results.py

if [ "$1" == "k_iris" ]; then
    python3 experiment_knn_fav.py
    python3 generate_plots_knn_fav.py
elif [ "$1" == "k_mnist" ]; then
    python3 experiment_knn_unfav.py
    python3 generate_plots_knn_unfav.py
elif [ "$1" == "a_iris" ]; then
    python3 experiment_ann_iris.py
    python3 generate_plots_ann.py
elif [ "$1" == "a_mnist" ]; then
    python3 experiment_ann_mnist.py
    python3 generate_plots_ann.py
fi
