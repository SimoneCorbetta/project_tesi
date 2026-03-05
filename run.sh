#!/bin/bash

python3 clear_results.py

if [ "$1" == "fav" ]; then
    python3 experiment_knn_fav.py
    python3 generate_plots_fav.py
elif [ "$1" == "unfav" ]; then
    python3 experiment_knn_unfav.py
    python3 generate_plots_unfav.py
fi
