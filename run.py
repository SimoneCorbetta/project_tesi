import sys

from clear_results import *
from experiment_knn_fav import *
from experiment_knn_unfav import *

clear_results()
if (sys.argv[1] == "k_iris"):
    experiment_knn_fav()
    generate_plots_knn_fav()
elif (sys.argv[1] == "k_mnist"):
    experiment_knn_unfav()
    generate_plots_knn_unfav()
elif (sys.argv[1] == "a_iris"):
    # python3 experiment_ann_iris.py
    # python3 generate_plots_ann.py
    pass
elif (sys.argv[1] == "a_mnist"):
    # python3 experiment_ann_mnist.py
    # python3 generate_plots_ann.py
    pass
else:
    assert(false)
