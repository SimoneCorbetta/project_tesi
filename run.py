import sys

from clear_results import *
from experiment_knn_fav import *
from experiment_knn_unfav import *
from generate_plots_knn_fav import *
from generate_plots_knn_unfav import *
from generate_plots_ann import *
from experiment_ann import *

clear_results()
if (sys.argv[1] == "k_iris"):
    experiment_knn_fav()
    generate_plots_knn_fav()
elif (sys.argv[1] == "k_mnist"):
    experiment_knn_unfav()
    generate_plots_knn_unfav()
elif (sys.argv[1] == "ann"):
    run_experiment()
    generate_plots_ann()
else:
    assert(False)
clear_results()
