import sys

from clear_results import *
from experiment_knn_fav import *
from experiment_knn_unfav import *
from generate_plots_knn_fav import *
from generate_plots_knn_unfav import *
from generate_plots_ann import *
from experiment_ann import *

# codice che permette di svolgere i vari esperimenti senza chiamare su terminale ogni file
# la sequenza consiste in:
#   1 pulire i file all'inizio, cosi facendo se qualche dato contiene dei dati puo' essere liberato per un successivo esperimento
#   2 in base al argv[1] che ho scritto sul terminale quando eseguo questo file, esegur in sequenza l'esperimento
#   3 e successivamente anche la generazione dei grafici basandoci sui dati raccolti dall'esperimento
#   4 ripulisco i file dai dati dell'esperimento appena svolto in modo che non ci siano salvataggi di file con una quantita' di dati inutili

clear_results()
clear_target_file()
if (sys.argv[1] == "iris"):
  experiment_knn_fav()
  generate_plots_knn_fav()
elif (sys.argv[1] == "mnist"):
  experiment_knn_unfav()
  generate_plots_knn_unfav()
elif (sys.argv[1] == "ann"):
  run_experiment()
  generate_plots_ann()
else:
  assert(False)
clear_target_file()
