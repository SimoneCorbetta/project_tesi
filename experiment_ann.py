import numpy as np
import time
import csv

from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split

from algoritm_ann import HNSW


def load_mnist(n_samples=5000):

    print("Loading MNIST dataset...")

    mnist = fetch_openml('mnist_784', version=1)

    X = mnist.data.values.astype(np.float32)

    # normalizzazione
    X /= 255.0

    # prendiamo solo una parte per velocità
    X = X[:n_samples]

    return X


def brute_force_search(data, query, k):

    dists = np.linalg.norm(data - query, axis=1)

    idx = np.argsort(dists)

    return idx[:k]


def recall(true_neighbors, approx_neighbors):

    true_set = set(true_neighbors)
    approx_set = set([i for _, i in approx_neighbors])

    return len(true_set & approx_set) / len(true_set)


def run_experiment():

    N = 5000
    k = 5

    data = load_mnist(N)

    train, queries = train_test_split(data, test_size=0.1, random_state=42)

    hnsw = HNSW()

    insertion_times = []

    print("Building HNSW index...")

    for v in train:

        start = time.time()

        hnsw.insert(v)

        insertion_times.append(time.time() - start)

    results = []

    print("Running queries...")

    for efSearch in [10, 20, 50, 100]:

        for q in queries:

            start = time.time()

            approx = hnsw.search(q, k=k, efSearch=efSearch)

            search_time = time.time() - start

            true = brute_force_search(train, q, k)

            acc = recall(true, approx)

            results.append([
                efSearch,
                search_time,
                acc
            ])

    save_results(results)


def save_results(results):

    with open("aggregated_results.csv", "w", newline="") as f:

        writer = csv.writer(f)

        writer.writerow([
            "efSearch",
            "search_time",
            "accuracy"
        ])

        writer.writerows(results)


if __name__ == "__main__":

    run_experiment()
 