import numpy as np

# ================================
# DEFINIZIONE DELLE FUNZIONI DI DISTANZA
# ================================

def euclidean(a, b):
    """
    Calcola la distanza euclidea tra due vettori a e b.
    Formula: sqrt(sum((a_i - b_i)^2))
    """
    return np.sqrt(np.sum((a - b) ** 2))


def manhattan(a, b):
    """
    Calcola la distanza Manhattan (L1 norm).
    Formula: sum(|a_i - b_i|)
    """
    return np.sum(np.abs(a - b))


def cosine(a, b):
    """
    Calcola la distanza coseno.
    È definita come: 1 - similarità coseno.
    Similarità coseno = (a·b) / (||a|| * ||b||)
    """
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)

    if norm_a == 0 or norm_b == 0:
        return 1  # massima distanza

    return 1 - np.dot(a, b) / (norm_a * norm_b)


# ================================
# IMPLEMENTAZIONE KNN ESATTO (BRUTE FORCE)
# ================================

def knn_exact(X, query, k, distance_func):
    """
    Implementazione brute-force del K-Nearest Neighbors.

    Parametri:
    - X: dataset (matrice n x d)
    - query: punto di query
    - k: numero di vicini richiesti
    - distance_func: funzione di distanza da utilizzare

    Output:
    - lista dei k punti più vicini (indice, distanza)
    """

    distances = []

    # Calcolo della distanza tra la query e ogni punto del dataset
    for i, point in enumerate(X):
        dist = distance_func(point, query)
        distances.append((i, dist))

    # Ordinamento crescente rispetto alla distanza
    distances.sort(key=lambda x: x[1])

    # Restituzione dei primi k elementi
    return distances[:k]
    