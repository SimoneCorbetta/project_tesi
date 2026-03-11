import numpy as np
import heapq
import random
import math
from collections import defaultdict


class HNSWNode:

    def __init__(self, idx, vector, level):
        self.idx = idx
        self.vector = vector
        self.level = level
        self.neighbors = defaultdict(list)


class HNSW:

    def __init__(self, M=5, M_max=10, efConstruction=100, mL=1/math.log(5)):

        self.M = M
        self.M_max = M_max
        self.efConstruction = efConstruction
        self.mL = mL

        self.nodes = {}

        self.entry_point = None
        self.max_layer = -1
        self.dim = None


    def distance(self, a, b):
        return np.linalg.norm(a - b)


    def random_layer(self):
        r = random.random()
        return int(-math.log(r) * self.mL)


    def greedy_search(self, query, ep_id, ef, layer):

        visited = set()
        candidates = []
        best = []

        ep = self.nodes[ep_id]
        dist = self.distance(query, ep.vector)

        heapq.heappush(candidates, (dist, ep_id))
        heapq.heappush(best, (-dist, ep_id))

        visited.add(ep_id)

        while candidates:

            dist, current = heapq.heappop(candidates)

            worst_best = -best[0][0]

            if dist > worst_best:
                break

            for neigh in self.nodes[current].neighbors[layer]:

                if neigh in visited:
                    continue

                visited.add(neigh)

                d = self.distance(query, self.nodes[neigh].vector)

                if len(best) < ef or d < worst_best:

                    heapq.heappush(candidates, (d, neigh))
                    heapq.heappush(best, (-d, neigh))

                    if len(best) > ef:
                        heapq.heappop(best)

        result = sorted([(-d, idx) for d, idx in best])

        return result


    def connect_neighbors(self, node_id, candidates, layer):

        neighbors = [idx for _, idx in candidates[:self.M]]

        self.nodes[node_id].neighbors[layer] = neighbors

        for n in neighbors:

            self.nodes[n].neighbors[layer].append(node_id)

            if len(self.nodes[n].neighbors[layer]) > self.M_max:

                dists = [
                    (self.distance(self.nodes[n].vector,
                                   self.nodes[x].vector), x)
                    for x in self.nodes[n].neighbors[layer]
                ]

                dists.sort()

                self.nodes[n].neighbors[layer] = [x for _, x in dists[:self.M_max]]


    def insert(self, vector):

        idx = len(self.nodes)

        if self.dim is None:
            self.dim = len(vector)

        level = self.random_layer()

        node = HNSWNode(idx, vector, level)

        self.nodes[idx] = node

        if self.entry_point is None:
            self.entry_point = idx
            self.max_layer = level
            return

        ep = self.entry_point

        for layer in range(self.max_layer, level, -1):

            res = self.greedy_search(vector, ep, 1, layer)
            ep = res[0][1]

        for layer in range(min(level, self.max_layer), -1, -1):

            candidates = self.greedy_search(vector, ep, self.efConstruction, layer)

            self.connect_neighbors(idx, candidates, layer)

            ep = candidates[0][1]

        if level > self.max_layer:

            self.entry_point = idx
            self.max_layer = level


    def search(self, query, k=5, efSearch=50):

        ep = self.entry_point

        for layer in range(self.max_layer, 0, -1):

            res = self.greedy_search(query, ep, 1, layer)
            ep = res[0][1]

        res = self.greedy_search(query, ep, efSearch, 0)

        return res[:k]