import numpy as np
from heapq import heappush, heappop, heapify, heapreplace, nlargest, nsmallest
from math import log2
from random import random
from operator import itemgetter


class HNSW:
    def __init__(self, distance_type, m=5, ef=200, m0=None, heuristic=True, vectorized=False):
        self.data = []
        self._m = m  # Number of bi-directional links
        self._ef = ef  # Size of the dynamic candidate list
        self._m0 = 2 * m if m0 is None else m0  # Max connections at level 0
        self._level_mult = 1 / log2(m)
        self._graphs = []  # Hierarchical graph layers
        self._enter_point = None  # Entry point in the graph

        # Select distance function
        if distance_type == "l2":
            self.distance_func = self._l2_distance
        else:
            raise ValueError('Invalid distance type! Choose "l2".')

        # Vectorized distance functions
        if vectorized:
            self.distance = self._single_distance
            self.vectorized_distance = self.distance_func
        else:
            self.distance = self.distance_func
            self.vectorized_distance = self._vectorized_distance

        # Neighbor selection function
        self._select = self._select_heuristic if heuristic else self._select_naive

    def _l2_distance(self, a, b):
        return np.linalg.norm(a - b)

    def _single_distance(self, x, y):
        return self.distance_func(x, [y])[0]

    def _vectorized_distance(self, x, ys):
        return [self.distance_func(x, y) for y in ys]

    def count_connected_components(self):

        components = []

        def dfs(node, level):
            stack = [node]
            while stack:
                current = stack.pop()
                for neighbor, _ in self._graphs[level].get(current, {}).items():
                    if neighbor not in visited:
                        visited.add(neighbor)
                        stack.append(neighbor)

        for idx, level in enumerate(self._graphs[::-1]):
            visited = set()
            component_count = 0
            nodes = (level.keys())
            for node in nodes:
                if node not in visited:
                    visited.add(node)
                    component_count += 1
                    dfs(node, idx)

            components.append(component_count)

        return components

    def add(self, elem, ef=None):
        if ef is None:
            ef = self._ef

        idx = len(self.data)
        self.data.append(elem)

        # Determine level for the new element
        level = int(-log2(random()) * self._level_mult) + 1

        if self._enter_point is not None:
            current_point = self._enter_point
            current_dist = self.distance(elem, self.data[current_point])

            # Search for closest neighbor at higher levels
            for layer in reversed(self._graphs[level:]):
                current_point, current_dist = self._search_layer_ef1(elem, current_point, current_dist, layer)

            ep = [(-current_dist, current_point)]
            for layer_level, layer in enumerate(reversed(self._graphs[:level])):
                max_neighbors = self._m if layer_level != 0 else self._m0

                # Search and connect neighbors at the current layer
                ep = self._search_layer(elem, ep, layer, ef)
                layer[idx] = {}
                self._select(layer[idx], ep, max_neighbors, layer, heap=True)

                # Add backlinks
                for neighbor_idx, dist in layer[idx].items():
                    self._select(layer[neighbor_idx], (idx, dist), max_neighbors, layer)
        else:
            # Initialize graphs if this is the first element
            self._graphs.append({idx: {}})
            self._enter_point = idx

        # Extend graphs if necessary
        for _ in range(len(self._graphs), level):
            self._graphs.append({idx: {}})
            self._enter_point = idx

    def search(self, q, k=None, ef=None):
        if ef is None:
            ef = self._ef

        if self._enter_point is None:
            raise ValueError("The graph is empty.")

        current_point = self._enter_point
        current_dist = self.distance(q, self.data[current_point])

        # Search at higher levels
        for layer in reversed(self._graphs[1:]):
            current_point, current_dist = self._search_layer_ef1(q, current_point, current_dist, layer)

        # Search at the base layer
        ep = [(-current_dist, current_point)]
        ep = self._search_layer(q, ep, self._graphs[0], ef)

        if k is not None:
            ep = nlargest(k, ep)
        else:
            ep.sort(reverse=True)

        return [(idx, -dist) for dist, idx in ep]

    def _search_layer_ef1(self, q, entry_point, dist_to_entry, layer):
        visited = set()
        candidates = [(dist_to_entry, entry_point)]
        best_point = entry_point
        best_dist = dist_to_entry
        visited.add(entry_point)

        while candidates:
            dist, current = heappop(candidates)
            if dist > best_dist:
                break

            neighbors = [n for n in layer[current] if n not in visited]
            visited.update(neighbors)
            neighbor_dists = self.vectorized_distance(q, [self.data[n] for n in neighbors])

            for neighbor, neighbor_dist in zip(neighbors, neighbor_dists):
                if neighbor_dist < best_dist:
                    best_point = neighbor
                    best_dist = neighbor_dist
                    heappush(candidates, (neighbor_dist, neighbor))

        return best_point, best_dist

    def _search_layer(self, q, ep, layer, ef):
        visited = set()
        candidates = [(-dist, idx) for dist, idx in ep]
        heapify(candidates)
        visited.update(idx for _, idx in ep)

        while candidates:
            dist, current = heappop(candidates)
            if dist > -ep[0][0]:
                break

            neighbors = [n for n in layer[current] if n not in visited]
            visited.update(neighbors)
            neighbor_dists = self.vectorized_distance(q, [self.data[n] for n in neighbors])

            for neighbor, neighbor_dist in zip(neighbors, neighbor_dists):
                mdist = -neighbor_dist
                if len(ep) < ef:
                    heappush(candidates, (neighbor_dist, neighbor))
                    heappush(ep, (mdist, neighbor))
                elif mdist > ep[0][0]:
                    heappush(candidates, (neighbor_dist, neighbor))
                    heapreplace(ep, (mdist, neighbor))

        return ep

    def _select_naive(self, neighbors, candidates, m, layer, heap=False):
        if heap:
            candidates = nlargest(m, candidates)
            unchecked = m - len(neighbors)
            candidates_to_add = candidates[:unchecked]
            candidates_to_check = candidates[unchecked:]

            if candidates_to_check:
                to_remove = nlargest(len(candidates_to_check), neighbors.items(), key=itemgetter(1))
            else:
                to_remove = []

            for mdist, idx in candidates_to_add:
                neighbors[idx] = -mdist

            for (mdist_new, idx_new), (idx_old, dist_old) in zip(candidates_to_check, to_remove):
                if dist_old <= -mdist_new:
                    break
                del neighbors[idx_old]
                neighbors[idx_new] = -mdist_new
        else:
            idx, dist = candidates
            if len(neighbors) < m:
                neighbors[idx] = dist
            else:
                max_idx, max_dist = max(neighbors.items(), key=itemgetter(1))
                if dist < max_dist:
                    del neighbors[max_idx]
                    neighbors[idx] = dist

    def _select_heuristic(self, neighbors, candidates, m, layer, heap=False):
        neighbor_dicts = [layer[idx] for idx in neighbors]

        def prioritize(idx, dist):
            proximity = any(nd.get(idx, float('inf')) < dist for nd in neighbor_dicts)
            return proximity, dist, idx

        if heap:
            candidates = nsmallest(m, (prioritize(idx, -mdist) for mdist, idx in candidates))
            unchecked = m - len(neighbors)
            candidates_to_add = candidates[:unchecked]
            candidates_to_check = candidates[unchecked:]

            if candidates_to_check:
                to_remove = nlargest(len(candidates_to_check), (prioritize(idx, dist) for idx, dist in neighbors.items()))
            else:
                to_remove = []

            for _, dist, idx in candidates_to_add:
                neighbors[idx] = dist

            for (p_new, d_new, idx_new), (p_old, d_old, idx_old) in zip(candidates_to_check, to_remove):
                if (p_old, d_old) <= (p_new, d_new):
                    break
                del neighbors[idx_old]
                neighbors[idx_new] = d_new
        else:
            idx, dist = candidates
            candidates = [prioritize(idx, dist)]
            if len(neighbors) < m:
                neighbors[idx] = dist
            else:
                max_idx, max_val = max(neighbors.items(), key=itemgetter(1))
                if dist < max_val:
                    del neighbors[max_idx]
                    neighbors[idx] = dist

    def __getitem__(self, idx):
        for layer in self._graphs:
            if idx in layer:
                yield from layer[idx].items()
            else:
                return
