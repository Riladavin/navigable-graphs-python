import numpy as np
import argparse
import random
from tqdm import tqdm
from queue import Queue


random.seed(251200)
from hnsw_fast import HNSW

def read_fbin(filename, start_idx=0, chunk_size=None):
    """ Read *.fbin file that contains float32 vectors
    Args:
        :param filename (str): path to *.fbin file
        :param start_idx (int): start reading vectors from this index
        :param chunk_size (int): number of vectors to read.
                                 If None, read all vectors
    Returns:
        Array of float32 vectors (numpy.ndarray)
    """
    with open(filename, "rb") as f:
        nvecs, dim = np.fromfile(f, count=2, dtype=np.int32)
        nvecs = (nvecs - start_idx) if chunk_size is None else chunk_size
        arr = np.fromfile(f, count=nvecs * dim, dtype=np.float32,
                          offset=start_idx * 4 * dim)
    return arr.reshape(nvecs, dim)

def get_adjacency_list(graph: HNSW):
    adj_list = {}
    verts_count = len(graph.data)
    for i, level in enumerate(graph._graphs):
        for vert, neighbours in level.items():
            if vert + i * verts_count not in adj_list:
                adj_list[vert + i * verts_count] = set()
            neighbours_ids = [neighbour_id + i * verts_count for neighbour_id, dist in neighbours.items()]
            adj_list[vert + i * verts_count].update(neighbours_ids)
            for neighbour_id in neighbours_ids:
                if neighbour_id not in adj_list:
                    adj_list[neighbour_id] = set()
                adj_list[neighbour_id].add(vert + i * verts_count)
    for level in range(1, len(graph._graphs)):
        for i in range(verts_count):
            v_from = i + level * verts_count
            v_to = i + (level - 1) * verts_count
            if v_to not in adj_list:
                adj_list[v_to] = set()
            if v_from not in adj_list:
                adj_list[v_from] = set()
            adj_list[v_from].add(v_to)
            adj_list[v_to].add(v_from)
    return adj_list


def count_components(graph: HNSW):
    components_count = 0
    adj_list = get_adjacency_list(graph)
    queue = Queue()
    visited = set()
    for i in range(len(graph.data) * len(graph._graphs)):
        if i not in visited:
            components_count += 1
            queue.put(i)
            visited.add(i)
            while not queue.empty():
                v = queue.get()
                for w in adj_list[v]:
                    if w not in visited:
                        queue.put(w)
                        visited.add(w)
    return components_count


def count_one_layer(adj_list):
    components_count = 0
    visited = set()
    queue = Queue()
    for node in adj_list.keys():
        if node in visited:
            continue
        components_count += 1
        queue.put(node)
        visited.add(node)
        while not queue.empty():
            v = queue.get()
            for w in adj_list[v]:
                if w not in visited:
                    queue.put(w)
                    visited.add(w)
    return components_count

def main():
    parser = argparse.ArgumentParser(description='Test recall of beam search method with KGraph.')
    parser.add_argument('--dataset', default='base.10M.fbin', help="Choose the dataset to use.")
    parser.add_argument('--K', type=int, default=5, help='The size of the neighbourhood')
    parser.add_argument('--M', type=int, default=50, help='Avg number of neighbors')
    parser.add_argument('--M0', type=int, default=50, help='Avg number of neighbors')
    parser.add_argument('--dim', type=int, default=2, help='Dimensionality of synthetic data (ignored for SIFT).')
    parser.add_argument('--n', type=int, default=200, help='Number of training points for synthetic data (ignored for SIFT).')
    parser.add_argument('--nq', type=int, default=50, help='Number of query points for synthetic data (ignored for SIFT).')
    parser.add_argument('--k', type=int, default=5, help='Number of nearest neighbors to search in the test stage')
    parser.add_argument('--ef', type=int, default=10, help='Size of the beam for beam search.')
    parser.add_argument('--m', type=int, default=3, help='Number of random entry points.')

    args = parser.parse_args()

    vecs = read_fbin(args.dataset)

    hnsw = HNSW(distance_type='l2', m=args.M, m0=args.M0, ef=args.ef)
    for x in tqdm(vecs):
        hnsw.add(x)

    for level_id, level in enumerate(hnsw._graphs):
        # forming adjacency lists
        adj_list = {}
        edges = set()
        for node, neighbours in level.items():
            for neighbour_id, _ in neighbours.items():
                assert (node, neighbour_id) not in edges
                edges.add((node, neighbour_id))
        for key in level.keys():
            adj_list[key] = []
        for v_from, v_to in edges:
            adj_list[v_from].append(v_to)
            if (v_to, v_from) not in edges:
                adj_list[v_to].append(v_from)
        print(f'components count for level {level_id}: {count_one_layer(adj_list)}')

    print(f'components count: {count_components(hnsw)}')


if __name__ == "__main__":
    main()
