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


visited = set()
adj_list = {}
topsort = []
countt = 0
def dfs(node, need_topsort=False):
    global visited
    global topsort
    global adj_list
    stack = [node]
    visited.add(node)
    second_time = set()
    while len(stack) > 0:
        node = stack.pop()
        if node in second_time:
            if need_topsort:
                topsort.append(node)
        else:
            stack.append(node)
            second_time.add(node)
            for neighbor in adj_list.get(node, []):
                if neighbor not in visited:
                    visited.add(neighbor)
                    stack.append(neighbor)

def count_one_layer(forward_adj_list, reverse_adj_list):
    global visited
    global adj_list
    global topsort
    visited = set()
    topsort = []
    adj_list = forward_adj_list
    for node in adj_list.keys():
        if node in visited:
            continue
        dfs(node, True)
    components_count = 0
    topsort = topsort[::-1]
    visited = set()
    adj_list = reverse_adj_list
    for node in topsort:
        if node not in visited:
            components_count += 1
            dfs(node)
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

    count = 0

    for level_id, level in enumerate(hnsw._graphs):
        # forming adjacency lists
        adj_list = {}
        reverse_adj_list = {}
        for node, neighbours in level.items():
            adj_list[node] = list(neighbours.keys())
        for from_node in level.keys():
            for to_node in adj_list[from_node]:
                if to_node not in reverse_adj_list:
                    reverse_adj_list[to_node] = []
                reverse_adj_list[to_node].append(from_node)
        print(f'components count for level {level_id}: {count_one_layer(adj_list, reverse_adj_list)}')
        count += count_one_layer(adj_list, reverse_adj_list)

    print(f'components count: {count}')


if __name__ == "__main__":
    main()
