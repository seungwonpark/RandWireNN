import os
import math
import argparse
import numpy as np


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Erdos-Renyi graph generator')
    parser.add_argument('-n', '--n_nodes', type=int, default=32,
                        help="number of nodes for random graph")
    parser.add_argument('-k', '--k_neighbors', type=int, required=True,
                        help="connecting neighboring nodes for WS")
    parser.add_argument('-p', '--prob', type=float, required=True,
                        help="probablity of rewiring for WS")
    parser.add_argument('-o', '--out_txt', type=str, required=True,
                        help="name of output txt file")
    args = parser.parse_args()
    n, k, p = args.n_nodes, args.k_neighbors, args.prob

    assert k % 2 == 0, "k must be even."
    assert 0 < k < n, "k must be larger than 0 and smaller than n."

    adj = [[False]*n for _ in range(n)] # adjacency matrix
    for i in range(n):
        adj[i][i] = True

    # initial connection
    for i in range(n):
        for j in range(i-k//2, i+k//2+1):
            real_j = j % n
            if real_j == i:
                continue
            adj[real_j][i] = adj[i][real_j] = True

    rand = np.random.uniform(0.0, 1.0, size=(n, k//2))
    for i in range(n):
        for j in range(1, k//2+1): # 'j' here is 'i' of paper's notation
            current = (i+j) % n
            if rand[i][j-1] < p: # rewire
                unoccupied = [x for x in range(n) if not adj[i][x]]
                rewired = np.random.choice(unoccupied)
                adj[i][current] = adj[current][i] = False
                adj[i][rewired] = adj[rewired][i] = True

    edges = list()
    for i in range(n):
        for j in range(i+1, n):
            if adj[i][j]:
                edges.append((i, j))

    edges.sort()

    os.makedirs('generated', exist_ok=True)
    with open(os.path.join('generated', args.out_txt), 'w') as f:
        f.write(str(n) + '\n')
        f.write(str(len(edges)) + '\n')
        for edge in edges:
            f.write('%d %d\n' % (edge[0], edge[1]))
