import os
import math
import argparse
import numpy as np


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Barabasi-Albert graph generator')
    parser.add_argument('-n', '--n_nodes', type=int, default=32,
                        help="number of nodes for random graph")
    parser.add_argument('-m', '--m_nodes', type=int, required=True,
                        help="initial number of nodes for random graph")
    parser.add_argument('-o', '--out_txt', type=str, required=True,
                        help="name of output txt file")
    args = parser.parse_args()
    n, m = args.n_nodes, args.m_nodes

    assert 1 <= m < n, "m must be smaller than n."

    edges = list()
    deg = np.zeros(n)

    for i in range(m, n):
        if i == m:
            for j in range(i):
                edges.append((j, i))
                deg[j] += 1
                deg[i] += 1
            continue

        connection = np.random.choice(range(n), size=m, replace=False,
                                        p=deg/np.sum(deg))
        for cnt in connection:
            edges.append((cnt, i))
            deg[cnt] += 1
            deg[i] += 1

    edges.sort()

    os.makedirs('generated', exist_ok=True)
    with open(os.path.join('generated', args.out_txt), 'w') as f:
        f.write(str(n) + '\n')
        f.write(str(len(edges)) + '\n')
        for edge in edges:
            f.write('%d %d\n' % (edge[0], edge[1]))
