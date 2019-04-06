import os


def read_graph(txtfile):
    txtpath = os.path.join('model', 'graphs', 'generated', txtfile)
    with open(txtpath, 'r') as f:
        num_nodes = int(f.readline().strip())
        num_edges = int(f.readline().strip())
        edges = list()
        for _ in range(num_edges):
            s, e = map(int, f.readline().strip().split())
            edges.append((s, e))

        temp = dict()
        temp['num_nodes'] = num_nodes
        temp['edges'] = edges
        return temp
