import os
import torch
import torch.nn as nn 

from utils.hparams import HParam
from model.model import RandWire



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

if __name__ == '__main__':
    hp = HParam('config/config.yaml')
    graphs = [
        read_graph(hp.model.graph0),
        read_graph(hp.model.graph1),
        read_graph(hp.model.graph2),
    ]

    model = RandWire(hp, graphs)

    x = torch.randn(3, 1, 224, 224)
    y = model(x)
    print(y.shape)
