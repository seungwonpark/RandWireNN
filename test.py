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
    hp = HParam('config/test.yaml')
    graphs = [
        read_graph(hp.model.graph0),
        read_graph(hp.model.graph1),
        read_graph(hp.model.graph2),
    ]

    print('Building Network...')
    model = RandWire(hp, graphs)

    x = torch.randn(16, 3, 224, 224) # RGB-channel 224x224 image with batch_size=16
    print('Input shape:')
    print(x.shape)
    y = model(x)
    print('Output shape:')
    print(y.shape) # [3, 1000]
