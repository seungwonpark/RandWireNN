import torch
import torch.nn as nn
from collections import deque

from .node import NodeOp


class DAGLayer(nn.Module):
    def __init__(self, in_channel, out_channel, num_nodes, edges):
        super(DAGLayer, self).__init__()
        self.num_nodes = num_nodes
        self.edges = edges

        self.adjlist = [[] for _ in range(num_nodes)] # adjacency list
        self.rev_adjlist = [[] for _ in range(num_nodes)] # reversed adjlist
        self.in_degree = [0 for _ in range(num_nodes)]
        self.out_degree = [0 for _ in range(num_nodes)]

        for s, e in edges:
            self.in_degree[e] += 1
            self.out_degree[s] += 1
            self.adjlist[s].append(e)
            self.rev_adjlist[e].append(s)

        self.input_nodes = [x for x in range(num_nodes)
                                if self.in_degree[x] == 0]
        self.output_nodes = [x for x in range(num_nodes)
                                if self.out_degree[x] == 0]
        assert len(self.input_nodes) > 0, '%d' % len(self.input_nodes)
        assert len(self.output_nodes) > 0, '%d' % len(self.output_nodes)

        for node in self.input_nodes:
            assert len(self.rev_adjlist[node]) == 0
            self.rev_adjlist[node].append(-1)

        self.nodes = nn.ModuleList([
            NodeOp(in_degree=max(1, self.in_degree[x]),
                   in_channel=in_channel,
                   out_channel=out_channel if x in self.output_nodes else in_channel,
                   stride=2 if x in self.input_nodes else 1)
            for x in range(num_nodes)])

    def forward(self, y):
        # y: [B, C, N, M]
        outputs = [None for _ in range(self.num_nodes)] + [y]
        queue = deque(self.input_nodes)
        in_degree = self.in_degree.copy()

        while queue:
            now = queue.popleft()
            input_list = [outputs[x] for x in self.rev_adjlist[now]]
            feed = torch.stack(input_list, dim=-1)
            outputs[now] = self.nodes[now](feed)
            for v in self.adjlist[now]:
                in_degree[v] -= 1
                if in_degree[v] == 0:
                    queue.append(v)

        out_list = [outputs[x] for x in self.output_nodes]
        return torch.mean(torch.stack(out_list), dim=0) # [B, C, N, M]
