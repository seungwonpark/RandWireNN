# RandWireNN
Unofficial PyTorch Implementation of:
[Exploring Randomly Wired Neural Networks for Image Recognition](https://arxiv.org/abs/1904.01569).

Things that are different from paper:
- We used CIFAR-100 dataset instead of ImageNet.
- We used Adam optimizer, instead [Distributed SGD](https://arxiv.org/abs/1706.02677).
- 

## Generate random DAG

```bash
cd model/graphs
python er.py -p 0.2 -o er_02.txt # Erdos-Renyi
python ba.py -m 7 -o ba_7.txt # Barbasi-Albert
python ws.py -k 4 -p 0.75 ws_4_075.txt # Watts-Strogatz
# number of nodes: -n option
```

All outputs from commands shown above will produce txt file like:
```
(number of nodes)
(number of edges)
(lines, each line representing edges)
```
