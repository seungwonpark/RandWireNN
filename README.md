# RandWireNN
Unofficial PyTorch Implementation of:
[Exploring Randomly Wired Neural Networks for Image Recognition](https://arxiv.org/abs/1904.01569). (WORK IN PROGRESS)

![](./assets/teaser.png)

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

## Test the model

```
python test.py
```

## TODO

New things will be added here.

- [x] implement ER, BA, WS graph generation
- [x] implement the model
- [x] write training/logging code with TensorboardX
- [ ] download ImageNet dataset and implement `dataloader.py`
- [ ] train the network
- [ ] estimate appropriate batch size for specific GPU
- [ ] write results here

I plan to use Adam optimizer, instead of [Distributed SGD](https://arxiv.org/abs/1706.02677).

## Author

[Seungwon Park](http://swpark.me) at MINDsLab (yyyyy@snu.ac.kr, swpark@mindslab.ai)

## License

Apache License 2.0

This repository contains codes adapted/copied from the followings:
- [utils/adabound.py](./utils/adabound.py) from https://github.com/Luolc/AdaBound (Apache License 2.0)
- [utils/hparams.py](./utils/hparams.py) from https://github.com/HarryVolek/PyTorch_Speaker_Verification (No License specified)
