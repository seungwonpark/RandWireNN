# RandWireNN
Unofficial PyTorch Implementation of:
[Exploring Randomly Wired Neural Networks for Image Recognition](https://arxiv.org/abs/1904.01569).

(WORK IN PROGRESS, currently preparing ImageNet dataset)

![](./assets/teaser.png)

## Dependencies

This code was tested on Python 3.6 with PyTorch 1.0.1. Other packages can be installed by:
```bash
pip install -r requirements.txt
```

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

## Train RandWireNN

1. Download ImageNet dataset. Train/val folder should contain list of 1,000 directories, each containing list of images for corresponding category.
1. Edit `config.yaml`
  ```bash
  cd config
  cp default.yaml config.yaml
  vim config.yaml # specify data directory, graph txt files
  ```
1. Train
  ```
  python trainer.py -c [config yaml] -m [name]
  ```
1. View tensorboardX
  ```
  tensorboard --logdir ./logs
  ```

## TODO

New things will be added here.

- [x] implement ER, BA, WS graph generation
- [x] implement the model
- [x] write training/logging code with TensorboardX
- [x] download ImageNet dataset and implement `dataloader.py`
- [ ] train the network
- [ ] estimate appropriate batch size for specific GPU
- [ ] write results here

I plan to use Adam optimizer, instead of [Distributed SGD](https://arxiv.org/abs/1706.02677).

## Author

Seungwon Park / [@seungwonpark](http://swpark.me)

## License

Apache License 2.0

This repository contains codes adapted/copied from the followings:
- [utils/adabound.py](./utils/adabound.py) from https://github.com/Luolc/AdaBound (Apache License 2.0)
- [utils/hparams.py](./utils/hparams.py) from https://github.com/HarryVolek/PyTorch_Speaker_Verification (No License specified)
