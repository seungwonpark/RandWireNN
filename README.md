# RandWireNN

[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/exploring-randomly-wired-neural-networks-for/image-classification-imagenet-image-reco)](https://paperswithcode.com/sota/image-classification-imagenet-image-reco?p=exploring-randomly-wired-neural-networks-for)

Unofficial PyTorch Implementation of:
[Exploring Randomly Wired Neural Networks for Image Recognition](https://arxiv.org/abs/1904.01569).

![](./assets/teaser.png)

## Results

**Currently, training is in progress.**

Validation result on Imagenet(ILSVRC2012) dataset:

| Top 1 accuracy (%)         | Paper | Here(WIP) |
| -------------------------- | ----- | --------- |
| RandWire-WS(4, 0.75), C=78 | 74.7  | 63.0      |


- (Currently trying Adabound optimizer)
- (2019.04.10) 63.0%: 450k steps with Adam optimizer, initial lr 0.001, lr decay about 0.1 for every 150k step
- (2019.04.07) 56.8%: Training took about 16 hours on AWS p3.2xlarge(NVIDIA V100). 120k steps were done in total, and Adam optimizer with `lr=0.001, batch_size=128` was used with no learning rate decay.
![](./assets/56dot8percent.png)
- I plan to try:
  - Learning rate decay for every 150k step, by 0.1
  - [AdaBound optimizer](https://github.com/Luolc/AdaBound)
  - TODO: implement [Distributed SGD](https://arxiv.org/abs/1706.02677) like [JiaminRen's implementation](https://github.com/JiaminRen/RandWireNN)
  - Any Pull Request reporting result with other training strategy will be welcome. Please let me know in prior if you're planning to do so.

## Dependencies

This code was tested on Python 3.6 with PyTorch 1.0.1. Other packages can be installed by:
```bash
pip install -r requirements.txt
```

## Generate random DAG

```bash
cd model/graphs
python er.py -p 0.2 -o er-02.txt # Erdos-Renyi
python ba.py -m 7 -o ba-7.txt # Barbasi-Albert
python ws.py -k 4 -p 0.75 ws-4-075.txt # Watts-Strogatz
# number of nodes: -n option
```

All outputs from commands shown above will produce txt file like:
```
(number of nodes)
(number of edges)
(lines, each line representing edges)
```

## Train RandWireNN

1. Download ImageNet dataset. Train/val folder should contain list of 1,000 directories, each containing list of images for corresponding category. For validation image files, this script can be useful: https://raw.githubusercontent.com/soumith/imagenetloader.torch/master/valprep.sh
1. Edit `config.yaml`
    ```bash
    cd config
    cp default.yaml config.yaml
    vim config.yaml # specify data directory, graph txt files
    ```
1. Train

    *Note.* Validation performed here won't use entire test set, since it will consume much time. (about 3 min.)
    ```
    python trainer.py -c [config yaml] -m [name]
    ```
1. View tensorboardX
    ```
    tensorboard --logdir ./logs
    ```

## Validation

Run full validation:

```bash
python validation.py -c [config path] -p [checkpoint path]
```

This will show accuracy and average test loss of the trained model.


## Author

Seungwon Park / [@seungwonpark](http://swpark.me)

## License

Apache License 2.0
