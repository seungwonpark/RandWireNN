import os
import time
import torch
import logging
import argparse

from utils.train import train
from utils.hparams import HParam
from utils.writer import MyWriter
from utils.graph_reader import read_graph
from dataset.dataloader import create_dataloader, MNIST_dataloader, CIFAR10_dataloader


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', type=str, required=True,
                        help="yaml file for configuration")
    parser.add_argument('-p', '--checkpoint_path', type=str, default=None, required=False,
                        help="path of checkpoint pt file")
    parser.add_argument('-m', '--model', type=str, required=True,
                        help="name of the model. used for logging/saving checkpoints")
    args = parser.parse_args()

    hp = HParam(args.config)
    with open(args.config, 'r') as f:
        hp_str = ''.join(f.readlines())

    pt_path = os.path.join('.', hp.log.chkpt_dir)
    out_dir = os.path.join(pt_path, args.model)
    os.makedirs(out_dir, exist_ok=True)

    log_dir = os.path.join('.', hp.log.log_dir)
    log_dir = os.path.join(log_dir, args.model)
    os.makedirs(log_dir, exist_ok=True)

    if args.checkpoint_path is not None:
        chkpt_path = args.checkpoint_path
    else:
        chkpt_path = None

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(os.path.join(log_dir,
                '%s-%d.log' % (args.model, time.time()))),
            logging.StreamHandler()
        ]
    )
    logger = logging.getLogger()
    
    if hp.data.train == '' or hp.data.val == '':
        logger.error("hp.data.train, hp.data.val cannot be empty")
        raise Exception("Please specify directories of train data.")

    if hp.model.graph0 == '' or hp.model.graph1 == '' or hp.model.graph2 == '':
        logger.error("hp.model.graph0, graph1, graph2 cannot be empty")
        raise Exception("Please specify random DAG architecture.")

    graphs = [
        read_graph(hp.model.graph0),
        read_graph(hp.model.graph1),
        read_graph(hp.model.graph2),
    ]

    writer = MyWriter(log_dir)
    
    dataset = hp.data.type
    switcher = {
            'MNIST': MNIST_dataloader,
            'CIFAR10':CIFAR10_dataloader,
            'ImageNet':create_dataloader,
            }
    assert dataset in switcher.keys(), 'Dataset type currently not supported'
    dl_func = switcher[dataset]
    trainset = dl_func(hp, args, True)
    valset = dl_func(hp, args, False)

    train(out_dir, chkpt_path, trainset, valset, writer, logger, hp, hp_str, graphs)
