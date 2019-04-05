import os
import time
import torch
import logging
import argparse

from utils.train import train
from utils.hparams import HParam
from utils.writer import MyWriter
from dataset.dataloader import create_dataloader


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
    
    if hp.data.train.dir == '' or hp.data.test.dir == '':
        logger.error("train_dir, test_dir cannot be empty")
        raise Exception("Please specify directories of train data.")

    writer = MyWriter(log_dir)
    
    trainset = create_dataloader(hp, args, True)
    valset = create_dataloader(hp, args, False)
    train(out_dir, chkpt_path, trainset, valset, writer, logger, hp, hp_str)
