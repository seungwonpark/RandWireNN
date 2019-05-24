import os
import math
import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
import adabound
import itertools
import traceback

from utils.hparams import load_hparam_str
from utils.evaluation import validate
from model.model import RandWire


def train(out_dir, chkpt_path, trainset, valset, writer, logger, hp, hp_str, graphs):
    model = RandWire(hp, graphs).cuda()

    if hp.train.optimizer == 'adam':
        optimizer = torch.optim.Adam(model.parameters(),
                                     lr=hp.train.adam)
    elif hp.train.optimizer == 'adabound':
        optimizer = adabound.AdaBound(model.parameters(),
                             lr=hp.train.adabound.initial,
                             final_lr=hp.train.adabound.final)
    elif hp.train.optimizer == 'sgd':
        optimizer = torch.optim.SGD(model.parameters(),
                                    lr=hp.train.sgd.lr,
                                    momentum=hp.train.sgd.momentum,
                                    weight_decay=hp.train.sgd.weight_decay)
    else:
        raise Exception("Optimizer not supported: %s" % hp.train.optimizer)

    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, hp.train.epoch)

    init_epoch = -1
    step = 0

    if chkpt_path is not None:
        logger.info("Resuming from checkpoint: %s" % chkpt_path)
        checkpoint = torch.load(chkpt_path)
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
        step = checkpoint['step']
        init_epoch = checkpoint['epoch']

        if hp_str != checkpoint['hp_str']:
            logger.warning("New hparams are different from checkpoint.")
            logger.warning("Will use new hparams.")
        # hp = load_hparam_str(hp_str)
    else:
        logger.info("Starting new training run")
        logger.info("Writing graph to tensorboardX...")
        writer.write_graph(model, torch.randn(7, hp.model.input_maps, 224, 224).cuda())
        logger.info("Finished.")

    try:
        model.train()
        for epoch in itertools.count(init_epoch+1):
            loader = tqdm.tqdm(trainset, desc='Train data loader')
            for data, target in loader:
                data, target = data.cuda(), target.cuda()
                optimizer.zero_grad()
                output = model(data)
                loss = F.nll_loss(output, target)
                loss.backward()
                optimizer.step()
                
                loss = loss.item()
                if loss > 1e8 or math.isnan(loss):
                    logger.error("Loss exploded to %.02f at step %d!" % (loss, step))
                    raise Exception("Loss exploded")

                writer.log_training(loss, step)
                loader.set_description('Loss %.02f at step %d' % (loss, step))
                step += 1                

            save_path = os.path.join(out_dir, 'chkpt_%03d.pt' % epoch)
            torch.save({
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'lr_scheduler': lr_scheduler.state_dict(),
                'step': step,
                'epoch': epoch,
                'hp_str': hp_str,
            }, save_path)
            logger.info("Saved checkpoint to: %s" % save_path)

            validate(model, valset, writer, epoch)
            lr_scheduler.step()

    except Exception as e:
        logger.info("Exiting due to exception: %s" % e)
        traceback.print_exc()
