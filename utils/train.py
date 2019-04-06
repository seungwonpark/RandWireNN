import os
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import traceback

from utils.adabound import AdaBound
from utils.hparams import load_hparam_str
from utils.evaluation import validate
from model.model import RandWire


def train(out_dir, chkpt_path, trainset, valset, writer, logger, hp, hp_str, graphs):
    model = RandWire(hp, graphs).cuda()
    logger.info("Writing graph to tensorboardX...")
    writer.write_graph(model, torch.randn(7, 3, 224, 224).cuda())
    logger.info("Finished.")

    if hp.train.optimizer == 'adam':
        optimizer = torch.optim.Adam(model.parameters(),
                                     lr=hp.train.adam)
    elif hp.train.optimizer == 'adabound':
        optimizer = AdaBound(model.parameters(),
                             lr=hp.train.adabound.initial,
                             final_lr=hp.train.adabound.final)
    else:
        raise Exception("Optimizer not supported: %s" % hp.train.optimizer)

    step = 0

    if chkpt_path is not None:
        logger.info("Resuming from checkpoint: %s" % chkpt_path)
        checkpoint = torch.load(chkpt_path)
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        step = checkpoint['step']

        if hp_str != checkpoint['hp_str']:
            logger.warning("New hparams are different from checkpoint.")
            logger.warning("Will use new hparams.")
        # hp = load_hparam_str(hp_str)
    else:
        logger.info("Starting new training run")

    try:
        model.train()
        while True:
            for data, target in trainset:
                data, target = data.cuda(), target.cuda()
                optimizer.zero_grad()
                output = model(data)
                loss = F.nll_loss(output, target)
                loss.backward()
                optimizer.step()
                
                step += 1
                loss = loss.item()
                if loss > 1e8 or math.isnan(loss):
                    logger.error("Loss exploded to %.02f at step %d!" % (loss, step))
                    raise Exception("Loss exploded")

                if step % hp.train.summary_interval == 0:
                    writer.log_training(loss, step)
                    logger.info("Wrote summary at step %d" % step)

                if step % hp.train.checkpoint_interval == 0:
                    save_path = os.path.join(out_dir, 'chkpt_%07d.pt' % step)
                    torch.save({
                        'model': model.state_dict(),
                        'optimizer': optimizer.state_dict(),
                        'step': step,
                        'hp_str': hp_str,
                    }, save_path)
                    logger.info("Saved checkpoint to: %s" % save_path)

                if step % hp.train.evaluation_interval == 0:
                    test_loss, accuracy = validate(model, valset, writer, step)
                    logger.info("Evaluation saved at step %d | test_loss: %.5f | accuracy: %.4f"
                                    % (step, test_loss, accuracy))
    except Exception as e:
        logger.info("Exiting due to exception: %s" % e)
        traceback.print_exc()
