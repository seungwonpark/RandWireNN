import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F


def validate(model, valset, writer=None, epoch=None):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for idx, (data, target) in tqdm.tqdm(enumerate(valset)):
            data, target = data.cuda(), target.cuda()
            output = model(data)
            test_loss += F.nll_loss(output, target).item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(valset.dataset)
    accuracy = correct / len(valset.dataset)

    if writer is not None:
        writer.log_evaluation(test_loss, accuracy, epoch)
    
    model.train()
    return test_loss, accuracy
