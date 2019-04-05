import torch
import torch.nn as nn
import torch.nn.functional as F


def validate(model, valset, writer, step):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for batch in valset:
            batch = batch.cuda()
            data, target = batch
            output = model(data)
            test_loss += F.nll_loss(output, target).item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(valset.dataset)
    accuracy = 100.0 * correct / len(valset.dataset)

    writer.log_evaluation(test_loss, accuracy, step)
    
    model.train()
    return test_loss, accuracy
