import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F


def validate(model, valset, writer, step):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for idx, (data, target) in tqdm.tqdm(enumerate(valset), total=10):
            data, target = data.cuda(), target.cuda()
            output = model(data)
            test_loss += F.nll_loss(output, target).item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
            if idx == 10 - 1:
                break

    test_loss /= (10 * valset.batch_size)
    accuracy = correct / (10 * valset.batch_size)

    writer.log_evaluation(test_loss, accuracy, step)
    
    model.train()
    return test_loss, accuracy
