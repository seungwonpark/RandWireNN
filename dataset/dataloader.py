# I followed ImageNet data loading convention shown in:
# https://github.com/pytorch/examples/blob/master/imagenet/main.py

import os
import numpy as np
import torch
import torchvision.transforms as transforms
from torchvision import datasets


def create_dataloader(hp, args, train):
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    if train:
        return torch.utils.data.DataLoader(
                datasets.ImageFolder(hp.data.train, transforms.Compose([
                    transforms.RandomResizedCrop(224),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    normalize,
                ])),
                batch_size=hp.data.batch_size, 
                num_workers=hp.data.num_workers,
                shuffle=True, pin_memory=True, drop_last=True)
    else:
        return torch.utils.data.DataLoader(
                datasets.ImageFolder(hp.data.val, transforms.Compose([
                    transforms.Resize(256),
                    transforms.CenterCrop(224),
                    transforms.ToTensor(),
                    normalize,
                ])),
                batch_size=hp.data.batch_size//8,
                num_workers=hp.data.num_workers,
                shuffle=False, pin_memory=True)

# MNIST data loading
    
def MNIST_dataloader(bs, download=True):
    '''
    :bs: int
        batch size of train and test dataloaders
    :download: bool
        whether to download a new copy of MNIST to ./data
    '''
    root = './data'
    if not os.path.exists(root):
        os.mkdir(root)
        
    transf = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (1.0,))])

    train_set = datasets.MNIST(root=root, train=True, transform=transf, download=download)
    test_set = datasets.MNIST(root=root, train=False, transform=transf, download=download)
    
    batch_size = len(train_set)
    
    trainloader = torch.utils.data.DataLoader(
                     dataset=train_set,
                     batch_size=bs,
                     shuffle=True)
    testloader = torch.utils.data.DataLoader(
                    dataset=test_set,
                    batch_size=bs,
                    shuffle=False)
    
    return trainloader, testloader

# CIFAR10 data loading 

def CIFAR10_dataloader(path, bs):
    '''
    :path: str
        path to raw CIFAR10, as found in https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz
    :bs: int
        batch size
    '''
    data1 = unpickle( f'{path}\data_batch_1')
    data2 = unpickle( f'{path}\data_batch_2')
    data3 = unpickle( f'{path}\data_batch_3')
    data4 = unpickle( f'{path}\data_batch_4')
    data5 = unpickle( f'{path}\data_batch_5')
    test = unpickle( f'{path}\\test_batch')
    
    ds = []
    dlabels = []
    test_ds = []
    test_dlabels = []

    for i in range(10000):
        im = np.reshape(data1[b'data'][i],(3, 32, 32))
        ds.append(im)
        dlabels.append(data1[b'labels'][i])
    for i in range(10000):
        im = np.reshape(data2[b'data'][i],(3, 32, 32))
        ds.append(im)
        dlabels.append(data2[b'labels'][i])
    for i in range(10000):
        im = np.reshape(data3[b'data'][i],(3, 32, 32))
        ds.append(im)
        dlabels.append(data3[b'labels'][i])
    for i in range(10000):
        im = np.reshape(data4[b'data'][i],(3, 32, 32))
        ds.append(im)
        dlabels.append(data4[b'labels'][i])
    for i in range(10000):
        im = np.reshape(data5[b'data'][i],(3, 32, 32))
        ds.append(im)
        dlabels.append(data5[b'labels'][i])
    for i in range(10000):
        im = np.reshape(test[b'data'][i],(3, 32, 32))
        test_ds.append(im)
        test_dlabels.append(test[b'labels'][i])
    
    train = torch.utils.data.TensorDataset(torch.Tensor(ds), torch.LongTensor(dlabels))
    test = torch.utils.data.TensorDataset(torch.Tensor(test_ds), torch.LongTensor(test_dlabels))

    trainloader = torch.utils.data.DataLoader(train, batch_size = bs)
    testloader = torch.utils.data.DataLoader(test, batch_size = bs)
    
    return trainloader, testloader

def unpickle(file):
    import pickle
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict