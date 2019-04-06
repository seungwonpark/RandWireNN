# I followed ImageNet data loading convention shown in:
# https://github.com/pytorch/examples/blob/master/imagenet/main.py

import os
import torch
import torchvision.transforms as transforms
from torchvision import datasets


def create_dataloader(hp, args, train):
	normalize = transforms.Normalize(
		mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

	if train:
		return torch.utils.data.DataLoader(
				datasets.ImageFolder(hp.data.train, transforms.Compose([
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
					transforms.ToTensor(),
					normalize,
				])),
				batch_size=hp.data.batch_size,
				num_workers=hp.data.num_workers,
				shuffle=False, pin_memory=True)
