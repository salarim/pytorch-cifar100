import math
import numpy as np
from PIL import Image

import torch
import torchvision


class DataLoaderConstructor:

    def __init__(self, train, batch_size, workers):
        self.train = train
        self.batch_size = batch_size
        self.workers = workers

        original_data, original_targets = self.get_data_targets()

        transforms = self.get_transforms()
        
        self.data_loader = self.create_dataloader(original_data, original_targets, transforms)

    def get_data_targets(self):
        dataset = torchvision.datasets.CIFAR100('./data',
                                                train=self.train, download=True)
        data, targets = dataset.data, dataset.targets

        return data, targets

    def get_transforms(self):
        mean = (0.5070751592371323, 0.48654887331495095, 0.4409178433670343)
        std = (0.2673342858792401, 0.2564384629170883, 0.27615047132568404)

        transforms = []
        if self.train:
            transforms.extend([torchvision.transforms.RandomCrop(32, padding=4),
                                torchvision.transforms.RandomHorizontalFlip()])
        transforms.extend([torchvision.transforms.ToTensor(),
                            torchvision.transforms.Normalize(mean, std)])
        return torchvision.transforms.Compose(transforms)

    def create_dataloader(self, data, targets, transforms):

        dataset = SimpleDataset(data, targets, transform=transforms)
        # dataset = torchvision.datasets.CIFAR100('./data', train=self.train, 
        #                                         download=True, transform=transforms)
        
        kwargs = {'num_workers': self.workers, 'pin_memory': True} if \
            torch.cuda.device_count() > 0 else {}
        data_loader = torch.utils.data.DataLoader(
            dataset, batch_size=self.batch_size, shuffle=True, **kwargs)

        return data_loader


class SimpleDataset(torch.utils.data.Dataset):

    def __init__(self, data, targets, transform=None):
        self.data = data
        self.targets = targets
        self.transform = transform
        
    def __len__(self):
        return len(self.targets)

    def __getitem__(self, idx):
        img, target = self.data[idx], self.targets[idx]
        img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)

        return img, target
