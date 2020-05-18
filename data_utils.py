import math
import numpy as np
from PIL import Image

import torch
import torchvision


class DataConfig:

    def __init__(self, train, dataset, dataset_type, is_continual, batch_size, 
                 workers,  tasks, exemplar_size, oversample_ratio):
        
        self.train = train
        self.dataset = dataset
        self.dataset_type = dataset_type
        self.is_continual = is_continual
        self.batch_size = batch_size
        self.workers = workers
        self.tasks = tasks
        self.exemplar_size = exemplar_size
        self.oversample_ratio = oversample_ratio


class DataLoaderConstructor:

    def __init__(self, config):
        self.config = config

        original_data, original_targets = self.get_data_targets(self.config.dataset)
        transforms = self.get_transforms(self.config.dataset)

        self.tasks_targets = [np.unique(original_targets)]
        indexes = [np.random.permutation(original_data.shape[0])]
        
        self.data_loaders = self.create_dataloaders(original_data, original_targets,
                                                    indexes, transforms)

    def get_data_targets(self, dataset_name):
        if dataset_name == 'cifar100':
            dataset = torchvision.datasets.CIFAR100('./data',
                                                     train=self.config.train, download=True)
            data, targets = dataset.data, dataset.targets
        else:
            raise ValueError('dataset is not supported.')
            
        # if torch.is_tensor(targets):
        #     data = data.numpy()
        #     targets = targets.numpy()
        # elif type(targets) == list:
        #     data = np.array(data)
        #     targets = np.array(targets)

        return data, targets

    def get_transforms(self, dataset_name):
        means = {
            'mnist':(0.1307,),
            'cifar10':(0.485, 0.456, 0.406),
            'cifar100':(0.5070751592371323, 0.48654887331495095, 0.4409178433670343),
            'imagenet':(0.485, 0.456, 0.406)
        }
        stds = {
            'mnist':(0.3081,),
            'cifar10':(0.229, 0.224, 0.225),
            'cifar100':(0.2673342858792401, 0.2564384629170883, 0.27615047132568404),
            'imagenet':(0.229, 0.224, 0.225)
        }

        transforms = []
        if dataset_name in ['cifar10', 'cifar100', 'imagenet'] and self.config.train:
            transforms.extend([torchvision.transforms.RandomCrop(32, padding=4),
                                torchvision.transforms.RandomHorizontalFlip()])
        transforms.extend([torchvision.transforms.ToTensor(),
                            torchvision.transforms.Normalize(means[dataset_name],
                                                             stds[dataset_name])])
        return torchvision.transforms.Compose(transforms)

    def create_dataloaders(self, data, targets, indexes, transforms):
        data_loaders = []

        for task_indexes in indexes:
            if self.config.dataset_type == 'softmax':
                dataset = SimpleDataset(data, targets, transform=transforms)
                # dataset = torchvision.datasets.CIFAR100('./data', train=self.config.train, 
                #                                         download=True, transform=transforms)
            
            kwargs = {'num_workers': self.config.workers, 'pin_memory': True} if \
                torch.cuda.device_count() > 0 else {}
            data_loader = torch.utils.data.DataLoader(
                dataset, batch_size=self.config.batch_size, shuffle=True, **kwargs)
            data_loaders.append(data_loader)

        return data_loaders


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
