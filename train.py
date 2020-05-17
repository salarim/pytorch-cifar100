# train.py
#!/usr/bin/env	python3

""" train network using pytorch

author baiyu
"""

import os
import sys
import time
import argparse
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms

from torch.utils.data import DataLoader
#from dataset import *
from torch.autograd import Variable

from conf import settings
from utils import get_network, get_training_dataloader, get_test_dataloader
from models.myresnet import resnet
from data_utils import DataConfig, DataLoaderConstructor

def train(epoch):

    net.train()

    batch_time = AverageMeter()
    data_time = AverageMeter()
    end = time.time()

    for batch_index, (images, labels) in enumerate(cifar100_training_loader):
        data_time.update(time.time() - end)

        images = Variable(images)
        labels = Variable(labels)

        labels = labels.cuda()
        images = images.cuda()

        optimizer.zero_grad()
        outputs = net(images)
        loss = loss_function(outputs, labels)
        loss.backward()
        optimizer.step()

        n_iter = (epoch - 1) * len(cifar100_training_loader) + batch_index + 1

        last_layer = list(net.children())[-1]

        batch_time.update(time.time() - end)
        end = time.time()

        print('Training Epoch: {epoch} [{trained_samples}/{total_samples}]\t'
            'DTime {data_time.avg:.3f}\t'
            'BTime {batch_time.avg:.3f}\t'
            'Loss: {:0.4f}\tLR: {:0.6f}'.format(
            loss.item(),
            optimizer.param_groups[0]['lr'],
            epoch=epoch,
            trained_samples=batch_index * args.b + len(images),
            total_samples=len(cifar100_training_loader.dataset),
            batch_time=batch_time, data_time=data_time
        ))


def eval_training(epoch):
    net.eval()

    test_loss = 0.0 # cost function error
    correct = 0.0

    for (images, labels) in cifar100_test_loader:
        images = Variable(images)
        labels = Variable(labels)

        images = images.cuda()
        labels = labels.cuda()

        outputs = net(images)
        loss = loss_function(outputs, labels)
        test_loss += loss.item()
        _, preds = outputs.max(1)
        correct += preds.eq(labels).sum()

    print('Test set: Average loss: {:.4f}, Accuracy: {:.4f}'.format(
        test_loss / len(cifar100_test_loader.dataset),
        correct.float() / len(cifar100_test_loader.dataset)
    ))
    print()

    return correct.float() / len(cifar100_test_loader.dataset)


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument('-net', type=str, required=True, help='net type')
    parser.add_argument('-gpu', type=bool, default=True, help='use gpu or not')
    parser.add_argument('-w', type=int, default=2, help='number of workers for dataloader')
    parser.add_argument('-b', type=int, default=128, help='batch size for dataloader')
    parser.add_argument('-s', type=bool, default=True, help='whether shuffle the dataset')
    parser.add_argument('-lr', type=float, default=0.1, help='initial learning rate')

    parser.add_argument('--new-model', action='store_true', default=False)
    parser.add_argument('--new-optim', action='store_true', default=False)
    parser.add_argument('--new-normalize', action='store_true', default=False)
    parser.add_argument('--disable-rotate', action='store_true', default=False)
    parser.add_argument('--new-data-loader', action='store_true', default=False)

    args = parser.parse_args()
    print(args)

    if args.new_model:
        net = resnet(depth=20, num_classes=100)
        if args.gpu:
            net = net.cuda()
    else:
        net = get_network(args, use_gpu=args.gpu)
    
    if args.new_optim:
        MILESTONES = [100, 150]
        gamma = 0.1
        weight_decay = 1e-4
    else:
        MILESTONES = [60, 120, 160]
        gamma = 0.2
        weight_decay = 5e-4

    if args.new_normalize:
        CIFAR100_TRAIN_MEAN = (0.4914, 0.4822, 0.4465)
        CIFAR100_TRAIN_STD = (0.2023, 0.1994, 0.2010)
    else:
        CIFAR100_TRAIN_MEAN = (0.5070751592371323, 0.48654887331495095, 0.4409178433670343)
        CIFAR100_TRAIN_STD = (0.2673342858792401, 0.2564384629170883, 0.27615047132568404)

    #data preprocessing:
    if args.new_data_loader:
        data_config = DataConfig(train=True, dataset='cifar100',
                                 dataset_type='softmax', is_continual=True, 
                                 batch_size=args.b, workers=args.w,  tasks=1, 
                                 exemplar_size=0, oversample_ratio=0.0)
        cifar100_training_loader = DataLoaderConstructor(data_config).data_loaders[0]
        data_config.train = False
        cifar100_test_loader = DataLoaderConstructor(data_config).data_loaders[0]
    else:
        cifar100_training_loader = get_training_dataloader(
            CIFAR100_TRAIN_MEAN,
            CIFAR100_TRAIN_STD,
            num_workers=args.w,
            batch_size=args.b,
            shuffle=args.s,
            disable_rotate=args.disable_rotate
        )
        
        cifar100_test_loader = get_test_dataloader(
            CIFAR100_TRAIN_MEAN,
            CIFAR100_TRAIN_STD,
            num_workers=args.w,
            batch_size=args.b,
            shuffle=args.s
        )
    
    loss_function = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=0.9, weight_decay=weight_decay)
    train_scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=MILESTONES, gamma=gamma) #learning rate decay
    iter_per_epoch = len(cifar100_training_loader)
    checkpoint_path = os.path.join(settings.CHECKPOINT_PATH, args.net, settings.TIME_NOW)

    #use tensorboard
    if not os.path.exists(settings.LOG_DIR):
        os.mkdir(settings.LOG_DIR)

    #create checkpoint folder to save model
    if not os.path.exists(checkpoint_path):
        os.makedirs(checkpoint_path)
    checkpoint_path = os.path.join(checkpoint_path, '{net}-{epoch}-{type}.pth')

    best_acc = 0.0
    for epoch in range(1, settings.EPOCH):

        train(epoch)
        acc = eval_training(epoch)

        train_scheduler.step(epoch)

        #start to save best performance model after learning rate decay to 0.01 
        if epoch > MILESTONES[1] and best_acc < acc:
            torch.save(net.state_dict(), checkpoint_path.format(net=args.net, epoch=epoch, type='best'))
            best_acc = acc
            continue

        if not epoch % settings.SAVE_EPOCH:
            torch.save(net.state_dict(), checkpoint_path.format(net=args.net, epoch=epoch, type='regular'))

