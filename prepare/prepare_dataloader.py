from os.path import join

import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


def prepare_dataloader(args):
    valsplit = args.valsplit
    data_dir = args.data_dir
    batch_size = args.batch_size

    # normalize
    if args.dataset == 'SVHN':
        normalize = transforms.Normalize(mean=[x / 255.0 for x in [109.9, 109.7, 113.8]],
                                         std=[x / 255.0 for x in [50.1, 50.6, 50.8]])
    elif args.dataset in ['IMAGENET', 'IMAGENETSUBSET']:
        # valdir = (args.data, 'val')
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])

    elif args.dataset == 'CIFAR10':
        normalize = transforms.Normalize(mean=[x / 255.0 for x in [125.3, 123.0, 113.9]],
                                         std=[x / 255.0 for x in [63.0, 62.1, 66.7]])
    elif args.dataset == 'MNIST':
        normalize = transforms.Normalize((0.1307,), (0.3081,))
    else:
        raise ValueError('no such dataset:'+args.dataset)

    preprocess = [transforms.ToTensor(), normalize]
    train_transformers = []
    test_transformers = []

    # augment
    if args.data_augmentation:
        if args.dataset in ['CIFAR10', 'CIFAR100', 'SVHN']:
            augment = [
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip()
            ]
        elif args.dataset in ['IMAGENET', 'IMAGENETSUBSET']:
            augment = [
                transforms.RandomResizedCrop(224),
                transforms.RandomHorizontalFlip(),
            ]
        train_transformers += augment

    train_transformers += preprocess

    train_transform = transforms.Compose(train_transformers)
    if args.dataset in ['IMAGENET', 'IMAGENETSUBSET']:
        test_transformers += [transforms.Resize((256, 256)),
                              transforms.CenterCrop(224)]
    test_transformers += preprocess
    test_transform = transforms.Compose(test_transformers)

    if args.dataset in ['IMAGENET', 'IMAGENETSUBSET']:
        fulldataset = datasets.ImageFolder(
            join(data_dir, 'train'), train_transform)
    else:
        fulldataset = datasets.__getattribute__(args.dataset)(
            data_dir, train=True, download=True, transform=train_transform)

        if args.dataset == 'SVHN':
            extra_dataset = datasets.SVHN(root=data_dir,
                                          split='extra',
                                          transform=train_transform,
                                          download=True)

            # Combine both training splits (https://arxiv.org/pdf/1605.07146.pdf)
            data = np.concatenate(
                [fulldataset.data, extra_dataset.data], axis=0)
            labels = np.concatenate(
                [fulldataset.labels, extra_dataset.labels], axis=0)
            fulldataset.data = data
            fulldataset.labels = labels

    fullsize = len(fulldataset)
    valsize = int(valsplit * fullsize)
    trainsize = fullsize - valsize
    traindataset, valdataset = torch.utils.data.random_split(
        fulldataset, [trainsize, valsize])
    traindataset.transform = train_transform
    valdataset.transform = test_transform

    trainloader = DataLoader(traindataset, batch_size=batch_size, shuffle=True,
                             num_workers=2, pin_memory=True)
    valloader = DataLoader(valdataset, batch_size=batch_size, shuffle=False,
                           num_workers=2, pin_memory=True)

    if args.dataset in ['IMAGENET', 'IMAGENETSUBSET']:
        testdataset = datasets.ImageFolder(
            join(data_dir, 'val'), transform=test_transform)
    else:
        testdataset = datasets.__getattribute__(args.dataset)(
            data_dir, train=False, transform=test_transform)

    testloader = torch.utils.data.DataLoader(
        testdataset, batch_size, shuffle=False, num_workers=2, pin_memory=True)

    # print(trainloader.dataset.transform)
    return trainloader, valloader, testloader
